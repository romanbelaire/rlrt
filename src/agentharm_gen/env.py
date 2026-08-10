"""AgentHarm-Gen environment: 3-layer tool red-team training."""

from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

from inspect_ai.dataset import Sample
from inspect_ai.model import Model

from src.agentharm.defender_model import is_openai_defender, load_defender_model
from src.agentharm.message_bridge import EpisodeState, conversation_to_agent_state
from src.agentharm_gen.dataset import (
    dataset_to_list,
    load_agentharm_gen_dataset,
    sample_to_metadata,
)
from src.agentharm_gen.defender_step import run_defender_turn_sync
from src.agentharm_gen.scoring import AgentHarmGenScorer
from src.agentharm_gen.step_scoring import (
    compute_step_rewards,
    update_tools_called_so_far,
)
from src.base import Conversation, Environment


class AgentHarmGenBatchDataset(TorchDataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class AgentHarmGenEnvironment(Environment):
    """Red-team environment on AgentHarm-Gen generated behaviors."""

    pre_step_rejection_scoring = False
    uses_episode_harm_reward = False
    uses_tool_step_rewards = True

    def __init__(
        self,
        batch_size: int,
        max_steps: int,
        target_model_name: str,
        options: dict[str, Any],
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        debug_mode: bool = False,
        evaluation_mode: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.target_model_name = target_model_name
        self.options = options
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.debug_mode = debug_mode

        tasks_path = options.get("tasks_path")
        dataset = load_agentharm_gen_dataset(tasks_path)
        self._all_samples = dataset_to_list(dataset)
        self._sample_index = 0

        self.defender_tool_choice = options.get("defender_tool_choice", "auto")
        self.defender_system_prompt_template = options.get("defender_system_prompt_template")
        self.defender_prompt_technique = options.get("defender_prompt_technique", "standard")
        self.max_defender_tool_rounds = int(options.get("max_defender_tool_rounds", 20))
        self.harm_threshold = float(options.get("harm_threshold", 0.5))
        self.refusal_penalty = float(options.get("refusal_penalty", 0.0))

        self._defender_model: Model = load_defender_model(
            target_model_name, defender_temperature=self.temperature
        )
        self._embedding_layer, self._embedding_tokenizer = self._init_embedding_backend(
            target_model_name
        )

        self.scorer = AgentHarmGenScorer(
            refusal_judge_model=options.get("refusal_judge", "openai/gpt-4o-2024-08-06"),
            semantic_judge_model=options.get("semantic_judge", "openai/gpt-4o-2024-08-06"),
            refusal_penalty=self.refusal_penalty,
        )

        self.torch_dataset = AgentHarmGenBatchDataset(self._all_samples)
        self.dataloader = DataLoader(
            self.torch_dataset,
            batch_size=batch_size,
            shuffle=not evaluation_mode,
            collate_fn=lambda batch: batch,
        )
        self.dataloader_iter = iter(self.dataloader)

        self.conversations: List[Conversation] = []
        self.episodes: List[EpisodeState] = []
        self.state: List[List[str]] = []
        self.base_state: List[str] = []
        self.current_step = 0
        self.dones: Optional[torch.Tensor] = None
        self._last_l3_rewards = torch.zeros(batch_size)

        print(
            f"[AgentHarmGenEnvironment] tasks={len(self._all_samples)} "
            f"batch_size={batch_size}"
        )

    def _init_embedding_backend(self, target_model_name: str):
        if is_openai_defender(target_model_name):
            return None, None
        api = self._defender_model.api
        hf_model = api.model
        tokenizer = api.tokenizer
        return hf_model.get_input_embeddings(), tokenizer

    def _get_embed_similarity_batched(
        self,
        sequences1: List[str],
        sequences2: List[str],
    ) -> torch.Tensor:
        if self._embedding_layer is None:
            raise ValueError(
                "AgentHarm-Gen instruction-following similarity requires a local HF defender model "
                f"(current defender: {self.target_model_name!r})"
            )
        if len(sequences1) != len(sequences2):
            raise ValueError(
                f"sequences1 length {len(sequences1)} != sequences2 length {len(sequences2)}"
            )

        embedding_layer = self._embedding_layer
        tokenizer = self._embedding_tokenizer
        device = embedding_layer.weight.device

        max_len = 512
        enc1 = tokenizer(
            sequences1,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc2 = tokenizer(
            sequences2,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        enc1_input_ids = enc1.input_ids.int().to(device)
        enc1_attention_mask = enc1.attention_mask.to(device)
        enc2_input_ids = enc2.input_ids.int().to(device)
        enc2_attention_mask = enc2.attention_mask.to(device)

        emb1 = embedding_layer(enc1_input_ids)
        emb2 = embedding_layer(enc2_input_ids)

        def _mean_pool(emb, mask):
            masked_emb = emb * mask.unsqueeze(-1)
            summed = masked_emb.sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6).unsqueeze(-1)
            return summed / counts

        seq_emb1 = _mean_pool(emb1, enc1_attention_mask)
        seq_emb2 = _mean_pool(emb2, enc2_attention_mask)
        cosine_sim = F.cosine_similarity(seq_emb1, seq_emb2, dim=-1)
        return ((cosine_sim + 1.0) / 2.0).detach()

    def _next_batch_samples(self) -> list[Sample]:
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)
        if isinstance(batch, Sample):
            return [batch]
        return list(batch)

    def reset(self) -> List[List[str]]:
        self.current_step = 0
        self.dones = torch.zeros(self.batch_size)
        samples = self._next_batch_samples()

        self.conversations = []
        self.episodes = []
        self.base_state = []

        n_irrelevant = int(self.options.get("n_irrelevant_tools", 0))
        for i, sample in enumerate(samples):
            meta = sample_to_metadata(sample)
            meta["_n_irrelevant_tools"] = n_irrelevant
            prompt = meta["prompt"]
            conv = Conversation(f"agentharm_gen_{meta['id']}_{i}")
            conv.set_original_harmful_prompt(prompt)
            conv.set_metadata("batch_index", i)
            conv.set_metadata("behavior_id", meta["id"])
            conv.set_metadata("category", meta.get("category", ""))
            self.conversations.append(conv)
            self.base_state.append(prompt)

            episode = EpisodeState(
                behavior_id=str(meta["id"]),
                task_name="agentharm_gen",
                metadata=meta,
            )
            self.episodes.append(episode)

        self.state = self._episodes_to_state()
        return [list(s) for s in self.state]

    def _episodes_to_state(self) -> List[List[str]]:
        state = []
        for conv in self.conversations:
            inject = self.current_step == 0
            state.append(conversation_to_agent_state(conv, inject_goal=inject))
        return state

    def step(
        self,
        actions: Union[List[str], str],
        system_prompt: Optional[Union[List[str], str]] = None,
        template_ids: Optional[List[str]] = None,
        predicted_tools: Optional[List[str]] = None,
        **kwargs,
    ) -> tuple[List[List[str]], List[str], torch.Tensor, torch.Tensor]:
        if isinstance(actions, str):
            actions = [actions]
        if system_prompt is not None and isinstance(system_prompt, str):
            system_prompt = [system_prompt]
        if predicted_tools is not None and isinstance(predicted_tools, str):
            predicted_tools = [predicted_tools]
        high_level_instructions = system_prompt

        target_responses: List[str] = []
        l2_rewards: List[float] = []
        l3_rewards: List[float] = []

        for i, action in enumerate(actions):
            episode = self.episodes[i]
            conv = self.conversations[i]
            instruction = None
            if high_level_instructions and i < len(high_level_instructions):
                instruction = high_level_instructions[i]

            predicted_tool = ""
            if predicted_tools and i < len(predicted_tools):
                predicted_tool = predicted_tools[i]

            messages_before = list(episode.messages)
            episode.tools_called_before_step = list(episode.tools_called_so_far)
            episode.predicted_tool = predicted_tool

            n_irrelevant = int(episode.metadata.get("_n_irrelevant_tools", 0))
            run_defender_turn_sync(
                episode,
                action,
                self._defender_model,
                self.max_new_tokens,
                self.temperature,
                self.defender_tool_choice,
                self.defender_system_prompt_template,
                self.defender_prompt_technique,
                self.max_defender_tool_rounds,
                n_irrelevant,
            )

            response_text = episode.defender_response_text()
            target_responses.append(response_text)
            conv.add_user_message(
                action, high_level_instruction=instruction, step_number=self.current_step
            )
            conv.add_assistant_message(response_text)
            conv.set_metadata("agentharm_messages", len(episode.messages))

            harm_result = self.scorer.score_episode(episode)
            episode.last_harm_score_dict = harm_result.score_dict
            episode.tools_called_so_far = update_tools_called_so_far(
                episode.tools_called_so_far, episode.messages
            )

            step_result = compute_step_rewards(
                predicted_tool=predicted_tool,
                messages_before=messages_before,
                messages_after=episode.messages,
                refusal=harm_result.refusal,
            )
            episode.step_rewards = {
                "l2": step_result.l2_reward,
                "l3": step_result.l3_reward,
            }
            l2_rewards.append(step_result.l2_reward)
            l3_rewards.append(step_result.l3_reward)

        self.current_step += 1
        self.state = self._episodes_to_state()

        rewards = torch.tensor(l2_rewards, dtype=torch.float32)
        self._last_l3_rewards = torch.tensor(l3_rewards, dtype=torch.float32)

        self.dones = (
            torch.ones(len(actions))
            if self.current_step >= self.max_steps
            else torch.zeros(len(actions))
        )
        for i, episode in enumerate(self.episodes):
            if len(episode.tools_called_so_far) >= len(episode.metadata["target_functions"]):
                self.dones[i] = 1.0

        return self.state.copy(), target_responses, rewards, self.dones

    def score_safety(
        self,
        responses: List[str],
        questions: Optional[List[str]] = None,
        reference_state: Optional[List[str]] = None,
        include_white_box: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self.scorer.score_episodes(self.episodes)

    def get_layer_rewards(self) -> dict[str, torch.Tensor]:
        return {
            "l2": torch.tensor(
                [ep.step_rewards["l2"] for ep in self.episodes], dtype=torch.float32
            ),
            "l3": self._last_l3_rewards,
        }
