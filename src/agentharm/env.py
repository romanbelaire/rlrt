"""AgentHarm environment: attacker as user, defender as inspect_ai tool agent."""

from typing import Any, List, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

from inspect_ai.dataset import Sample
from inspect_ai.model import Model

from src.agentharm.dataset import dataset_to_list, load_agentharm_dataset, sample_to_metadata
from src.agentharm.defender_model import is_openai_defender, load_defender_model
from src.agentharm.defender_step import run_defender_turn_sync
from src.agentharm.message_bridge import EpisodeState, conversation_to_agent_state
from src.agentharm.scoring import AgentHarmScorer
from src.base import Conversation, Environment


class AgentHarmBatchDataset(TorchDataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


class AgentHarmEnvironment(Environment):
    """
    Red-team environment on AgentHarm behaviors.

    - Attacker (trainable policy) plays the user role.
    - Defender (target_model) is an inspect_ai tool-using assistant.
    - score_safety() returns official AgentHarm harm scores (higher = more harmful).
    """

    # Harm score needs a full defender tool loop; cannot score rejection candidates pre-step.
    pre_step_rejection_scoring = False
    # Use step() harm rewards for critic targets instead of LlamaGuard-style q_high.
    uses_episode_harm_reward = True

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

        task_name = options["task_name"]
        split = options["split"]
        if evaluation_mode and split == "val":
            split = options.get("eval_split", "test_public")

        self.task_name: Literal["harmful", "benign", "chat"] = task_name
        self.split = split
        dataset = load_agentharm_dataset(
            task_name=task_name,
            split=split,
            force_redownload=options.get("force_redownload", False),
            behavior_ids=options.get("behavior_ids") or None,
            detailed_behaviors=options.get("detailed_behaviors"),
            hint_included=options.get("hint_included"),
        )
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

        self.scorer = AgentHarmScorer(
            task_name=task_name,
            refusal_judge_model=options.get("refusal_judge", "openai/gpt-4o-2024-08-06"),
            semantic_judge_model=options.get("semantic_judge", "openai/gpt-4o-2024-08-06"),
            refusal_penalty=self.refusal_penalty,
        )

        self.torch_dataset = AgentHarmBatchDataset(self._all_samples)
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

        print(
            f"[AgentHarmEnvironment] task={task_name} split={split} "
            f"behaviors={len(self._all_samples)} batch_size={batch_size}"
        )

    def _init_embedding_backend(self, target_model_name: str):
        if is_openai_defender(target_model_name):
            return None, None
        api = self._defender_model.api
        hf_model = api.model
        tokenizer = api.tokenizer
        return hf_model.get_input_embeddings(), tokenizer

    def _conversations_to_state(self) -> List[List[str]]:
        return self._episodes_to_state()

    def _get_embed_similarity_batched(
        self,
        sequences1: List[str],
        sequences2: List[str],
    ) -> torch.Tensor:
        if self._embedding_layer is None:
            raise ValueError(
                "AgentHarm instruction-following similarity requires a local HF defender model "
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
            conv = Conversation(f"agentharm_{meta['id']}_{i}")
            conv.set_original_harmful_prompt(prompt)
            conv.set_metadata("batch_index", i)
            conv.set_metadata("behavior_id", meta["id"])
            conv.set_metadata("category", meta.get("category", ""))
            self.conversations.append(conv)
            self.base_state.append(prompt)

            episode = EpisodeState(
                behavior_id=str(meta["id"]),
                task_name=self.task_name,
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
        **kwargs,
    ) -> tuple[List[List[str]], List[str], torch.Tensor, torch.Tensor]:
        if isinstance(actions, str):
            actions = [actions]
        if system_prompt is not None and isinstance(system_prompt, str):
            system_prompt = [system_prompt]
        high_level_instructions = system_prompt

        target_responses: List[str] = []
        for i, action in enumerate(actions):
            episode = self.episodes[i]
            conv = self.conversations[i]
            instruction = None
            if high_level_instructions and i < len(high_level_instructions):
                instruction = high_level_instructions[i]

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
            )

            response_text = episode.defender_response_text()
            target_responses.append(response_text)
            conv.add_user_message(
                action, high_level_instruction=instruction, step_number=self.current_step
            )
            conv.add_assistant_message(response_text)
            conv.set_metadata("agentharm_messages", len(episode.messages))

        self.current_step += 1
        self.state = self._episodes_to_state()

        rewards = self.score_safety(target_responses)
        self.dones = torch.ones(len(actions)) if self.current_step >= self.max_steps else torch.zeros(len(actions))

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
