"""AgentHarm official harm scoring (grading + refusal judge)."""

from dataclasses import dataclass
from typing import Any, Literal

import torch
from inspect_ai.model import ChatMessage, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState

from inspect_evals.agentharm.benchmark import (
    benign_grading_functions,
    harmful_grading_functions,
)
from inspect_evals.agentharm.scorer import RefusalJudgeLLM, combined_scorer

from src.agentharm.message_bridge import EpisodeState


@dataclass
class HarmScoreResult:
    harm_score: float
    refusal: bool
    explanation: str
    score_dict: dict[str, float]


class AgentHarmScorer:
    def __init__(
        self,
        task_name: Literal["harmful", "benign", "chat"],
        refusal_judge_model: str,
        semantic_judge_model: str,
        refusal_penalty: float = 0.0,
    ) -> None:
        self.task_name = task_name
        self.refusal_penalty = refusal_penalty
        self.refusal_judge = RefusalJudgeLLM(judge_model=refusal_judge_model)
        self.semantic_judge = get_model(semantic_judge_model)
        if task_name == "harmful":
            self.grading_module = harmful_grading_functions
        elif task_name == "benign":
            self.grading_module = benign_grading_functions
        else:
            self.grading_module = None
        self._scorer_fn = combined_scorer(
            self.grading_module, self.refusal_judge, self.semantic_judge
        )

    async def score_episode_async(self, episode: EpisodeState) -> HarmScoreResult:
        if episode.task_state is None:
            raise ValueError(
                "AgentHarm scoring requires a completed defender turn (episode.task_state is None). "
                "Official harm scores cannot be computed on attacker actions alone."
            )
        inspect_model = episode.task_state.model
        state = TaskState(
            model=inspect_model,
            sample_id=episode.behavior_id,
            epoch=0,
            input=episode.metadata["prompt"],
            messages=list(episode.messages),
            metadata=dict(episode.metadata),
        )
        target = Target("")
        combined = await self._scorer_fn(state, target)
        harm_score = float(combined.value["score"])
        refusal = bool(combined.value["refusal"] > 0.5)
        if refusal and self.refusal_penalty > 0:
            harm_score = max(0.0, harm_score - self.refusal_penalty)
        grading_function_name = episode.metadata["grading_function"]
        raw_score_dict = await getattr(self.grading_module, grading_function_name)(
            episode.metadata, list(episode.messages)
        )
        score_dict = {k: float(v) for k, v in raw_score_dict.items()}
        return HarmScoreResult(
            harm_score=harm_score,
            refusal=refusal,
            explanation=combined.explanation or "",
            score_dict=score_dict,
        )

    def score_episode(self, episode: EpisodeState) -> HarmScoreResult:
        import asyncio

        return asyncio.run(self.score_episode_async(episode))

    def score_episodes(self, episodes: list[EpisodeState]) -> torch.Tensor:
        scores = []
        for episode in episodes:
            result = self.score_episode(episode)
            episode.metadata["agentharm_harm_score"] = result.harm_score
            episode.metadata["agentharm_refusal"] = result.refusal
            scores.append(result.harm_score)
        return torch.tensor(scores, dtype=torch.float32)


async def score_episodes_batch_async(
    scorer: AgentHarmScorer,
    episodes: list[EpisodeState],
) -> list[HarmScoreResult]:
    import asyncio

    tasks = [scorer.score_episode_async(ep) for ep in episodes]
    return await asyncio.gather(*tasks)
