"""AgentHarm-Gen dataset loading for RL training."""

from typing import Any

from inspect_ai.dataset import Dataset, Sample

from src.agentharm_gen.harness.dataset import load_generated_dataset


def sample_to_metadata(sample: Sample) -> dict[str, Any]:
    meta = dict(sample.metadata)
    meta["prompt"] = sample.input
    meta["id"] = sample.id
    return meta


def dataset_to_list(dataset: Dataset) -> list[Sample]:
    return list(dataset)


def load_agentharm_gen_dataset(tasks_path: str | None = None) -> Dataset:
    return load_generated_dataset(tasks_path)
