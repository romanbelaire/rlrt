"""AgentHarm dataset loading via inspect_evals."""

from typing import Any, Literal

from inspect_ai.dataset import Dataset, Sample

from inspect_evals.agentharm.utils import filter_dataset, load_dataset


def load_agentharm_dataset(
    task_name: Literal["harmful", "benign", "chat"],
    split: Literal["val", "test_public", "test_private"],
    force_redownload: bool = False,
    behavior_ids: list[str] | None = None,
    detailed_behaviors: bool | None = None,
    hint_included: bool | None = None,
) -> Dataset:
    dataset = load_dataset(task_name, split, force_redownload)
    ids = behavior_ids if behavior_ids else None
    return filter_dataset(dataset, ids, detailed_behaviors, hint_included)


def sample_to_metadata(sample: Sample) -> dict[str, Any]:
    meta = dict(sample.metadata)
    meta["prompt"] = sample.input
    meta["id"] = sample.id
    return meta


def dataset_to_list(dataset: Dataset) -> list[Sample]:
    return list(dataset)
