# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepared task-loss validation datasets for GPT validation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _get_ltor_masks_and_position_ids
from megatron.core.datasets.utils import Split


DCLM_CORE_TASK_LOSS_TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "commonsense_qa",
    "copa",
    "hellaswag",
    "openbookqa",
    "piqa",
    "winogrande",
    "wsc273",
    "lambada_openai",
    "coqa",
    "squadv2",
    "agieval_lsat_ar",
    "bigbench_language_identification_multiple_choice",
    "bigbench_qa_wikidata_generate_until",
    "bigbench_dyck_languages_generate_until",
    "bigbench_operators_generate_until",
    "bigbench_repeat_copy_logic_generate_until",
    "bigbench_cs_algorithms_generate_until",
]


DEFAULT_TASK_LOSS_DATA_DIR = "/e/project1/laionize/shechter1/core_megatron_datasets"


@dataclass(frozen=True)
class TaskLossSample:
    """Single prepared task-loss example."""

    token_ids: tuple[int, ...]
    label_positions: tuple[int, ...]
    label_ids: tuple[int, ...]


class PreparedTaskLossGPTDataset(Dataset):
    """GPT-style validation dataset with loss masked to prepared answer tokens."""

    split = Split.valid
    index_split = Split.valid

    def __init__(
        self,
        name: str,
        samples: list[TaskLossSample],
        config: GPTDatasetConfig,
    ) -> None:
        if not samples:
            raise ValueError(f"Prepared task-loss dataset {name} has no samples")
        self.name = name
        self.samples = samples
        self.config = config
        self.sequence_length = config.sequence_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = torch.zeros(self.sequence_length, dtype=torch.long)
        labels = torch.zeros(self.sequence_length, dtype=torch.long)
        loss_mask = torch.zeros(self.sequence_length, dtype=torch.float)

        sample = self.samples[idx]
        token_ids = torch.tensor(sample.token_ids, dtype=torch.long)
        tokens[: token_ids.numel()] = token_ids
        label_positions = torch.tensor(sample.label_positions, dtype=torch.long)
        label_ids = torch.tensor(sample.label_ids, dtype=torch.long)
        labels[label_positions] = label_ids
        loss_mask[label_positions] = 1.0

        attention_mask, _, position_ids = _get_ltor_masks_and_position_ids(
            tokens,
            self.config.tokenizer.eod,
            self.config.reset_position_ids,
            self.config.reset_attention_mask,
            self.config.eod_mask_loss,
            self.config.create_attention_mask,
        )

        item = {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
        if self.config.create_attention_mask:
            item["attention_mask"] = attention_mask
        return item


def resolve_task_loss_task_names(value: str | None) -> list[str]:
    """Resolve a task-loss task list or alias."""

    value = (value or "").strip()
    if not value or value.lower() in {"none", "off", "false", "0"}:
        return []
    if value in {"dclm-core-22", "dclm_core_22", "dclm-core", "dclm_core"}:
        return list(DCLM_CORE_TASK_LOSS_TASKS)
    return [item.strip() for item in value.split(",") if item.strip()]


def load_prepared_task_loss_validation_datasets(
    args: Any,
    config: GPTDatasetConfig,
) -> list[PreparedTaskLossGPTDataset]:
    """Load prepared task-loss validation datasets from local files."""

    task_names = resolve_task_loss_task_names(getattr(args, "task_eval_tasks", None))
    if not task_names:
        return []

    data_dir = os.path.abspath(
        os.path.expanduser(
            getattr(args, "task_eval_data_dir", None) or DEFAULT_TASK_LOSS_DATA_DIR
        )
    )
    metadata = _read_metadata(data_dir)
    if metadata and metadata.get("sequence_length") != config.sequence_length:
        raise ValueError(
            f"Prepared task-loss datasets use sequence_length={metadata.get('sequence_length')}, "
            f"but this run uses sequence_length={config.sequence_length}"
        )

    datasets = []
    missing_tasks = []
    for task_name in task_names:
        path = os.path.join(data_dir, f"{task_name}.pt")
        if not os.path.isfile(path):
            missing_tasks.append(task_name)
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "task_loss_samples_v1":
            raise ValueError(f"Unsupported prepared task-loss dataset format in {path}")
        samples = [
            TaskLossSample(
                token_ids=tuple(int(token) for token in sample["token_ids"]),
                label_positions=tuple(int(pos) for pos in sample["label_positions"]),
                label_ids=tuple(int(label) for label in sample["label_ids"]),
            )
            for sample in payload["samples"]
        ]
        datasets.append(PreparedTaskLossGPTDataset(task_name, samples, config))

    if missing_tasks:
        raise RuntimeError(
            "Missing prepared task-loss validation datasets for: " + ", ".join(missing_tasks)
        )

    return datasets


def _read_metadata(data_dir: str) -> dict[str, Any] | None:
    metadata_path = os.path.join(data_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
