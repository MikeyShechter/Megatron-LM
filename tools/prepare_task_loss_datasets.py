#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Prepare cached task-loss validation samples for offline Megatron runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from megatron.training.datasets.task_loss_dataset import resolve_task_loss_task_names


TASK_DATASET_CACHE_DIRS = {
    "arc_easy": ["allenai___ai2_arc"],
    "arc_challenge": ["allenai___ai2_arc"],
    "boolq": ["aps___super_glue"],
    "commonsense_qa": ["tau___commonsense_qa"],
    "copa": ["aps___super_glue"],
    "hellaswag": ["Rowan___hellaswag"],
    "openbookqa": ["allenai___openbookqa"],
    "piqa": ["baber___piqa"],
    "winogrande": ["allenai___winogrande"],
    "wsc273": ["winograd_wsc"],
    "lambada_openai": ["EleutherAI___lambada_openai"],
    "coqa": ["EleutherAI___coqa"],
    "squadv2": ["lighteval___squad_v2"],
    "agieval_lsat_ar": ["hails___agieval-lsat-ar"],
    "bigbench_language_identification_multiple_choice": ["hails___bigbench"],
    "bigbench_qa_wikidata_generate_until": ["hails___bigbench"],
    "bigbench_dyck_languages_generate_until": ["hails___bigbench"],
    "bigbench_operators_generate_until": ["hails___bigbench"],
    "bigbench_repeat_copy_logic_generate_until": ["hails___bigbench"],
    "bigbench_cs_algorithms_generate_until": ["hails___bigbench"],
}


DEFAULT_OUTPUT_DIR = Path("/e/project1/laionize/shechter1/core_megatron_datasets")
DEFAULT_WRITABLE_HF_HOME = Path("/e/project1/laionize/shechter1/.cache/huggingface")
DEFAULT_TOKENIZER = "EleutherAI/gpt-neox-20b"


@dataclass(frozen=True)
class PreparedSample:
    token_ids: tuple[int, ...]
    label_positions: tuple[int, ...]
    label_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare lm-eval task-loss validation datasets as local torch files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write prepared datasets into. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--tasks",
        default="dclm-core-22",
        help="Comma-separated lm-eval task names, or dclm-core-22.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=1024,
        help="Sequence length used when truncating prompt+answer examples.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Maximum usable examples per task. Values <= 0 use all examples.",
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help="HuggingFace tokenizer name or local path, loaded from local files only.",
    )
    parser.add_argument(
        "--shared-hf-home",
        type=Path,
        default=None,
        help="Read-only HF cache root containing datasets/. Default: auto-detect local shared cache.",
    )
    parser.add_argument(
        "--writable-hf-home",
        type=Path,
        default=DEFAULT_WRITABLE_HF_HOME,
        help=f"Writable HF cache root for symlinks and tokenizer cache. Default: {DEFAULT_WRITABLE_HF_HOME}",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    args = parse_args()

    task_names = resolve_task_loss_task_names(args.tasks)
    if not task_names:
        raise SystemExit("No task-loss tasks requested")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_hf_home = args.shared_hf_home or _default_shared_hf_home()
    hf_home = configure_task_loss_hf_caches(
        str(shared_hf_home) if shared_hf_home else None,
        str(args.writable_hf_home.expanduser()) if args.writable_hf_home else None,
        task_names,
    )
    _force_hf_libraries_offline()

    tokenizer = _build_task_loss_tokenizer(args.tokenizer)

    from lm_eval.tasks import TaskManager, get_task_dict

    task_manager = TaskManager()
    task_dict = get_task_dict(task_names, task_manager)

    metadata = {
        "format": "task_loss_samples_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tasks": task_names,
        "sequence_length": args.seq_length,
        "max_examples": args.max_examples,
        "tokenizer": args.tokenizer,
        "shared_hf_home": str(shared_hf_home) if shared_hf_home else None,
        "writable_hf_home": hf_home,
        "datasets": {},
    }

    missing_tasks = []
    for task_name in task_names:
        task_obj = task_dict.get(task_name)
        if task_obj is None:
            missing_tasks.append(task_name)
            continue

        task_obj.set_config(key="num_fewshot", value=0)
        samples, seen_docs, skipped_docs = _prepare_task_samples(
            task_name,
            task_obj,
            tokenizer,
            sequence_length=args.seq_length,
            max_examples=args.max_examples,
        )
        if not samples:
            missing_tasks.append(task_name)
            continue

        task_path = output_dir / f"{task_name}.pt"
        _save_task_file(
            task_path,
            task_name=task_name,
            samples=samples,
            sequence_length=args.seq_length,
            tokenizer=args.tokenizer,
        )
        metadata["datasets"][task_name] = {
            "path": str(task_path),
            "num_samples": len(samples),
            "seen_docs": seen_docs,
            "skipped_docs": skipped_docs,
        }
        logging.info(
            "Prepared %d samples for %s (%d docs seen, %d skipped)",
            len(samples),
            task_name,
            seen_docs,
            skipped_docs,
        )

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if missing_tasks:
        raise RuntimeError(
            "Failed to prepare task-loss datasets for: " + ", ".join(missing_tasks)
        )

    logging.info("Wrote task-loss metadata to %s", metadata_path)


def configure_task_loss_hf_caches(
    shared_hf_home: str | None,
    writable_hf_home: str | None,
    task_names: list[str] | None = None,
) -> str:
    writable_hf_home = writable_hf_home or _default_writable_hf_home()
    writable_hf_home = os.path.abspath(os.path.expanduser(writable_hf_home))
    os.makedirs(writable_hf_home, exist_ok=True)

    os.environ["HF_HOME"] = writable_hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(writable_hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(writable_hf_home, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(writable_hf_home, "datasets_task_loss")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    for path in (
        os.environ["HF_HUB_CACHE"],
        os.environ["TRANSFORMERS_CACHE"],
        os.environ["HF_DATASETS_CACHE"],
    ):
        os.makedirs(path, exist_ok=True)

    if shared_hf_home:
        shared_hf_home = os.path.abspath(os.path.expanduser(shared_hf_home))
        shared_datasets = os.path.join(shared_hf_home, "datasets")
        cache_dirs = _cache_dirs_for_tasks(task_names or [])
        if cache_dirs:
            for cache_dir in cache_dirs:
                _mirror_cache_tree(
                    os.path.join(shared_datasets, cache_dir),
                    os.path.join(os.environ["HF_DATASETS_CACHE"], cache_dir),
                )
        else:
            _mirror_cache_tree(shared_datasets, os.environ["HF_DATASETS_CACHE"])

    return writable_hf_home


def _default_shared_hf_home() -> Path | None:
    candidates = [
        Path("/datasets/playground/mmlaion/shared/oellm_shared_evals"),
        Path("/e/data1/datasets/playground/mmlaion/shared/oellm_shared_evals"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _default_writable_hf_home() -> str:
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return os.path.join(xdg_cache, "huggingface")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def _skip_cache_file(name: str) -> bool:
    return name.endswith(".lock") or name.endswith(".incomplete") or name.endswith(".tmp")


def _mirror_cache_tree(src: str, dst: str) -> None:
    if not src or not os.path.isdir(src):
        return
    if os.path.islink(dst):
        os.unlink(dst)
    os.makedirs(dst, exist_ok=True)

    try:
        names = os.listdir(src)
    except OSError as err:
        logging.warning("Cannot list cache %s: %s", src, err)
        return

    for name in names:
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if _skip_cache_file(name):
            if os.path.islink(dst_path):
                os.unlink(dst_path)
            continue
        if os.path.isdir(src_path) and not os.path.islink(src_path):
            _mirror_cache_tree(src_path, dst_path)
            continue
        if os.path.lexists(dst_path):
            continue
        try:
            os.symlink(src_path, dst_path)
        except OSError as err:
            logging.debug("Symlink failed for %s: %s", src_path, err)


def _cache_dirs_for_tasks(task_names: list[str]) -> list[str]:
    dirs = set()
    for task_name in task_names:
        dirs.update(TASK_DATASET_CACHE_DIRS.get(task_name, []))
    return sorted(dirs)


def _force_hf_libraries_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        import datasets.config

        datasets.config.HF_DATASETS_OFFLINE = True
    except Exception:
        pass


def _build_task_loss_tokenizer(tokenizer_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)


def _prepare_task_samples(
    task_name: str,
    task_obj: Any,
    tokenizer: Any,
    *,
    sequence_length: int,
    max_examples: int,
) -> tuple[list[dict[str, list[int]]], int, int]:
    samples = []
    seen_docs = 0
    skipped_docs = 0

    for doc in _get_task_docs(task_obj):
        seen_docs += 1
        try:
            sample = _build_sample(
                str(task_obj.doc_to_text(doc)),
                _resolve_answer(task_obj, doc),
                tokenizer,
                sequence_length,
            )
        except Exception as err:  # pylint: disable=broad-exception-caught
            logging.debug("Skipping bad doc for %s: %s", task_name, err)
            sample = None

        if sample is None:
            skipped_docs += 1
            continue

        samples.append(
            {
                "token_ids": list(sample.token_ids),
                "label_positions": list(sample.label_positions),
                "label_ids": list(sample.label_ids),
            }
        )
        if max_examples > 0 and len(samples) >= max_examples:
            break

    return samples, seen_docs, skipped_docs


def _get_task_docs(task_obj: Any) -> Any:
    if task_obj.has_test_docs():
        return task_obj.test_docs()
    if task_obj.has_validation_docs():
        return task_obj.validation_docs()
    if task_obj.has_training_docs():
        return task_obj.training_docs()
    return []


def _resolve_answer(task_obj: Any, doc: Any) -> str:
    target = task_obj.doc_to_target(doc)
    choices = None
    config = getattr(task_obj, "config", None)
    if getattr(config, "doc_to_choice", None) is not None:
        try:
            choices = task_obj.doc_to_choice(doc)
        except Exception:
            choices = None

    if isinstance(target, (list, tuple)):
        target = target[0] if target else ""

    if choices is not None:
        try:
            if isinstance(target, int):
                return str(choices[target])
            if isinstance(target, str) and target.strip().isdigit():
                return str(choices[int(target.strip())])
        except Exception:
            pass

    return str(target)


def _build_sample(
    prompt: str,
    answer: str,
    tokenizer: Any,
    sequence_length: int,
) -> PreparedSample | None:
    if not answer:
        return None

    prompt_ids = _tokenize(tokenizer, prompt)
    full_ids = _tokenize(tokenizer, prompt + answer)
    boundary = _find_answer_boundary(prompt_ids, full_ids)

    if len(full_ids) > sequence_length + 1:
        offset = len(full_ids) - (sequence_length + 1)
        full_ids = full_ids[offset:]
        boundary -= offset

    if boundary < 1:
        return None

    n_tokens = min(len(full_ids), sequence_length + 1)
    input_len = min(n_tokens, sequence_length)
    hi = min(n_tokens - 1, sequence_length)
    lo = boundary - 1
    if hi <= lo:
        return None

    label_positions = tuple(range(lo, hi))
    label_ids = tuple(int(full_ids[position + 1]) for position in label_positions)
    return PreparedSample(
        token_ids=tuple(int(token) for token in full_ids[:input_len]),
        label_positions=label_positions,
        label_ids=label_ids,
    )


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _find_answer_boundary(prompt_ids: list[int], full_ids: list[int]) -> int:
    boundary = 0
    upper = min(len(prompt_ids), len(full_ids))
    while boundary < upper and prompt_ids[boundary] == full_ids[boundary]:
        boundary += 1
    return boundary


def _save_task_file(
    path: Path,
    *,
    task_name: str,
    samples: list[dict[str, list[int]]],
    sequence_length: int,
    tokenizer: str,
) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    payload = {
        "format": "task_loss_samples_v1",
        "task_name": task_name,
        "sequence_length": sequence_length,
        "tokenizer": tokenizer,
        "num_samples": len(samples),
        "samples": samples,
    }
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


if __name__ == "__main__":
    main()
