#!/usr/bin/env python3
"""Evaluate a split-router checkpoint with router and weighter expert selection.

Run this through the same distributed launcher as pretrain_gpt.py and pass the
completed run's spec.yaml. The spec's output_dir is used as the checkpoint load
directory. Evaluation is validation-only; training, test-set evaluation, and
prepared downstream task evaluation are disabled by this entry point.
"""

import time
from functools import partial

import pretrain_gpt as gpt
from megatron.core.enums import ModelType
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import parse_and_validate_args


def _validation_only_datasets_provider(train_valid_test_num_samples, vp_stage=None):
    train_ds, valid_ds, _ = gpt.train_valid_test_datasets_provider(
        train_valid_test_num_samples, vp_stage=vp_stage
    )
    return train_ds, valid_ds, None


def main() -> None:
    main_entry_time = time.time()
    gpt.set_startup_timestamps(
        program_start=gpt._PROGRAM_START_TIME, main_entry=main_entry_time
    )

    pretrain, store = gpt.inprocess_restart.maybe_wrap_for_inprocess_restart(gpt.pretrain)
    args = parse_and_validate_args(
        extra_args_provider=gpt.add_modelopt_args if gpt.has_nvidia_modelopt else None,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    )

    args.skip_train = True
    args.no_load_optim = True
    args.no_load_rng = True
    args.skip_task_eval = True
    args.eval_split_router_with_weighter = True
    args.eval_split_router_with_router_weights = True
    args.multiple_validation_sets = False
    args.task_loss_eval_task_names = []
    if args.wandb_exp_name:
        args.wandb_exp_name = f"{args.wandb_exp_name}-router-weighter-eval"

    full_config = pretrain_cfg_container_from_args(args)
    _validation_only_datasets_provider.is_distributed = True
    pretrain(
        full_config,
        _validation_only_datasets_provider,
        partial(gpt.model_provider, gpt.gpt_builder),
        ModelType.encoder_or_decoder,
        gpt.forward_step,
        store=store,
        get_embedding_ranks=gpt.get_embedding_ranks,
    )


if __name__ == "__main__":
    main()
