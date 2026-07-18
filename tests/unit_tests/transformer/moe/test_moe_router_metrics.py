# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import (
    _PRE_ACTIVATION_METRIC_LOG_NAMES,
    _build_moe_router_metrics_log,
)


def _add_pre_activation_metrics(metrics):
    for metric_name in _PRE_ACTIVATION_METRIC_LOG_NAMES:
        metrics[f"{metric_name}_top1_pre_activation_sum"] = torch.zeros(2)
        metrics[f"{metric_name}_top2_pre_activation_sum"] = torch.zeros(2)
        metrics[f"{metric_name}_pre_activation_sum"] = torch.zeros(2)
        metrics[f"{metric_name}_top_count"] = torch.zeros(2)
        metrics[f"{metric_name}_value_count"] = torch.zeros(2)

    metrics["router_logits_top1_pre_activation_sum"] = torch.tensor([2.0, 4.0])
    metrics["router_logits_top2_pre_activation_sum"] = torch.tensor([1.0, 3.0])
    metrics["router_logits_pre_activation_sum"] = torch.tensor([6.0, 10.0])
    metrics["router_logits_top_count"] = torch.tensor([2.0, 2.0])
    metrics["router_logits_value_count"] = torch.tensor([4.0, 4.0])

    metrics["learnable_both_bias_top1_pre_activation_sum"] = torch.tensor([1.0, 5.0])
    metrics["learnable_both_bias_top2_pre_activation_sum"] = torch.tensor([0.5, 2.5])
    metrics["learnable_both_bias_pre_activation_sum"] = torch.tensor([3.0, 9.0])
    metrics["learnable_both_bias_top_count"] = torch.tensor([1.0, 3.0])
    metrics["learnable_both_bias_value_count"] = torch.tensor([2.0, 6.0])

    return metrics


def test_build_val_moe_router_metrics_log():
    metrics = {
        "tokens_per_expert": torch.tensor(
            [
                [4.0, 2.0, 2.0, 0.0],
                [1.0, 1.0, 1.0, 5.0],
            ]
        ),
        "prob_sum": torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ),
        "token_count": torch.tensor([8.0, 8.0]),
        "entropy_sum": torch.tensor([4.0, 6.0]),
        "avg_1_2_coef_diff_sum": torch.tensor([2.0, 4.0]),
        "ste_in_rect_count": torch.tensor([2.0, 3.0]),
        "ste_selected_count": torch.tensor([8.0, 8.0]),
        "ste_over_rect_count": torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 4.0],
            ]
        ),
    }

    log = _build_moe_router_metrics_log(
        _add_pre_activation_metrics(metrics),
        prefix="val",
        num_experts=4,
        loss_scale=0.5,
        moe_router_load_balancing_type="aux_loss",
    )

    assert log["router_logits/top1_pre_activation"] == pytest.approx(1.5)
    assert log["router_logits/top2_pre_activation"] == pytest.approx(1.0)
    assert log["router_logits/avg_pre_activation"] == pytest.approx(2.0)
    assert log["router_bias/both/top1_pre_activation"] == pytest.approx(1.5)
    assert log["router_bias/both/top2_pre_activation"] == pytest.approx(0.75)
    assert log["router_bias/both/avg_pre_activation"] == pytest.approx(1.5)
    assert all(not key.startswith("val/router_logits/") for key in log)
    assert all(not key.startswith("val/router_bias/") for key in log)
    assert log["val/router_entropy"] == pytest.approx(0.625)
    assert log["vio/MaxVioGlobal"] == pytest.approx(1.25)
    assert log["vio/MaxVioGlobalWorstLayer"] == pytest.approx(1.5)
    assert log["vio/TotalVioGlobal"] == pytest.approx(2.5)
    assert log["vio/MaxVio/Layer 0"] == pytest.approx(1.0)
    assert log["vio/MaxVio/Layer 1"] == pytest.approx(1.5)
    assert log["val/aux_loss"] == pytest.approx(1.0)
    assert log["val/router_values/avg_1_2_coef_diff"] == pytest.approx(0.375)
    assert log["ste/all_layers/in_rect_frac"] == pytest.approx(5.0 / 16.0)
    assert log["ste/all_layers/max_over_rect"] == pytest.approx(0.5)
    assert log["ste/all_layers/avg_over_rect"] == pytest.approx(0.15625)
    assert all(not key.startswith("val/MaxVio") for key in log)
    assert all(not key.startswith("val/TotalVio") for key in log)
    assert all(not key.startswith("val/ste/") for key in log)


def test_build_task_validation_moe_router_metrics_log_skips_pre_activation_diagnostics():
    metrics = {
        "tokens_per_expert": torch.tensor(
            [
                [4.0, 2.0, 2.0, 0.0],
                [1.0, 1.0, 1.0, 5.0],
            ]
        ),
        "prob_sum": torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ),
        "token_count": torch.tensor([8.0, 8.0]),
        "entropy_sum": torch.tensor([4.0, 6.0]),
        "avg_1_2_coef_diff_sum": torch.tensor([2.0, 4.0]),
        "ste_in_rect_count": torch.tensor([2.0, 3.0]),
        "ste_selected_count": torch.tensor([8.0, 8.0]),
        "ste_over_rect_count": torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 4.0],
            ]
        ),
    }

    log = _build_moe_router_metrics_log(
        _add_pre_activation_metrics(metrics),
        prefix="val/squadv2",
        num_experts=4,
        loss_scale=0.5,
        moe_router_load_balancing_type="aux_loss",
    )

    assert log["task_entropy/squadv2"] == pytest.approx(0.625)
    assert all("pre_activation" not in key for key in log)
    assert all(not key.startswith("val/squadv2/router_logits/") for key in log)
    assert all(not key.startswith("val/squadv2/router_bias/") for key in log)


def test_build_regular_validation_moe_router_metrics_log_emits_root_pre_activation_diagnostics():
    metrics = {
        "tokens_per_expert": torch.tensor(
            [
                [4.0, 2.0, 2.0, 0.0],
                [1.0, 1.0, 1.0, 5.0],
            ]
        ),
        "prob_sum": torch.tensor(
            [
                [2.0, 2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0, 2.0],
            ]
        ),
        "token_count": torch.tensor([8.0, 8.0]),
        "entropy_sum": torch.tensor([4.0, 6.0]),
        "avg_1_2_coef_diff_sum": torch.tensor([2.0, 4.0]),
        "ste_in_rect_count": torch.tensor([2.0, 3.0]),
        "ste_selected_count": torch.tensor([8.0, 8.0]),
        "ste_over_rect_count": torch.tensor(
            [
                [1.0, 2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0, 4.0],
            ]
        ),
    }

    log = _build_moe_router_metrics_log(
        _add_pre_activation_metrics(metrics),
        prefix="val/regular",
        num_experts=4,
        loss_scale=0.5,
        moe_router_load_balancing_type="aux_loss",
    )

    assert log["router_logits/top1_pre_activation"] == pytest.approx(1.5)
    assert log["router_bias/both/top2_pre_activation"] == pytest.approx(0.75)
    assert log["val/router_entropy"] == pytest.approx(0.625)


def test_build_train_moe_router_metrics_log_skips_val_only_diagnostics():
    metrics = {
        "tokens_per_expert": torch.tensor([[4.0, 2.0, 2.0, 0.0]]),
        "prob_sum": torch.tensor([[2.0, 2.0, 2.0, 2.0]]),
        "token_count": torch.tensor([8.0]),
        "entropy_sum": torch.tensor([4.0]),
        "avg_1_2_coef_diff_sum": torch.tensor([2.0]),
        "ste_in_rect_count": torch.tensor([2.0]),
        "ste_selected_count": torch.tensor([8.0]),
        "ste_over_rect_count": torch.tensor([[1.0, 2.0, 0.0, 0.0]]),
    }

    log = _build_moe_router_metrics_log(
        metrics,
        prefix="train",
        num_experts=4,
        loss_scale=1.0,
        moe_router_load_balancing_type="centered_fsq",
    )

    assert log["train/router_values/avg_1_2_coef_diff"] == pytest.approx(0.25)
    assert log["train/aux_loss"] == pytest.approx(1.5)
    assert "router_values/avg_1_2_coef_diff" not in log
    assert "ste/all_layers/in_rect_frac" not in log
    assert "train/MaxViobatch" not in log
    assert all("MaxVio" not in key for key in log)
