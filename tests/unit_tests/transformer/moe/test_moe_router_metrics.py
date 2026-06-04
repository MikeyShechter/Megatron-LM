# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import _build_moe_router_metrics_log


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
        metrics,
        prefix="val",
        num_experts=4,
        loss_scale=0.5,
        moe_router_load_balancing_type="aux_loss",
    )

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
