# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.tokenizers.utils.build_tokenizer import vocab_size_with_padding
from megatron.core.pipeline_parallel.schedules import _accumulate_metagrad_raw_sensitivities
from megatron.core.transformer.moe.moe_utils import (
    add_to_moe_metagrad_raw_sensitivity,
    clear_moe_metagrad_prev_sensitivities,
    clear_moe_metagrad_raw_sensitivities,
    consume_metagrad_losses_tracker,
    get_moe_metagrad_checkpoint_state,
    get_moe_metagrad_prev_sensitivities,
    get_moe_metagrad_raw_sensitivities,
    get_moe_metagrad_scalar_state,
    load_moe_metagrad_checkpoint_state,
    save_to_metagrad_losses_tracker,
    set_moe_metagrad_prev_sensitivities,
)
from megatron.training import checkpointing as checkpointing_module
from megatron.training import training as training_module
from megatron.training.checkpointing import _shard_moe_metagrad_state, save_grads
from megatron.training.global_vars import set_args
from megatron.training.training import (
    _apply_metagrad_scalar_updates,
    _average_metagrad_raw_sensitivities,
    _compute_metagrad_scalar,
    _get_or_create_metagrad_optimizer,
    build_train_valid_test_data_iterators,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


def mock_train_valid_test_datasets_provider(train_val_test_num_samples):
    return iter([1]), iter([2]), iter([3])


class _LenDataloader:
    """Fake dataloader with __len__ (required by the full_validation path)
    and __iter__ (consumed via cyclic_iter)."""

    def __init__(self, data):
        self._data = list(data)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


def mock_multi_valid_full_datasets_provider(train_val_test_num_samples):
    return (iter([1]), [_LenDataloader([2, 2]), _LenDataloader([20, 20, 20])], iter([3]))


def create_test_args():
    # Set dummy values for the args.
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = 1
    args.train_iters = 1
    args.eval_interval = 1
    args.eval_iters = 1
    args.global_batch_size = 1
    args.consumed_train_samples = 1
    args.consumed_valid_samples = 1
    args.dataloader_type = "external"
    args.skip_train = False
    args.start_eval_at_iter = None
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None

    return args


def clear_metagrad_scalar_state():
    state = get_moe_metagrad_scalar_state()
    state["scalar_params"] = {}
    state["scalar_optimizer"] = None
    state["pending_scalar_optimizer_state"] = None
    state.pop("restored_scalar_params", None)


class TestTraining:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        args = create_test_args()
        set_args(args)
        clear_moe_metagrad_raw_sensitivities()
        clear_moe_metagrad_prev_sensitivities()
        clear_metagrad_scalar_state()
        consume_metagrad_losses_tracker()

    def test_build_train_valid_test_data_iterators(self):
        train_iter, valid_iter, test_iter = build_train_valid_test_data_iterators(
            mock_train_valid_test_datasets_provider
        )
        train_data = next(train_iter)
        valid_data = next(valid_iter)
        test_data = next(test_iter)
        assert (train_data, valid_data, test_data) == (1, 2, 3)

    def test_build_train_valid_test_data_iterators_multi_full_validation(self):
        """multiple_validation_sets + full_validation builds a list of iterators
        (one per validation set) and sets args.eval_iters to the per-loader
        lengths MAX-reduced across DP ranks."""
        args = create_test_args()
        args.multiple_validation_sets = True
        args.full_validation = True
        set_args(args)
        _, valid_iters, _ = build_train_valid_test_data_iterators(
            mock_multi_valid_full_datasets_provider
        )
        assert isinstance(valid_iters, list)
        assert len(valid_iters) == 2
        assert next(valid_iters[0]) == 2
        assert next(valid_iters[1]) == 20
        # data_parallel_size=1, so MAX across DP ranks equals the local lengths
        assert args.eval_iters == [2, 3]

    def test_closed_formula_vocab_size_with_padding(self):
        def old_round_impl(after, multiple):
            while (after % multiple) != 0:
                after += 1
            return after

        args = SimpleNamespace()
        args.rank = 0
        args.tensor_model_parallel_size = 1

        for vocab in range(1, 600000, 1000):
            for mult in [1, 17, 32, 64, 128]:
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

        for vocab in range(1, 10_000, 500):
            for mult in range(1, 1024 + 1):
                args.make_vocab_size_divisible_by = mult
                assert old_round_impl(vocab, mult) == vocab_size_with_padding(vocab, args, False), (
                    vocab,
                    mult,
                )

    def test_metagrad_checkpoint_state_round_trip(self):
        class DenseAndExpertModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dense = torch.nn.Parameter(torch.zeros(2, device="cuda"))
                self.expert = torch.nn.Parameter(torch.zeros(2, device="cuda"))
                self.expert.allreduce = False

        model = DenseAndExpertModel()
        model.config = SimpleNamespace(
            moe_load_balance_ste_width=0.1,
            moe_aux_loss_coeff=0.01,
        )
        args = SimpleNamespace(
            metagrad_params="width",
            metagrad_lr=0.001,
            moe_load_balance_ste_width=0.1,
            moe_aux_loss_coeff=0.01,
        )
        dense_sensitivity = torch.full_like(model.dense, 3.0)
        expert_sensitivity = torch.full_like(model.expert, 5.0)
        set_moe_metagrad_prev_sensitivities(
            "width",
            {
                id(model.dense): (model.dense, dense_sensitivity),
                id(model.expert): (model.expert, expert_sensitivity),
            },
        )
        _apply_metagrad_scalar_updates(args, model.config, None, None, {"width": 1.0})

        live_optimizer = get_moe_metagrad_scalar_state()["scalar_optimizer"]
        live_optimizer_devices = {
            (id(param), key): value.device
            for param, optimizer_state in live_optimizer.state.items()
            for key, value in optimizer_state.items()
            if torch.is_tensor(value)
        }
        checkpoint_state = get_moe_metagrad_checkpoint_state([model])
        assert "model0.dense" in checkpoint_state["dense_prev"]["width"]
        assert "model0.expert" in checkpoint_state["expert_prev"]["width"]
        assert live_optimizer_devices == {
            (id(param), key): value.device
            for param, optimizer_state in live_optimizer.state.items()
            for key, value in optimizer_state.items()
            if torch.is_tensor(value)
        }
        assert all(
            value.device.type == "cpu"
            for optimizer_state in checkpoint_state["scalar_optimizer"]["state"].values()
            for value in optimizer_state.values()
            if torch.is_tensor(value)
        )
        _apply_metagrad_scalar_updates(args, model.config, None, None, {"width": 1.0})
        assert args.moe_load_balance_ste_width == pytest.approx(0.098)

        load_moe_metagrad_checkpoint_state(checkpoint_state, [model], args=args)
        restored_state = _get_or_create_metagrad_optimizer(args)

        restored_prev = get_moe_metagrad_prev_sensitivities("width")
        torch.testing.assert_close(
            restored_prev[id(model.dense)][1], dense_sensitivity
        )
        torch.testing.assert_close(
            restored_prev[id(model.expert)][1], expert_sensitivity
        )
        assert args.moe_load_balance_ste_width == pytest.approx(0.099)
        assert model.config.moe_load_balance_ste_width == pytest.approx(0.099)
        assert restored_state["scalar_params"]["width"].item() == pytest.approx(0.099)
        assert restored_state["scalar_optimizer"].state

    def test_metagrad_checkpoint_uses_expert_parallel_shard(self, monkeypatch):
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_pipeline_model_parallel_world_size", lambda: 2
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_tensor_model_parallel_world_size", lambda: 4
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_expert_tensor_parallel_world_size", lambda: 2
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_expert_model_parallel_world_size", lambda: 8
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_pipeline_model_parallel_rank", lambda: 1
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_tensor_model_parallel_rank", lambda: 3
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_expert_tensor_parallel_rank", lambda: 1
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_expert_model_parallel_rank", lambda: 6
        )
        monkeypatch.setattr(
            checkpointing_module.mpu,
            "get_data_parallel_rank",
            lambda with_context_parallel: 11,
        )
        monkeypatch.setattr(
            checkpointing_module.mpu, "get_expert_data_parallel_rank", lambda: 2
        )

        state = _shard_moe_metagrad_state(
            {
                "dense_prev": {"width": {"dense": torch.ones(1)}},
                "expert_prev": {"width": {"expert": torch.ones(1)}},
                "scalar_params": {},
                "scalar_optimizer": None,
            }
        )

        dense = state["dense_prev"]
        expert = state["expert_prev"]
        assert dense.global_shape == (2, 4)
        assert dense.global_offset == (1, 3)
        assert dense.replica_id == 11
        assert expert.global_shape == (2, 2, 8)
        assert expert.global_offset == (1, 1, 6)
        assert expert.replica_id == 2

    def test_metagrad_averages_expert_sensitivities_over_expert_dp(self, monkeypatch):
        dense_group = object()
        expert_group = object()
        dense = torch.nn.Parameter(torch.zeros(1, device="cuda"))
        expert = torch.nn.Parameter(torch.zeros(1, device="cuda"))
        expert.allreduce = False
        add_to_moe_metagrad_raw_sensitivity("width", dense, torch.full_like(dense, 2.0))
        add_to_moe_metagrad_raw_sensitivity("width", expert, torch.full_like(expert, 3.0))

        monkeypatch.setattr(
            training_module.mpu,
            "get_data_parallel_group",
            lambda with_context_parallel: dense_group,
        )
        monkeypatch.setattr(
            training_module.mpu, "get_expert_data_parallel_group", lambda: expert_group
        )
        monkeypatch.setattr(
            torch.distributed,
            "get_world_size",
            lambda group: 4 if group is dense_group else 2,
        )
        reduced_groups = []

        def fake_all_reduce(tensor, op, group):
            reduced_groups.append(group)
            tensor.mul_(4 if group is dense_group else 2)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        _average_metagrad_raw_sensitivities()

        assert reduced_groups == [dense_group, expert_group]
        torch.testing.assert_close(
            get_moe_metagrad_raw_sensitivities("width")[id(dense)][1],
            torch.full_like(dense, 2.0),
        )
        torch.testing.assert_close(
            get_moe_metagrad_raw_sensitivities("width")[id(expert)][1],
            torch.full_like(expert, 3.0),
        )

    def test_metagrad_reduces_expert_dot_over_ep_group(self, monkeypatch):
        class FakeOptimizer:
            def get_loss_scale(self):
                return torch.ones([], device="cuda")

        dense_group = object()
        expert_group = object()
        dense_tp_group = object()
        expert_tp_group = object()
        dense = torch.nn.Parameter(torch.zeros(1, device="cuda"))
        expert = torch.nn.Parameter(torch.zeros(1, device="cuda"))
        expert.allreduce = False
        dense.main_grad = torch.full_like(dense, 2.0)
        expert.main_grad = torch.full_like(expert, 3.0)
        set_moe_metagrad_prev_sensitivities(
            "width",
            {
                id(dense): (dense, torch.full_like(dense, 5.0)),
                id(expert): (expert, torch.full_like(expert, 7.0)),
            },
        )

        monkeypatch.setattr(
            training_module.mpu, "get_model_parallel_group", lambda: dense_group
        )
        monkeypatch.setattr(
            training_module.mpu,
            "get_expert_tensor_model_pipeline_parallel_group",
            lambda: expert_group,
        )
        monkeypatch.setattr(
            training_module.mpu,
            "get_tensor_model_parallel_group",
            lambda: dense_tp_group,
        )
        monkeypatch.setattr(
            training_module.mpu,
            "get_expert_tensor_parallel_group",
            lambda: expert_tp_group,
        )
        monkeypatch.setattr(
            training_module.tensor_parallel,
            "param_is_not_tensor_parallel_duplicate",
            lambda param, tp_group: True,
        )
        monkeypatch.setattr(
            torch.distributed,
            "get_process_group_ranks",
            lambda group: [0] if group is dense_group else [0, 1],
        )
        reduced_groups = []

        def fake_all_reduce(tensor, op, group):
            reduced_groups.append(group)
            if group is expert_group:
                tensor[0].add_(100.0)
                tensor[1].add_(1.0)

        monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
        scalar = _compute_metagrad_scalar(FakeOptimizer(), "width")

        assert reduced_groups == [dense_group, expert_group]
        assert scalar == pytest.approx(131.0)

    def test_metagrad_reads_fused_main_grad_delta_and_restores_buffer(self):
        class FusedWeightGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, param):
                ctx.param = param
                return param.sum()

            @staticmethod
            def backward(ctx, grad_output):
                param = ctx.param
                param.main_grad.add_(torch.full_like(param.main_grad, 3.0) * grad_output)
                param.grad_added_to_main_grad = True
                return torch.full_like(param, float("nan"))

        model = torch.nn.Linear(2, 1, bias=False, device="cuda")
        model.weight.main_grad = torch.full_like(model.weight, 7.0, dtype=torch.float32)
        model.weight.grad_added_to_main_grad = True
        save_to_metagrad_losses_tracker("width", FusedWeightGradient.apply(model.weight))

        _accumulate_metagrad_raw_sensitivities(model)

        raw = get_moe_metagrad_raw_sensitivities("width")[id(model.weight)][1]
        torch.testing.assert_close(raw, torch.full_like(raw, 3.0))
        torch.testing.assert_close(
            model.weight.main_grad, torch.full_like(model.weight.main_grad, 7.0)
        )
        assert model.weight.grad_added_to_main_grad

    def test_metagrad_restores_te_fuser_saved_tensor_ranges(self):
        class OperationContext:
            def __init__(self):
                self._saved_tensors_range = (0, 1)

        class ConsumingFuserBackward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, param):
                ctx.basic_op_ctxs = [OperationContext()]
                ctx.tensor_objects = [None, None]
                ctx.param_shape = param.shape
                ctx.save_for_backward(param.detach().clone(), None)
                return param.sum()

            @staticmethod
            def backward(ctx, grad_output):
                op_ctx = ctx.basic_op_ctxs[0]
                if op_ctx._saved_tensors_range is None:
                    raise TypeError("saved tensor range was consumed")
                if ctx.tensor_objects is None:
                    raise TypeError("tensor object metadata was consumed")
                saved_tensor, optional_tensor = ctx.saved_tensors
                assert optional_tensor is None
                if saved_tensor.numel() == 0:
                    raise RuntimeError("saved tensor data was consumed")
                op_ctx._saved_tensors_range = None
                ctx.tensor_objects = None
                if not hasattr(saved_tensor, "_do_not_clear"):
                    saved_tensor.data = torch.empty(0, device=saved_tensor.device)
                return torch.ones(ctx.param_shape, device=grad_output.device) * grad_output

        model = torch.nn.Linear(2, 1, bias=False, device="cuda")
        loss = ConsumingFuserBackward.apply(model.weight)
        save_to_metagrad_losses_tracker("width", loss)
        save_to_metagrad_losses_tracker("coeff", loss)

        _accumulate_metagrad_raw_sensitivities(model)
        loss.backward()

        for name in ("width", "coeff"):
            raw = get_moe_metagrad_raw_sensitivities(name)[id(model.weight)][1]
            torch.testing.assert_close(raw, torch.ones_like(raw))

    def teardown_method(self, method):
        clear_moe_metagrad_raw_sensitivities()
        clear_moe_metagrad_prev_sensitivities()
        clear_metagrad_scalar_state()
        consume_metagrad_losses_tracker()
        Utils.destroy_model_parallel()


class TestSaveGrads:
    """Tests for the save_grads function."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_save_grads(self, tmp_path_dist_ckpt):
        """Test that save_grads creates the correct directory structure and saves
        state_dict correctly.

        With TP=1, PP=1 on 8 GPUs, we have 8 DP ranks. Only the rank with
        expert_data_parallel_rank==0 should save. All ranks verify the result.
        """
        save_dir = str(tmp_path_dist_ckpt / "test_save_grads")

        with TempNamedDir(save_dir, sync=True) as save_dir:
            # Create a mock state_dict with gradients (use deterministic values for reproducibility).
            state_dict = defaultdict(dict)
            state_dict["model_chunk0"]["layer.weight"] = torch.arange(16).reshape(4, 4).float()
            state_dict["model_chunk0"]["layer.bias"] = torch.arange(4).float()

            iteration = 100
            grad_label = "wgrads"

            # All ranks call save_grads, but only expert_data_parallel_rank==0 actually saves.
            save_grads(save_dir, dict(state_dict), iteration, grad_label)

            # Synchronize before checking results since only rank 0 saves.
            torch.distributed.barrier()

            # All ranks verify the file was created by rank 0.
            expected_dir = Path(save_dir) / grad_label / f"iter_{iteration:07d}"
            assert expected_dir.exists(), f"Expected directory {expected_dir} to exist"

            expected_file = expected_dir / "mp_rank_00.pth"
            assert expected_file.exists(), f"Expected file {expected_file} to exist"

            # Verify saved content.
            loaded = torch.load(expected_file)
            assert "model_chunk0" in loaded
            assert "layer.weight" in loaded["model_chunk0"]
            assert "layer.bias" in loaded["model_chunk0"]
            assert torch.equal(
                loaded["model_chunk0"]["layer.weight"], state_dict["model_chunk0"]["layer.weight"]
            )
            assert torch.equal(
                loaded["model_chunk0"]["layer.bias"], state_dict["model_chunk0"]["layer.bias"]
            )
