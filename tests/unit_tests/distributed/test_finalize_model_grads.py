# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import inspect
import os

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import (
    _allreduce_non_tensor_model_parallel_grads,
    _allreduce_word_embedding_grads,
    _update_router_qb_beta,
    reset_model_temporary_tensors,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestUpdateRouterQBBeta:
    """Exercises the QB bias update in finalize_model_grads against a real MoE router."""

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_moe_layer(self, ema):
        num_experts = 8
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="quantile_balancing",
            moe_router_score_function="softmax",
            moe_router_topk=2,
            moe_aux_loss_coeff=0,
            moe_router_quantile_balancing_ema=ema,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_experts, moe_grouped_gemm=False
        )
        return config, MoELayer(config, submodules.mlp.submodules).cuda()

    @pytest.mark.parametrize("ema", [0.0, 0.9])
    def test_update_router_qb_beta(self, ema):
        config, moe_layer = self._build_moe_layer(ema)
        router = moe_layer.router
        router.train()
        router.qb_beta.copy_(torch.randn_like(router.qb_beta))

        hidden = torch.randn((32, 2, config.hidden_size)).cuda().bfloat16()
        router(hidden)
        router(hidden)
        assert router.qb_beta_count.item() == 2
        assert router.qb_beta_accum.abs().sum().item() > 0

        local_avg = router.qb_beta_accum / router.qb_beta_count.clamp(min=1).to(torch.float32)
        torch.distributed.all_reduce(
            local_avg, op=torch.distributed.ReduceOp.AVG, group=torch.distributed.group.WORLD
        )
        blended = ema * router.qb_beta + (1.0 - ema) * local_avg
        expected = blended - blended.mean(dim=-1, keepdim=True)

        _update_router_qb_beta([moe_layer], config, dp_cp_group=torch.distributed.group.WORLD)

        torch.testing.assert_close(router.qb_beta, expected)
        torch.testing.assert_close(
            router.qb_beta.mean(), torch.zeros((), device=router.qb_beta.device)
        )

        reset_model_temporary_tensors(config, [moe_layer])
        torch.testing.assert_close(router.qb_beta_accum, torch.zeros_like(router.qb_beta_accum))
        assert router.qb_beta_count.item() == 0

    def test_update_router_qb_beta_skips_eval(self):
        config, moe_layer = self._build_moe_layer(ema=0.0)
        router = moe_layer.router
        router.qb_beta.copy_(torch.ones_like(router.qb_beta))
        router.qb_beta_accum.copy_(
            torch.arange(router.qb_beta.numel(), dtype=torch.float32, device=router.qb_beta.device)
        )
        router.qb_beta_count.fill_(1)
        before = router.qb_beta.clone()
        router.eval()

        _update_router_qb_beta([moe_layer], config, dp_cp_group=torch.distributed.group.WORLD)

        torch.testing.assert_close(router.qb_beta, before)


class TestAllReduceLNGrads:

    def init_model(self, share_embeddings_and_output_weights: bool = False):
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            qk_layernorm=True,
            pipeline_dtype=torch.float32,
        )

        self.model = GPTModel(
            config=self.transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
            vocab_size=100,
            max_sequence_length=4,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
        )

    def setup_method(self, method):
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("freeze_model,tp_size", [(True, 2), (False, 2)])
    def test_allreduce_layernorm_grads(self, freeze_model, tp_size):
        self.tp_size = tp_size
        self.pp_size = 1
        Utils.initialize_model_parallel(tensor_model_parallel_size=self.tp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model()
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)

        _allreduce_non_tensor_model_parallel_grads(
            [self.model], self.transformer_config, parallel_state.get_tensor_model_parallel_group()
        )

    @pytest.mark.parametrize(
        ("freeze_model", "pp_size", "share_embeddings"),
        [(True, 2, True), (False, 2, True), (True, 2, False), (False, 2, False)],
    )
    def test_allreduce_word_embedding_grads(self, freeze_model, pp_size, share_embeddings):
        self.tp_size = 1
        self.pp_size = pp_size
        Utils.initialize_model_parallel(pipeline_model_parallel_size=self.pp_size)
        model_parallel_cuda_manual_seed(123)

        self.init_model(share_embeddings)
        self.model.cuda()
        self.model.ddp_config = DistributedDataParallelConfig()

        for param in self.model.parameters():
            if freeze_model:
                param.requires_grad = False
            else:
                param.grad = torch.ones_like(param)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        embd_group = parallel_state.get_embedding_group()

        _allreduce_word_embedding_grads([self.model], self.transformer_config, embd_group, pp_group)
