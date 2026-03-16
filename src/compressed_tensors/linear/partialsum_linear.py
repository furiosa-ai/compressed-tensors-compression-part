"""
PartialSumLinear: nn.Linear-based module with partial sum QDQ for TP simulation.

Does NOT inherit from CompressedLinear. Instead, mirrors CompressedLinear's
from_linear pattern but uses its own forward that splits matmul into
num_ranks partial sums with QDQ on each partial result.

Created at model load time by apply_quantization_config when
PartialSumConfig is present in QuantizationConfig.
"""

from __future__ import annotations

import re
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear
from torch.nn.modules import Linear

from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.quantization.utils import is_fp4
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import (
    compute_dynamic_scales_and_zp,
    generate_gparam,
)
from compressed_tensors.utils import register_offload_parameter
from compressed_tensors.utils.offload import get_execution_device


__all__ = [
    "PartialSumLinear",
    "PartialSumMoeExperts",
    "is_partial_sum_target",
    "get_partial_sum_layers",
    "get_partial_sum_errors",
    "reset_partial_sum_errors",
]


class PartialSumLinear(Linear):
    """
    Linear module with partial sum QDQ for tensor parallelism simulation.

    Mirrors CompressedLinear's structure (from_linear, compressed weight handling)
    but does NOT inherit from CompressedLinear. Forward splits the matmul into
    num_ranks partial sums and applies QDQ to each partial result.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "PartialSumLinear should not be initialized directly. "
            "Use the from_linear method instead.",
            UserWarning,
        )

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        module: Linear,
        quantization_scheme: Optional[QuantizationScheme] = None,
        quantization_format: Optional[str] = None,
        num_ranks: int = 4,
        partial_sum_quant_args: Optional[QuantizationArgs] = None,
        collect_errors: bool = False,
    ):
        """
        Convert an nn.Linear to PartialSumLinear (in-place class swap).

        When quantization_scheme is provided (compressed model), initializes
        quantization params and sets up compressor for compressed weights.
        When quantization_scheme is None (W16A16 unquantized), keeps the
        original weight and only applies partial sum QDQ in forward.

        :param module: dense linear module to replace
        :param quantization_scheme: quantization config (None for W16A16)
        :param quantization_format: compression format (None for W16A16)
        :param num_ranks: number of simulated TP ranks
        :param partial_sum_quant_args: QDQ args for partial sum results
        :param collect_errors: whether to collect QDQ error metrics
        """
        module.__class__ = PartialSumLinear

        # partial sum specific attributes
        module.num_ranks = num_ranks
        module.partial_sum_quant_args = partial_sum_quant_args
        module.collect_errors = collect_errors
        module._partial_sum_errors = []

        if quantization_scheme is not None and quantization_format is not None:
            module.compressor = BaseCompressor.load_from_registry(
                quantization_format
            )
            init_device = get_execution_device(module)

            initialize_module_for_quantization(
                module, quantization_scheme, force_zero_point=False
            )

            compression_params: Dict[str, Tuple] = (
                module.compressor.compression_param_info(
                    module.weight.shape, quantization_scheme.weights
                )
            )

            delattr(module, "weight")

            for name, (shape, dtype) in compression_params.items():
                param = Parameter(
                    torch.empty(shape, device=init_device, dtype=dtype),
                    requires_grad=False,
                )
                register_offload_parameter(module, name, param)

            module.quantization_status = QuantizationStatus.COMPRESSED
        else:
            module.compressor = None
            module.quantization_status = QuantizationStatus.FROZEN

        if hasattr(module, "_old_forward"):
            module._old_forward = PartialSumLinear.forward.__get__(
                module, PartialSumLinear
            )

        return module

    def forward(self, input: Tensor) -> Tensor:
        """
        Decompress weights if needed, then perform partial sum matmul with QDQ.
        """
        if (
            self.quantization_status == QuantizationStatus.COMPRESSED
            and self.compressor is not None
        ):
            weight_data = self.compressor.decompress_module(self)
            param = Parameter(weight_data, requires_grad=False)
            register_offload_parameter(self, "weight", param)
            self.quantization_status = QuantizationStatus.FROZEN

        input_dim = input.shape[-1]
        output_dim = self.weight.shape[0]
        num_ranks = self.num_ranks

        if input_dim % num_ranks != 0:
            warnings.warn(
                f"PartialSumLinear: input_dim ({input_dim}) is not divisible by "
                f"num_ranks ({num_ranks}). Falling back to standard linear.",
                UserWarning,
            )
            return linear(input, self.weight, self.bias)

        shard_size = input_dim // num_ranks
        batch_shape = input.shape[:-1]

        input_2d = input.reshape(-1, input_dim)
        X_parallel = input_2d.view(-1, num_ranks, shard_size).permute(1, 0, 2)

        W_parallel = self.weight.view(output_dim, num_ranks, shard_size).permute(
            1, 2, 0
        )

        partial_results = torch.bmm(X_parallel, W_parallel)

        partial_results_qdq = _qdq_partial_sums(
            partial_results, self.partial_sum_quant_args
        )

        if self.collect_errors:
            with torch.no_grad():
                error = (partial_results_qdq - partial_results).abs().mean().item()
                self._partial_sum_errors.append(error)

        output = partial_results_qdq.sum(dim=0)

        if self.bias is not None:
            output = output + self.bias

        return output.view(*batch_shape, output_dim)

    def get_avg_error(self) -> float:
        if not self._partial_sum_errors:
            return 0.0
        return sum(self._partial_sum_errors) / len(self._partial_sum_errors)

    def reset_errors(self):
        self._partial_sum_errors.clear()


class PartialSumMoeExperts(nn.Module):
    """
    MoE Experts module with partial sum QDQ on down_proj for TP simulation.

    Mirrors CompressedMoeExperts's structure (from_moe, compressed weight handling)
    but does NOT inherit from CompressedMoeExperts. Forward applies partial sum
    QDQ only to the down_proj matmul; gate_proj and up_proj use standard F.linear.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "PartialSumMoeExperts should not be initialized directly. "
            "Use the from_moe method instead.",
            UserWarning,
        )

    @classmethod
    @torch.no_grad()
    def from_moe(
        cls,
        module: nn.Module,
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
        num_ranks: int = 4,
        partial_sum_quant_args: Optional[QuantizationArgs] = None,
        collect_errors: bool = False,
    ):
        """
        Convert an ExaoneMoeExperts to PartialSumMoeExperts (in-place class swap).

        Same pattern as CompressedMoeExperts.from_moe but stores partial sum
        config for down_proj.

        :param module: ExaoneMoeExperts module to replace
        :param quantization_scheme: quantization config for the module
        :param quantization_format: compression format module is stored as
        :param num_ranks: number of simulated TP ranks
        :param partial_sum_quant_args: QDQ args for partial sum results
        :param collect_errors: whether to collect QDQ error metrics
        """
        module.__class__ = PartialSumMoeExperts
        module.compressor = BaseCompressor.load_from_registry(quantization_format)
        init_device = get_execution_device(module)

        module.quantization_scheme = quantization_scheme

        # partial sum specific attributes (applied to down_proj only)
        module.num_ranks = num_ranks
        module.partial_sum_quant_args = partial_sum_quant_args
        module.collect_errors = collect_errors
        module._partial_sum_errors = []

        # Split gate_up_proj into gate_proj and up_proj
        gate_proj_weight, up_proj_weight = module.gate_up_proj.chunk(2, dim=1)
        down_proj_weight = module.down_proj

        delattr(module, "gate_up_proj")
        delattr(module, "down_proj")

        gate_proj_module = nn.Module()
        up_proj_module = nn.Module()
        down_proj_module = nn.Module()

        module.add_module("gate_proj", gate_proj_module)
        module.add_module("up_proj", up_proj_module)
        module.add_module("down_proj", down_proj_module)

        for submodule, weight in [
            (gate_proj_module, gate_proj_weight),
            (up_proj_module, up_proj_weight),
            (down_proj_module, down_proj_weight),
        ]:
            submodule.weight = Parameter(weight, requires_grad=False)

            initialize_module_for_quantization(
                submodule, quantization_scheme, force_zero_point=False
            )

            compression_params: Dict[str, Tuple] = (
                module.compressor.compression_param_info(
                    weight.shape, quantization_scheme.weights
                )
            )

            delattr(submodule, "weight")

            for name, (shape, dtype) in compression_params.items():
                param = Parameter(
                    torch.empty(shape, device=init_device, dtype=dtype),
                    requires_grad=False,
                )
                register_offload_parameter(submodule, name, param)

        module.quantization_status = QuantizationStatus.COMPRESSED

        if hasattr(module, "_old_forward"):
            module._old_forward = PartialSumMoeExperts.forward.__get__(
                module, PartialSumMoeExperts
            )

        return module

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """
        Decompresses weights, runs MoE forward with partial sum QDQ on down_proj.
        """
        if self.quantization_status == QuantizationStatus.COMPRESSED:
            self._decompressed_gate_proj = self.compressor.decompress_module(
                self.gate_proj
            )
            self._decompressed_up_proj = self.compressor.decompress_module(
                self.up_proj
            )
            self._decompressed_down_proj = self.compressor.decompress_module(
                self.down_proj
            )
            self.quantization_status = QuantizationStatus.FROZEN

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            )
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # gate_proj and up_proj: standard matmul
            gate = nn.functional.linear(
                current_state, self._decompressed_gate_proj[expert_idx]
            )
            up = nn.functional.linear(
                current_state, self._decompressed_up_proj[expert_idx]
            )
            current_hidden_states = self.act_fn(gate) * up

            # down_proj: partial sum matmul + QDQ
            current_hidden_states = self._partial_sum_linear(
                current_hidden_states,
                self._decompressed_down_proj[expert_idx],
            )

            current_hidden_states = (
                current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            )
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )

        return final_hidden_states

    def _partial_sum_linear(self, input: Tensor, weight: Tensor) -> Tensor:
        """
        Partial sum matmul + QDQ for a single expert's down_proj.

        :param input: (tokens, intermediate_dim)
        :param weight: (output_dim, intermediate_dim)
        """
        input_dim = input.shape[-1]
        output_dim = weight.shape[0]
        num_ranks = self.num_ranks

        if input_dim % num_ranks != 0:
            warnings.warn(
                f"PartialSumMoeExperts: input_dim ({input_dim}) is not divisible by "
                f"num_ranks ({num_ranks}). Falling back to standard linear for down_proj.",
                UserWarning,
            )
            return nn.functional.linear(input, weight)

        shard_size = input_dim // num_ranks
        batch_shape = input.shape[:-1]

        input_2d = input.reshape(-1, input_dim)
        X_parallel = input_2d.view(-1, num_ranks, shard_size).permute(1, 0, 2)

        W_parallel = weight.view(output_dim, num_ranks, shard_size).permute(1, 2, 0)

        partial_results = torch.bmm(X_parallel, W_parallel)

        partial_results_qdq = _qdq_partial_sums(
            partial_results, self.partial_sum_quant_args
        )

        if self.collect_errors:
            with torch.no_grad():
                error = (partial_results_qdq - partial_results).abs().mean().item()
                self._partial_sum_errors.append(error)

        output = partial_results_qdq.sum(dim=0)
        return output.view(*batch_shape, output_dim)

    def get_avg_error(self) -> float:
        if not self._partial_sum_errors:
            return 0.0
        return sum(self._partial_sum_errors) / len(self._partial_sum_errors)

    def reset_errors(self):
        self._partial_sum_errors.clear()


# ============================================================
# Standalone helpers
# ============================================================


def _qdq_partial_sums(
    partial_results: Tensor,
    quant_args: QuantizationArgs,
) -> Tensor:
    """
    Apply QDQ to partial sums.

    Handles GROUP and TENSOR_GROUP strategies:
    - GROUP (FP8): per-group scales only
    - TENSOR_GROUP (FP4): global_scale + per-group local scales
    """
    original_shape = partial_results.shape
    original_dtype = partial_results.dtype

    flat_results = partial_results.reshape(-1, original_shape[-1])

    global_scale = None
    if quant_args.strategy == QuantizationStrategy.TENSOR_GROUP:
        if is_fp4(quant_args):
            min_val = flat_results.min()
            max_val = flat_results.max()
            global_scale = generate_gparam(
                updated_min_val=min_val,
                updated_max_val=max_val,
                scale_data=FP8_E4M3_DATA,
                quant_data=FP4_E2M1_DATA,
            )
        else:
            warnings.warn(
                f"PartialSumLinear: TENSOR_GROUP with {quant_args.num_bits}-bit "
                f"{quant_args.type} is not supported for global_scale computation. "
                f"Only FP4 (NVFP4) is currently supported.",
                UserWarning,
            )

    scale, zero_point = compute_dynamic_scales_and_zp(
        value=flat_results,
        args=quant_args,
        module=None,
        global_scale=global_scale,
    )

    qdq_flat = fake_quantize(
        x=flat_results,
        scale=scale,
        zero_point=zero_point,
        args=quant_args,
        global_scale=global_scale,
    )

    return qdq_flat.reshape(original_shape).to(original_dtype)


def is_partial_sum_target(
    name: str,
    targets: Optional[List[str]],
    ignore: Optional[List[str]],
) -> bool:
    """
    Check if a module name matches partial_sum targets and is not ignored.

    :param name: full module name (e.g. "model.layers.0.self_attn.q_proj")
    :param targets: list of target patterns (exact names or 're:' regex). None = match all.
    :param ignore: list of ignore patterns (exact names or 're:' regex)
    """
    if targets is not None:
        if not _matches_any(name, targets):
            return False

    if ignore:
        if _matches_any(name, ignore):
            return False

    return True


def _matches_any(name: str, patterns: List[str]) -> bool:
    for pattern in patterns:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], name):
                return True
        elif pattern == name:
            return True
    return False


def get_partial_sum_layers(model) -> List[Tuple[str, nn.Module]]:
    """Get all PartialSumLinear and PartialSumMoeExperts layers in the model."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (PartialSumLinear, PartialSumMoeExperts))
    ]


def get_partial_sum_errors(model) -> Dict[str, float]:
    """Get per-layer average QDQ errors for PartialSumLinear layers."""
    errors = {}
    for name, module in get_partial_sum_layers(model):
        if module._partial_sum_errors:
            errors[name] = (
                sum(module._partial_sum_errors) / len(module._partial_sum_errors)
            )
    return errors


def reset_partial_sum_errors(model):
    """Reset collected error metrics for all PartialSumLinear layers."""
    for _, module in get_partial_sum_layers(model):
        module.reset_errors()
