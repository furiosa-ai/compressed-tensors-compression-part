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
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear
from torch.nn.modules import Linear

from compressed_tensors.compressors.base import BaseCompressor
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
        quantization_scheme: QuantizationScheme,
        quantization_format: str,
        num_ranks: int = 4,
        partial_sum_quant_args: Optional[QuantizationArgs] = None,
        collect_errors: bool = False,
    ):
        """
        Convert an nn.Linear to PartialSumLinear (in-place class swap).

        Same pattern as CompressedLinear.from_linear: initializes quantization
        params, sets up compressor for compressed weights, replaces weight with
        compressed parameters.

        :param module: dense linear module to replace
        :param quantization_scheme: quantization config for the module
        :param quantization_format: compression format module is stored as
        :param num_ranks: number of simulated TP ranks
        :param partial_sum_quant_args: QDQ args for partial sum results
        :param collect_errors: whether to collect QDQ error metrics
        """
        module.__class__ = PartialSumLinear
        module.compressor = BaseCompressor.load_from_registry(quantization_format)
        init_device = get_execution_device(module)

        # partial sum specific attributes
        module.num_ranks = num_ranks
        module.partial_sum_quant_args = partial_sum_quant_args
        module.collect_errors = collect_errors
        module._partial_sum_errors = []

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

        if hasattr(module, "_old_forward"):
            module._old_forward = PartialSumLinear.forward.__get__(
                module, PartialSumLinear
            )

        return module

    def forward(self, input: Tensor) -> Tensor:
        """
        Decompress weights if needed, then perform partial sum matmul with QDQ.
        """
        if self.quantization_status == QuantizationStatus.COMPRESSED:
            weight_data = self.compressor.decompress_module(self)
            param = Parameter(weight_data, requires_grad=False)
            register_offload_parameter(self, "weight", param)
            self.quantization_status = QuantizationStatus.FROZEN

        input_dim = input.shape[-1]
        output_dim = self.weight.shape[0]
        num_ranks = self.num_ranks

        if input_dim % num_ranks != 0:
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
        min_val = flat_results.min()
        max_val = flat_results.max()
        global_scale = generate_gparam(
            updated_min_val=min_val,
            updated_max_val=max_val,
            scale_data=FP8_E4M3_DATA,
            quant_data=FP4_E2M1_DATA,
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


def get_partial_sum_layers(model) -> List[Tuple[str, PartialSumLinear]]:
    """Get all PartialSumLinear layers in the model."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, PartialSumLinear)
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
