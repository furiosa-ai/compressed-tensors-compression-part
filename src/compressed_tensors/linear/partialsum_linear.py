"""
PartialSumCompressedLinear: CompressedLinear with partial sum QDQ for TP simulation.

Supports multiple quantization types:
- FP8 (float8_e4m3fn, float8_e5m2)
- FP4 (NVFP4)
- Custom QuantizationArgs
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Union
import fnmatch

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear

from compressed_tensors.linear.compressed_linear import CompressedLinear
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
)
from compressed_tensors.quantization.lifecycle.forward import fake_quantize
from compressed_tensors.quantization.utils import (
    compute_dynamic_scales_and_zp,
    generate_gparam,
)
from compressed_tensors.utils import register_offload_parameter


# ============================================================
# Quantization Presets
# ============================================================

class QuantPreset(str, Enum):
    """Predefined quantization presets for partial sum QDQ"""
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4_NVFP4 = "fp4_nvfp4"      # NVFP4 format (TENSOR_GROUP, FP8 scale)


@dataclass
class QuantPresetConfig:
    """Configuration for a quantization preset"""
    num_bits: int
    quant_type: QuantizationType
    symmetric: bool
    strategy: QuantizationStrategy
    group_size: int
    
    def to_quant_args(
        self, 
        strategy: Optional[QuantizationStrategy], 
        group_size: Optional[int]
    ) -> QuantizationArgs:
        """Convert preset config to QuantizationArgs"""
        return QuantizationArgs(
            num_bits=self.num_bits,
            type=self.quant_type,
            symmetric=self.symmetric,
            strategy=strategy,
            group_size=group_size,
            dynamic=True,
        )


# Preset configurations
QUANT_PRESETS = {
    QuantPreset.FP8_E4M3: QuantPresetConfig(
        num_bits=8,
        quant_type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
    ),
    QuantPreset.FP8_E5M2: QuantPresetConfig(
        num_bits=8,
        quant_type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.GROUP,
        group_size=32,
    ),
    QuantPreset.FP4_NVFP4: QuantPresetConfig(
        num_bits=4,
        quant_type=QuantizationType.FLOAT,
        symmetric=True,
        strategy=QuantizationStrategy.TENSOR_GROUP,
        group_size=16,
    ),
}


def get_quant_args(
    quant_config: Union[str, QuantPreset, QuantizationArgs],
) -> QuantizationArgs:
    """
    Convert quant_config to QuantizationArgs.
    
    - str / QuantPreset: use the preset's default strategy/group_size
    - QuantizationArgs: return as-is
    """
    if isinstance(quant_config, QuantizationArgs):
        return quant_config
    
    if isinstance(quant_config, str):
        quant_config = QuantPreset(quant_config)
    
    preset_config = QUANT_PRESETS.get(quant_config)
    if preset_config is None:
        raise ValueError(f"Unknown quantization preset: {quant_config}")
    
    return preset_config.to_quant_args(
        strategy=preset_config.strategy,
        group_size=preset_config.group_size,
    )


# ============================================================
# PartialSumCompressedLinear
# ============================================================

class PartialSumCompressedLinear(CompressedLinear):
    """
    CompressedLinear with partial sum QDQ for tensor parallelism simulation.
    
    Supports multiple quantization formats:
    - FP8 (e4m3, e5m2)
    - FP4 (NVFP4)
    - Custom QuantizationArgs
    """

    # Instance attributes
    num_ranks: int
    quant_args: QuantizationArgs
    collect_errors: bool
    _quant_errors: list

    @classmethod
    @torch.no_grad()
    def from_compressed_linear(
        cls,
        module: CompressedLinear,
        num_ranks: int = 4,
        quant_config: Union[str, QuantPreset, QuantizationArgs] = QuantPreset.FP8_E4M3,
        collect_errors: bool = False,
    ) -> "PartialSumCompressedLinear":
        """
        Convert an existing CompressedLinear to PartialSumCompressedLinear (in-place).
        Forward chain (input activation QDQ + accelerate hooks) preserved.
        """
        module.__class__ = cls
        
        quant_args = get_quant_args(quant_config)
        
        module.num_ranks = num_ranks
        module.quant_args = quant_args
        module.collect_errors = collect_errors
        module._quant_errors = []
        
        cls._fix_forward_chain(module)
        
        return module

    @staticmethod
    def _fix_forward_chain(module: "PartialSumCompressedLinear") -> None:
        """
        Re-wrap forward chain so that wrapped_forward captures
        PartialSumCompressedLinear.forward instead of CompressedLinear.forward.
        
        With accelerate:
            module.forward [accelerate] -> module._old_forward [wrapped_forward]
              -> input QDQ -> forward_func_orig -> PartialSumCompressedLinear.forward
        Without accelerate:
            module.forward [wrapped_forward] -> input QDQ -> forward_func_orig
        """
        from compressed_tensors.quantization.lifecycle.forward import (
            wrap_module_forward_quantized,
        )
        
        scheme = getattr(module, "quantization_scheme", None)
        has_accelerate = hasattr(module, "_old_forward")
        
        if scheme is not None:
            if has_accelerate:
                accel_forward = module.__dict__.pop("forward", None)
                if hasattr(module, "_old_forward"):
                    delattr(module, "_old_forward")
                
                wrap_module_forward_quantized(module, scheme)
                
                module._old_forward = module.__dict__.pop("forward")
                if accel_forward is not None:
                    module.forward = accel_forward
            else:
                if "forward" in module.__dict__:
                    del module.__dict__["forward"]
                wrap_module_forward_quantized(module, scheme)
        else:
            if has_accelerate:
                module._old_forward = PartialSumCompressedLinear.forward.__get__(
                    module, PartialSumCompressedLinear
                )

    def forward(self, input: Tensor) -> Tensor:
        """Forward with partial sum quantization for TP simulation."""
        
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
        X_reshaped = input_2d.view(-1, num_ranks, shard_size)
        X_parallel = X_reshaped.permute(1, 0, 2)  # (R, B, S)

        W_reshaped = self.weight.view(output_dim, num_ranks, shard_size)
        W_parallel = W_reshaped.permute(1, 2, 0)  # (R, S, O)

        partial_results = torch.bmm(X_parallel, W_parallel)

        partial_results_qdq = self._qdq_partial_sums(partial_results)

        if self.collect_errors:
            with torch.no_grad():
                error = (partial_results_qdq - partial_results).abs().mean().item()
                self._quant_errors.append(error)

        output = partial_results_qdq.sum(dim=0)

        if self.bias is not None:
            output = output + self.bias

        return output.view(*batch_shape, output_dim)

    def _qdq_partial_sums(self, partial_results: Tensor) -> Tensor:
        """
        Apply QDQ to partial sums using compressed-tensors.
        
        Handles both GROUP and TENSOR_GROUP strategies:
        - GROUP (FP8): per-group scales only
        - TENSOR_GROUP (FP4): global_scale + per-group local scales
        """
        original_shape = partial_results.shape
        original_dtype = partial_results.dtype

        flat_results = partial_results.reshape(-1, original_shape[-1])
        
        global_scale = None
        if self.quant_args.strategy == QuantizationStrategy.TENSOR_GROUP:
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
            args=self.quant_args,
            module=None,
            global_scale=global_scale,
        )

        qdq_flat = fake_quantize(
            x=flat_results,
            scale=scale,
            zero_point=zero_point,
            args=self.quant_args,
            global_scale=global_scale,
        )

        return qdq_flat.reshape(original_shape).to(original_dtype)

    def get_avg_error(self) -> float:
        """Get average quantization error."""
        if not self._quant_errors:
            return 0.0
        return sum(self._quant_errors) / len(self._quant_errors)

    def reset_metrics(self):
        """Reset collected metrics."""
        self._quant_errors.clear()
    
    def get_quant_info(self) -> dict:
        """Get quantization configuration info."""
        return {
            "num_ranks": self.num_ranks,
            "num_bits": self.quant_args.num_bits,
            "type": str(self.quant_args.type),
            "strategy": str(self.quant_args.strategy),
            "group_size": self.quant_args.group_size,
            "symmetric": self.quant_args.symmetric,
        }


# ============================================================
# Utility functions
# ============================================================

def _get_parent_module(model, name: str):
    """Get parent module from full name."""
    parts = name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def replace_with_partialsum_linear(
    model,
    num_ranks: int = 4,
    quant_config: Union[str, QuantPreset, QuantizationArgs] = QuantPreset.FP8_E4M3,
    collect_errors: bool = False,
    target_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
) -> int:
    """
    Replace CompressedLinear modules with PartialSumCompressedLinear.
    
    Args:
        model: Model to modify
        num_ranks: Number of simulated TP ranks
        quant_config: str ("fp8_e4m3", "int8", ...) / QuantPreset / QuantizationArgs
        collect_errors: Whether to collect error metrics
        target_patterns: Glob patterns for layers to include (None = all)
        exclude_patterns: Glob patterns for layers to exclude
    """
    replaced_count = 0
    modules_to_replace = []
    
    for name, module in model.named_modules():
        if not isinstance(module, CompressedLinear):
            continue
        if isinstance(module, PartialSumCompressedLinear):
            continue
            
        should_replace = True
        
        if target_patterns is not None:
            should_replace = any(
                fnmatch.fnmatch(name, p) for p in target_patterns
            )
        
        if exclude_patterns is not None:
            if any(fnmatch.fnmatch(name, p) for p in exclude_patterns):
                should_replace = False
        
        if should_replace:
            modules_to_replace.append(name)
    
    for name in modules_to_replace:
        parent = _get_parent_module(model, name)
        child_name = name.split('.')[-1]
        module = getattr(parent, child_name)
        
        PartialSumCompressedLinear.from_compressed_linear(
            module,
            num_ranks=num_ranks,
            quant_config=quant_config,
            collect_errors=collect_errors,
        )
        replaced_count += 1
    
    quant_args = get_quant_args(quant_config)
    print(f"Replaced {replaced_count} layers -> PartialSumCompressedLinear")
    print(f"   Config: {quant_args.num_bits}bit {quant_args.type}, "
          f"strategy={quant_args.strategy}, group_size={quant_args.group_size}, "
          f"num_ranks={num_ranks}")
    
    return replaced_count


def get_partialsum_layers(model) -> List[tuple]:
    """
    Get all PartialSumCompressedLinear layers in the model.
    
    Returns:
        List of (name, module) tuples
    """
    return [
        (name, module) 
        for name, module in model.named_modules() 
        if isinstance(module, PartialSumCompressedLinear)
    ]


def get_total_quant_error(model) -> float:
    """
    Get average quantization error across all PartialSumCompressedLinear layers.
    """
    layers = get_partialsum_layers(model)
    if not layers:
        return 0.0
    
    total_errors = []
    for _, layer in layers:
        total_errors.extend(layer._quant_errors)
    
    if not total_errors:
        return 0.0
    return sum(total_errors) / len(total_errors)


def reset_all_errors(model):
    """Reset errors for all PartialSumCompressedLinear layers."""
    for _, layer in get_partialsum_layers(model):
        layer.reset_metrics()


__all__ = [
    "PartialSumCompressedLinear",
    "QuantPreset",
    "get_quant_args",
    "replace_with_partialsum_linear",
    "get_partialsum_layers",
    "get_total_quant_error",
    "reset_all_errors",
]
