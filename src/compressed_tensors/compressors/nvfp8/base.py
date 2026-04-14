# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.utils import TensorStateDict


__all__ = ["NVFP8QuantizationCompressor"]


@BaseCompressor.register(name=CompressionFormat.nvfp8_quantized.value)
class NVFP8QuantizationCompressor(BaseCompressor):
    """
    Compressor for NVFP8 quantized models (TENSOR_GROUP, group_size=16).

    Weights are stored as float8_e4m3fn with per-group-16 FP8 scales and a
    per-tensor float32 global_scale. The scale is cast to its declared
    scale_dtype (float8_e4m3fn) for storage, matching NVFP4 behavior.
    """

    @classmethod
    def _compress_scale(
        cls, scale: torch.Tensor, weights: QuantizationArgs
    ) -> torch.Tensor:
        scale_dtype = weights.scale_dtype or torch.float8_e4m3fn
        return scale.to(scale_dtype)

    @classmethod
    def _decompress_scale(cls, scale: torch.Tensor) -> torch.Tensor:
        return scale.to(torch.float32)

    @classmethod
    def compress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.get("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)
        zero_point = state_dict.get("weight_zero_point", None)
        weights = scheme.weights

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            zero_point=zero_point,
            args=weights,
            dtype=weights.pytorch_dtype(),
            global_scale=global_scale,
        )

        state_dict["weight"] = quantized_weight
        state_dict["weight_scale"] = cls._compress_scale(scale, weights)
        state_dict = cls._remove_symmetric_zp(state_dict, scheme)

        return state_dict

    @classmethod
    def decompress(
        cls, state_dict: TensorStateDict, scheme: QuantizationScheme
    ) -> TensorStateDict:
        state_dict = state_dict.copy()
        weight = state_dict.pop("weight")
        scale = state_dict.get("weight_scale")
        global_scale = state_dict.get("weight_global_scale", None)
        zero_point = state_dict.get("weight_zero_point", None)
        g_idx = state_dict.get("weight_g_idx", None)

        scale_float = cls._decompress_scale(scale)

        state_dict["weight"] = dequantize(
            x_q=weight,
            scale=scale_float,
            zero_point=zero_point,
            g_idx=g_idx,
            global_scale=global_scale,
        )

        return state_dict

    @classmethod
    def can_compress(cls, module_type: type, scheme: QuantizationScheme) -> bool:
        """NVFP8 matches FP8 with TENSOR_GROUP strategy and group_size=16."""
        return (
            module_type == torch.nn.Linear
            and scheme.weights is not None
            and scheme.weights.num_bits == 8
            and scheme.weights.type == QuantizationType.FLOAT.value
            and scheme.weights.strategy == QuantizationStrategy.TENSOR_GROUP.value
            and scheme.weights.group_size == 16
        )
