# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NVFP4+ compressor: FP4 with GROUP strategy, BF16 scales, group_size=16.

Unlike NVFP4PackedCompressor (TENSOR_GROUP with FP8 scales + FP32 global_scale),
NVFP4+ uses GROUP strategy with fixed BF16 scales and no global_scale.
"""

from typing import Dict, Optional, Tuple

import torch
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.compressors.quantized_compressors.base import (
    BaseQuantizationCompressor,
)
from compressed_tensors.config import CompressionFormat
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize, quantize
from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from torch import Tensor


__all__ = ["NVFP4PlusCompressor"]


@BaseCompressor.register(name=CompressionFormat.nvfp4plus_quantized.value)
class NVFP4PlusCompressor(BaseQuantizationCompressor):
    """
    NVFP4+ compressor.

    Differences from NVFP4PackedCompressor:
      - scale dtype: BF16 (fixed) instead of FP8
      - global_scale: not used (GROUP strategy, not TENSOR_GROUP)
      - group_size: 16 (same as NVFP4)
      - Storage: FP4 values packed into uint8 (same as NVFP4)
    """

    @property
    def compression_param_names(self) -> Tuple[str]:
        return (
            "weight_packed",
            "weight_scale",
            "weight_zero_point",
        )

    def compression_param_info(
        self,
        weight_shape: torch.Size,
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> Dict[str, Tuple[torch.Size, torch.dtype]]:
        output = {
            "weight_packed": (
                torch.Size((weight_shape[0], weight_shape[1] // 2)),
                torch.uint8,
            ),
        }
        return output

    def compress_weight(
        self,
        weight: Tensor,
        scale: Tensor,
        quantization_args: QuantizationArgs,
        device: Optional[torch.device] = None,
        zero_point: Optional[torch.Tensor] = None,
        g_idx: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if scale.dtype != torch.bfloat16:
            raise ValueError(
                f"NVFP4+ requires BF16 scales, but got {scale.dtype}."
            )

        quantized_weight = quantize(
            x=weight,
            scale=scale,
            global_scale=None,
            zero_point=zero_point,
            g_idx=g_idx,
            args=quantization_args,
        )

        compressed_dict = {}
        weight_packed = pack_fp4_to_uint8(quantized_weight)
        if device is not None:
            weight_packed = weight_packed.to(device)
        compressed_dict["weight_packed"] = weight_packed
        return compressed_dict

    def decompress_weight(
        self,
        compressed_data: Dict[str, Tensor],
        quantization_args: Optional[QuantizationArgs] = None,
    ) -> torch.Tensor:
        weight = compressed_data["weight_packed"]
        scale = compressed_data["weight_scale"]
        m, n = weight.shape
        unpacked = unpack_fp4_from_uint8(weight, m, n * 2)
        decompressed_weight = dequantize(
            x_q=unpacked,
            scale=scale,
            global_scale=None,
            dtype=unpacked.dtype,
        )
        return decompressed_weight
