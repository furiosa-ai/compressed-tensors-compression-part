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

import torch

from compressed_tensors.compressors.quantized_compressors.nvfp4plus_quantized import (
    NVFP4PlusCompressor,
)
from compressed_tensors.quantization.quant_args import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.utils.helpers import (
    calculate_qparams,
)


# ---------------------------------------------------------------------------
# calculate_qparams for NVFP4+
# ---------------------------------------------------------------------------


class TestCalculateQparamsNVFP4Plus:
    def _make_nvfp4plus_args(self):
        return QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.GROUP,
            group_size=16,
            symmetric=True,
        )

    def test_scale_dtype_is_bf16(self):
        """NVFP4+ scales must be BF16, not FP8."""
        args = self._make_nvfp4plus_args()
        min_vals = torch.tensor([-3.0, -2.0], dtype=torch.bfloat16)
        max_vals = torch.tensor([3.0, 2.0], dtype=torch.bfloat16)

        scales, zero_points = calculate_qparams(min_vals, max_vals, args)

        assert scales.dtype != torch.float8_e4m3fn, "NVFP4+ scales must not be FP8"
        assert zero_points.dtype == torch.bfloat16

    def test_no_global_scale_influence(self):
        """global_scale should not affect NVFP4+ (GROUP) path."""
        args = self._make_nvfp4plus_args()
        min_vals = torch.tensor([-4.0], dtype=torch.bfloat16)
        max_vals = torch.tensor([4.0], dtype=torch.bfloat16)

        scales_no_gs, _ = calculate_qparams(min_vals, max_vals, args, global_scale=None)
        fake_gs = torch.tensor(0.5, dtype=torch.float32)
        scales_with_gs, _ = calculate_qparams(
            min_vals, max_vals, args, global_scale=fake_gs
        )

        assert torch.equal(scales_no_gs, scales_with_gs)

# ---------------------------------------------------------------------------
# NVFP4PlusCompressor compress/decompress round-trip
# ---------------------------------------------------------------------------


class TestNVFP4PlusCompressDecompress:
    def _make_args(self):
        return QuantizationArgs(
            num_bits=4,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy.GROUP,
            group_size=16,
            symmetric=True,
        )

    def test_compress_produces_packed_uint8(self):
        args = self._make_args()
        compressor = NVFP4PlusCompressor()

        weight = torch.randn(4, 32, dtype=torch.bfloat16)
        scale = torch.ones(4, 2, dtype=torch.bfloat16)

        result = compressor.compress_weight(
            weight=weight, scale=scale, quantization_args=args
        )

        assert "weight_packed" in result
        assert result["weight_packed"].dtype == torch.uint8
        assert result["weight_packed"].shape == (4, 16)  # 32 / 2

    def test_round_trip(self):
        """Compress then decompress should produce valid FP4 values."""
        args = self._make_args()
        compressor = NVFP4PlusCompressor()

        rows, cols = 4, 32
        weight = torch.randn(rows, cols, dtype=torch.bfloat16)
        scale = torch.ones(rows, cols // 16, dtype=torch.bfloat16) * 0.5

        compressed = compressor.compress_weight(
            weight=weight, scale=scale, quantization_args=args
        )

        compressed_data = {
            "weight_packed": compressed["weight_packed"],
            "weight_scale": scale,
        }
        decompressed = compressor.decompress_weight(
            compressed_data=compressed_data, quantization_args=args
        )

        assert decompressed.shape == (rows, cols)
        assert decompressed.dtype == torch.bfloat16

    def test_compression_param_names(self):
        compressor = NVFP4PlusCompressor()
        names = compressor.compression_param_names
        assert "weight_packed" in names
        assert "weight_scale" in names
        assert "weight_zero_point" in names
        assert "weight_global_scale" not in names

    def test_compression_param_info(self):
        compressor = NVFP4PlusCompressor()
        info = compressor.compression_param_info(torch.Size([8, 64]))
        assert info["weight_packed"] == (torch.Size([8, 32]), torch.uint8)
