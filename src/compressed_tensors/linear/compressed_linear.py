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

import warnings
from typing import Dict, Tuple

import torch
import torch.nn as nn
from compressed_tensors.compressors.base import BaseCompressor
from compressed_tensors.quantization import (
    QuantizationScheme,
    QuantizationStatus,
    initialize_module_for_quantization,
)
from compressed_tensors.utils import register_offload_parameter
from compressed_tensors.utils.offload import get_execution_device
from torch import Tensor
from torch.nn import Parameter
from torch.nn.functional import linear
from torch.nn.modules import Linear


class CompressedLinear(Linear):
    """
    Wrapper module for running a compressed forward pass of a quantized Linear module.
    The wrapped layer will decompressed on each forward call.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "CompressedLinear should not be initialized directly. "
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
    ):
        """
        :param module: dense linear module to replace
        :param quantization_scheme: quantization config for the module to wrap
        :param quantization_format: compression format module is stored as
        :return: CompressedLinear module wrapping the input module
        """
        module.__class__ = CompressedLinear
        module.compressor = BaseCompressor.load_from_registry(quantization_format)
        init_device = get_execution_device(module)

        # this will initialize all the scales and zero points
        initialize_module_for_quantization(
            module, quantization_scheme, force_zero_point=False
        )

        # get the shape and dtype of compressed parameters
        compression_params: Dict[str, Tuple] = module.compressor.compression_param_info(
            module.weight.shape, quantization_scheme.weights
        )

        # no need for this once quantization is initialized, will be replaced
        # with the compressed parameter
        delattr(module, "weight")

        # populate compressed weights and quantization parameters
        for name, (shape, dtype) in compression_params.items():
            param = Parameter(
                torch.empty(shape, device=init_device, dtype=dtype), requires_grad=False
            )
            register_offload_parameter(module, name, param)

        # mark module as compressed
        module.quantization_status = QuantizationStatus.COMPRESSED

        # handles case where forward is wrapped in new_forward by accelerate hooks
        if hasattr(module, "_old_forward"):
            module._old_forward = CompressedLinear.forward.__get__(
                module, CompressedLinear
            )

        return module

    def forward(self, input: Tensor) -> Tensor:
        """
        Decompresses the weight, then runs the wrapped forward pass
        """
        if self.quantization_status == QuantizationStatus.COMPRESSED:
            weight_data = self.compressor.decompress_module(self)
            param = Parameter(weight_data, requires_grad=False)
            register_offload_parameter(self, "weight", param)

            self.quantization_status = QuantizationStatus.FROZEN

        return linear(input, self.weight, self.bias)


class CompressedMoeExperts(nn.Module):
    """
    Wrapper module for running a compressed forward pass of a quantized MoE Experts module.
    The wrapped layer will be decompressed on each forward call.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        warnings.warn(
            "CompressedMoeExperts should not be initialized directly. "
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
    ):
        """
        :param module: ExaoneMoeExperts module to replace
        :param quantization_scheme: quantization config for the module to wrap
        :param quantization_format: compression format module is stored as
        :return: CompressedMoeExperts module wrapping the input module
        """
        module.__class__ = CompressedMoeExperts
        module.compressor = BaseCompressor.load_from_registry(quantization_format)
        init_device = get_execution_device(module)

        # Store original attributes needed for forward
        module.quantization_scheme = quantization_scheme

        # Split gate_up_proj into gate_proj and up_proj
        # gate_up_proj shape: (num_experts, 2 * intermediate_dim, hidden_dim)
        gate_proj, up_proj = module.gate_up_proj.chunk(2, dim=1)
        # gate_proj, up_proj shape: (num_experts, intermediate_dim, hidden_dim)

        # Initialize quantization for gate_proj
        gate_proj_compression_params: Dict[str, Tuple] = module.compressor.compression_param_info(
            gate_proj.shape, quantization_scheme.weights
        )

        # Initialize quantization for up_proj
        up_proj_compression_params: Dict[str, Tuple] = module.compressor.compression_param_info(
            up_proj.shape, quantization_scheme.weights
        )

        # Initialize quantization for down_proj
        down_proj_compression_params: Dict[str, Tuple] = module.compressor.compression_param_info(
            module.down_proj.shape, quantization_scheme.weights
        )

        # Remove original parameters
        delattr(module, "gate_up_proj")
        delattr(module, "down_proj")

        # Create submodules to hold compressed parameters
        gate_proj_module = nn.Module()
        up_proj_module = nn.Module()
        down_proj_module = nn.Module()

        # Register submodules
        module.add_module("gate_proj", gate_proj_module)
        module.add_module("up_proj", up_proj_module)
        module.add_module("down_proj", down_proj_module)

        # Populate compressed weights for gate_proj
        for name, (shape, dtype) in gate_proj_compression_params.items():
            param = Parameter(
                torch.empty(shape, device=init_device, dtype=dtype), requires_grad=False
            )
            register_offload_parameter(gate_proj_module, name, param)

        # Populate compressed weights for up_proj
        for name, (shape, dtype) in up_proj_compression_params.items():
            param = Parameter(
                torch.empty(shape, device=init_device, dtype=dtype), requires_grad=False
            )
            register_offload_parameter(up_proj_module, name, param)

        # Populate compressed weights for down_proj
        for name, (shape, dtype) in down_proj_compression_params.items():
            param = Parameter(
                torch.empty(shape, device=init_device, dtype=dtype), requires_grad=False
            )
            register_offload_parameter(down_proj_module, name, param)

        # Mark module as compressed
        module.quantization_status = QuantizationStatus.COMPRESSED

        # handles case where forward is wrapped in new_forward by accelerate hooks
        if hasattr(module, "_old_forward"):
            module._old_forward = CompressedMoeExperts.forward.__get__(
                module, CompressedMoeExperts
            )

        return module

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """
        Decompresses the weights, then runs the MoE forward pass
        """
        if self.quantization_status == QuantizationStatus.COMPRESSED:
            # Decompress gate_proj
            gate_proj_data = self.compressor.decompress_module(self.gate_proj)
            self._decompressed_gate_proj = gate_proj_data

            # Decompress up_proj
            up_proj_data = self.compressor.decompress_module(self.up_proj)
            self._decompressed_up_proj = up_proj_data

            # Decompress down_proj
            down_proj_data = self.compressor.decompress_module(self.down_proj)
            self._decompressed_down_proj = down_proj_data

            self.quantization_status = QuantizationStatus.FROZEN

        # Original ExaoneMoeExperts forward logic (modified to use separate gate_proj and up_proj)
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate = nn.functional.linear(current_state, self._decompressed_gate_proj[expert_idx])
            up = nn.functional.linear(current_state, self._decompressed_up_proj[expert_idx])
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self._decompressed_down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states