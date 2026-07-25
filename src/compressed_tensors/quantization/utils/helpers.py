# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import math

import torch
from compressed_tensors.quantization.quant_args import (
    FP4_E2M1_DATA,
    FP8_E4M3_DATA,
    FloatArgs,
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
    round_to_quantized_type_dtype,
)
from compressed_tensors.quantization.utils.mxfp_utils import (
    generate_mx_scales,
    maybe_convert_from_mx_exp,
    should_generate_mx_scales,
)
from loguru import logger
from torch import FloatTensor, IntTensor, Tensor
from torch.nn import Module


__all__ = [
    "is_module_quantized",
    "is_model_quantized",
    "module_type",
    "get_torch_bit_depth",
    "can_quantize",
    "KV_CACHE_TARGETS",
    "compute_dynamic_scales_and_zp",
    "calculate_range",
    "calculate_qparams",
    "generate_gparam",
    "strategy_cdiv",
    "calculate_block_padding",
    "maybe_pad_tensor_for_block_quant",
    "scale_search_offset",
]

# target the self_attn layer
# QuantizedKVParameterCache is responsible for obtaining the k_scale and v_scale
KV_CACHE_TARGETS = ["re:.*self_attn$"]

_LOGGER: logging.Logger = logging.getLogger(__name__)


def calculate_qparams(
    min_vals: Tensor,
    max_vals: Tensor,
    quantization_args: QuantizationArgs,
    global_scale: Tensor | None = None,
) -> tuple[FloatTensor, IntTensor]:
    """
    :param min_vals: tensor of min value(s) to calculate scale(s) and zero point(s)
        from
    :param max_vals: tensor of max value(s) to calculate scale(s) and zero point(s)
        from
    :param quantization_args: settings to quantization
    :param global_scale: additional global scale to scale the locally generated scale
        currently only applied/supported for Fp4

    :return: tuple of the calculated scale(s) and zero point(s). For FP4, the calculated
        scale is of dtype FP8
    """
    # based on the implementations for consuming quantized values,
    # 0.0 must always be representable within the quantized range
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))

    device = min_vals.device

    bit_min, bit_max = calculate_range(quantization_args, device)
    bit_range = bit_max - bit_min

    # 1. Generate scale and zero-point
    if quantization_args.symmetric:
        max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
        if should_generate_mx_scales(args=quantization_args):
            scales = generate_mx_scales(
                x=max_val_pos, num_bits=quantization_args.num_bits
            )
        else:
            scales = max_val_pos / (float(bit_range) / 2)
        zero_points = torch.zeros(scales.shape, device=device, dtype=min_vals.dtype)
    else:
        if (
            quantization_args.num_bits == 4
            and quantization_args.type == QuantizationType.FLOAT
        ):
            raise NotImplementedError(
                "Asymmetric Quantization is not supported for FP4"
            )
        scales = (max_vals - min_vals) / float(bit_range)
        zero_points = bit_min - (min_vals / scales)
        zero_points = torch.clamp(zero_points, bit_min, bit_max)

    # 2. Conditionally scale the generated local scale by a global_scale
    if global_scale is not None:
        scales = global_scale * scales

    # 3. Conditionally round the scale to the quantized dtype, if scale_dtype is set
    if quantization_args.scale_dtype is not None:
        scales = round_to_quantized_type_dtype(
            scales, dtype=quantization_args.scale_dtype
        )

    # 4. Optionally remove exponent
    scales = maybe_convert_from_mx_exp(quantization_args, scales)

    # 5. Update any 0s with small values to
    # prevent div by 0
    eps = _get_dtype_eps(
        dtype=(
            quantization_args.scale_dtype
            if quantization_args.scale_dtype is not None
            else scales.dtype
        )
    )
    scales = torch.where(
        scales == 0,
        torch.tensor(eps, dtype=scales.dtype, device=device),
        scales,
    )

    # 6. Round the zp to zp_dtype
    zero_points = round_to_quantized_type_dtype(
        zero_points, dtype=quantization_args.zp_dtype, cast_to_original_dtype=False
    )

    if scales.ndim == 0:
        scales = scales.reshape(1)
        zero_points = zero_points.reshape(1)

    return scales, zero_points


@torch.no_grad()
def scale_search_offset(
    x_blocks: Tensor,
    scale: Tensor,
    args: QuantizationArgs,
    global_scale: Tensor | None = None,
) -> Tensor:
    """ScaleSearch (SYB): replace each per-block scale with the E4M3-representable
    neighbor that minimizes the block quant MSE.

    The neighbor is obtained by adding an integer offset f to the int8 bit-pattern of
    the standard (already E4M3-rounded) block scale, scanning f in
    [args.scale_search_fmin, args.scale_search_fmax]. The effective QDQ scale is
    ``scale / global_scale`` (matching the forward path), so the MSE is forward-exact.
    Element quantization uses ``args`` so this works for both NVFP8 (E4M3 element) and
    NVFP4 (E2M1 element); the block scale itself is E4M3 in both.

    :param x_blocks: grouped tensor of shape ``(*scale.shape, group_size)``
    :param scale: standard per-block scale (E4M3), shape ``(*qparam_shape,)``
    :param args: quantization args carrying scale_search_fmin/fmax and element type
    :param global_scale: optional per-tensor global scale (FP4/FP8 two-level scheme)
    :return: searched scale, same shape/dtype as ``scale``
    """
    from compressed_tensors.quantization.quant_args import (
        round_to_quantized_type_args,
    )

    if x_blocks.shape[:-1] != scale.shape:
        raise ValueError(
            f"scale_search_offset expects x_blocks {tuple(x_blocks.shape)} to be "
            f"scale {tuple(scale.shape)} + (group_size,)"
        )
    f_min = int(getattr(args, "scale_search_fmin", -2))
    f_max = int(getattr(args, "scale_search_fmax", 8))
    e4m3 = FP8_E4M3_DATA.dtype  # block scale dtype (both NVFP4 and NVFP8)
    work_dtype = x_blocks.dtype if x_blocks.is_floating_point() else torch.float32
    x = x_blocks.to(work_dtype)
    q_min, q_max = calculate_range(args, x.device)  # element range: FP4 +/-6, FP8 +/-448
    gs = (
        global_scale.to(work_dtype).flatten()[0]
        if global_scale is not None
        else torch.ones((), dtype=work_dtype, device=x.device)
    )

    def block_mse(scale_stored: Tensor) -> Tensor:
        eff = (scale_stored / gs).unsqueeze(-1)
        q = round_to_quantized_type_args(x / eff, args, q_min, q_max) * eff
        return ((q - x) ** 2).mean(dim=-1, keepdim=True)

    s0 = scale.to(e4m3)  # standard scale is already E4M3-rounded
    bits0 = s0.view(torch.uint8).to(torch.int16).unsqueeze(-1)  # (*qparam, 1)
    best_mse = block_mse(scale.to(work_dtype))
    best_bits = bits0.clone()
    for f in range(f_min, f_max + 1):
        if f == 0:
            continue
        # valid positive E4M3 bit-patterns are [1, 126]; 0 -> div0, 127 -> NaN
        bits = (bits0 + f).clamp(1, 126)
        cand = bits.squeeze(-1).to(torch.uint8).contiguous().view(e4m3).to(work_dtype)
        mse = block_mse(cand)
        take = mse < best_mse
        best_mse = torch.where(take, mse, best_mse)
        best_bits = torch.where(take, bits, best_bits)
    new_scale = (
        best_bits.squeeze(-1).clamp(1, 126).to(torch.uint8).contiguous().view(e4m3)
    )
    return new_scale.to(scale.dtype)


def compute_dynamic_scales_and_zp(
    value: Tensor,
    args: QuantizationArgs,
    module: torch.nn.Module,
    global_scale: Tensor | None = None,
):
    """
    Returns the computed scales and zero points for dynamic activation
    quantization.

    :param value: tensor to calculate quantization parameters for
    :param args: quantization args
    :param reduce_dims: optional tuple of dimensions to reduce along,
        returned scale and zero point will be shaped (1,) along the
        reduced dimensions
    :return: tuple of scale and zero point derived from the observed tensor
    """

    keep_dims = True
    if args.strategy == QuantizationStrategy.TOKEN:
        dim = {0, 1}
        reduce_dims = tuple(idx for idx in range(value.ndim) if idx not in dim)
    elif args.strategy == QuantizationStrategy.TENSOR:
        reduce_dims = None
    elif args.strategy in (
        QuantizationStrategy.TENSOR_GROUP,
        QuantizationStrategy.GROUP,
    ):

        reduce_dims = -1
        keep_dims = False

        reshaped_dims = (
            math.ceil(value.shape[-1] / args.group_size),
            args.group_size,
        )
        value = value.unflatten(-1, reshaped_dims)

    else:
        supported_strategies = (
            QuantizationStrategy.TOKEN,
            QuantizationStrategy.TENSOR,
            QuantizationStrategy.TENSOR_GROUP,
            QuantizationStrategy.GROUP,
        )
        raise ValueError(
            "Dynamic quantization is only supported for ",
            f"{supported_strategies}",
        )

    if not reduce_dims:
        min_val, max_val = torch.aminmax(value)
    else:
        min_val = torch.amin(value, dim=reduce_dims, keepdims=keep_dims)
        max_val = torch.amax(value, dim=reduce_dims, keepdims=keep_dims)

    # For TENSOR_GROUP with no static global_scale (dynamic=True, no calibration),
    # compute global_scale dynamically from the current input's distribution.
    if (
        args.strategy == QuantizationStrategy.TENSOR_GROUP
        and global_scale is None
    ):
        quant_data = FP8_E4M3_DATA if args.num_bits == 8 else FP4_E2M1_DATA
        global_scale = generate_gparam(
            min_val.amin().reshape(1),
            max_val.amax().reshape(1),
            quant_data=quant_data,
        )

    scales, zero_points = calculate_qparams(
        min_val, max_val, args, global_scale=global_scale
    )
    # ScaleSearch (SYB) on dynamic activation scales: `value` is already grouped as
    # (*scales.shape, group_size) for GROUP / TENSOR_GROUP above.
    if getattr(args, "scale_search", False) and args.strategy in (
        QuantizationStrategy.GROUP,
        QuantizationStrategy.TENSOR_GROUP,
    ):
        scales = scale_search_offset(value, scales, args, global_scale=global_scale)
    return scales, zero_points


def calculate_range(
    quantization_args: QuantizationArgs, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculated the effective quantization range for the given Quantization Args

    :param quantization_args: quantization args to get range of
    :param device: device to store the range to
    :return: tuple endpoints for the given quantization range
    """
    if quantization_args.type == QuantizationType.INT:
        bit_range = 2.0**quantization_args.num_bits
        q_max = torch.tensor(bit_range / 2 - 1, device=device)
        q_min = torch.tensor(-bit_range / 2, device=device)
    elif quantization_args.type == QuantizationType.FLOAT:
        if quantization_args.num_bits == 8:
            q_max = torch.tensor(FP8_E4M3_DATA.max, device=device)
            q_min = torch.tensor(FP8_E4M3_DATA.min, device=device)
        elif quantization_args.num_bits == 4:
            q_max = torch.tensor(FP4_E2M1_DATA.max, device=device)
            q_min = torch.tensor(FP4_E2M1_DATA.min, device=device)
        else:
            raise NotImplementedError(
                "Range calculation only supported for 4 and 8 bits"
            )
    else:
        raise ValueError(f"Invalid quantization type {quantization_args.type}")

    return q_min, q_max


def is_module_quantized(module: Module) -> bool:
    """
    Check if a module is quantized, based on the existence of a non-empty quantization
    scheme

    :param module: pytorch module to check
    :return: True if module is quantized, False otherwise
    """
    if not hasattr(module, "quantization_scheme"):
        return False

    if module.quantization_scheme.weights is not None:
        return True

    if module.quantization_scheme.input_activations is not None:
        return True

    if module.quantization_scheme.output_activations is not None:
        return True

    return False


def is_model_quantized(model: Module) -> bool:
    """
    Check if any modules in a model are quantized, based on the existence of a non-empty
    quantization scheme in at least one module

    :param model: pytorch model
    :return: True if model is quantized, False otherwise
    """
    return any(is_module_quantized(submodule) for submodule in model.modules())


def module_type(module: Module) -> str:
    """
    Gets a string representation of a module type

    :module: pytorch module to get type of
    :return: module type as a string
    """
    return type(module).__name__


def get_torch_bit_depth(value: torch.Tensor) -> int:
    """
    Determine the number of bits used to represent the dtype of a tensor

    :param value: tensor to check bit depth of
    :return: bit depth of each element in the value tensor
    """
    try:
        bit_depth = torch.finfo(value.dtype).bits
    except TypeError:
        bit_depth = torch.iinfo(value.dtype).bits

    return bit_depth


def can_quantize(value: torch.Tensor, quant_args: "QuantizationArgs") -> bool:  # noqa
    """
    Checks if value can be quantized by quant_args.

    :param value: tensor to check for quantization
    :param quant_args: QuantizationArgs to use for quantization
    :return: False if value is already quantized to quant_args or value is incompatible
    with quant_args, True if value can be quantized with quant_args
    """
    bit_depth = get_torch_bit_depth(value)
    requested_depth = quant_args.num_bits
    if bit_depth < quant_args.num_bits:
        _LOGGER.warn(
            f"Can't quantize tensor with bit depth {bit_depth} to {requested_depth}."
            "The QuantizationArgs provided are not compatible with the input tensor."
        )

    return bit_depth > quant_args.num_bits


def generate_gparam(
    updated_min_val: torch.Tensor,
    updated_max_val: torch.Tensor,
    scale_data: FloatArgs | None = FP8_E4M3_DATA,
    quant_data: FloatArgs | None = FP4_E2M1_DATA,
    dtype: torch.dtype | None = torch.float32,
):
    """
    Generate a global scale for an entire tensor (input_tensor).
    Goal of the scale is to ensure that the quantization (local) scale
    falls into the approproiate dtype range.

    E.g. for NVFP4, group (local) scales are in dtype FP8. The global_scale
    attempts to use the entire FP8 dtype range while mapping a per-group max
    to the FP4 max (quant_data=FP4_E2M1_DATA, max=6.0).

    For NVFP8, pass quant_data=FP8_E4M3_DATA (max=448) so the formula
    becomes scale_data.max * FP8_max / max_val_pos.
    """
    min_vals = torch.min(updated_min_val, torch.zeros_like(updated_min_val))
    max_vals = torch.max(updated_max_val, torch.zeros_like(updated_max_val))
    max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
    max_val_pos = torch.clamp(max_val_pos, min=torch.finfo(max_val_pos.dtype).tiny)
    global_scale = scale_data.max * quant_data.max / max_val_pos

    # Replace any NaN or Inf with 1.0. NaN arises when max_val_pos was NaN
    # (clamp does not propagate NaN, so it passes through to the division).
    # Inf arises when max_val_pos was near zero and the division overflows float32.
    # global_scale=1 means no global scaling, which is a safe fallback for
    # uncalibrated experts.
    global_scale = torch.nan_to_num(global_scale, nan=1.0, posinf=1.0, neginf=1.0)

    return global_scale.to(dtype).reshape([1])


def strategy_cdiv(
    value: int,
    divisor: int,
    strategy: QuantizationStrategy | None,
    strict: bool = False,
) -> int:
    dividend = math.ceil(value / divisor)
    if dividend * divisor != value:
        message = (
            f"{strategy} quantization strategy requires strict division of "
            f"weight/activation size {value} and group/block size {divisor}. "
            "consider reducing the group/block size or ignoring modules with "
            f"weights not divisible by {divisor}"
        )
        if strict:
            raise ValueError(message)

        else:
            logger.bind(log_once=True).warning(message)

    return dividend


def _get_dtype_eps(dtype: torch.dtype) -> float:
    if dtype == FP8_E4M3_DATA.dtype:
        return 0.125
    elif dtype == FP4_E2M1_DATA.dtype:
        return 0.25
    elif torch.is_floating_point(torch.tensor([], dtype=dtype)):
        return torch.finfo(dtype).eps
    else:
        return 1


def calculate_block_padding(
    shape: tuple[int, ...],
    block_structure: tuple[int, int],
) -> tuple[int, int]:
    """
    Calculate the padding needed to make tensor dimensions divisible by block size.

    For block quantization, dimensions must be divisible by the block size for
    proper scale alignment when layers are merged in inference frameworks like vLLM.

    :param shape: shape of the tensor (at least 2D)
    :param block_structure: [block_height, block_width] for block quantization
    :return: tuple of (pad_rows, pad_cols) needed to make dimensions divisible
    """
    if len(shape) < 2:
        raise ValueError(f"Tensor must be at least 2D, got shape {shape}")

    rows, cols = shape[-2], shape[-1]
    block_height, block_width = block_structure

    pad_rows = (block_height - rows % block_height) % block_height
    pad_cols = (block_width - cols % block_width) % block_width

    return pad_rows, pad_cols


def maybe_pad_tensor_for_block_quant(
    tensor: torch.Tensor,
    block_structure: tuple[int, int],
) -> torch.Tensor:
    """
    Pad a tensor so its dimensions are divisible by the block size.

    This is essential for FP8 block quantization when dimensions are not
    divisible by block size. The padding ensures that when weights are
    merged in inference frameworks (like vLLM's gate_up_proj), the scale
    tensor blocks align correctly.

    :param tensor: tensor to pad (at least 2D)
    :param block_structure: [block_height, block_width] for block quantization
    :return: padded tensor
    """
    original_shape = tensor.shape

    pad_rows, pad_cols = calculate_block_padding(original_shape, block_structure)

    if pad_rows == 0 and pad_cols == 0:
        return tensor

    # F.pad uses (left, right, top, bottom) for last two dimensions
    padded_tensor = torch.nn.functional.pad(
        tensor, (0, pad_cols, 0, pad_rows), mode="constant", value=0
    )

    return padded_tensor
