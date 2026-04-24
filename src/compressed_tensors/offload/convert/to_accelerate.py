# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from typing import Optional

import torch
from compressed_tensors.offload.cache import DiskCache, OffloadCache
from compressed_tensors.offload.convert.helpers import (
    DEFAULT_OFFLOAD_DEVICE,
    get_tensors,
    norm_device,
)
from compressed_tensors.offload.module import remove_module_offload
from compressed_tensors.utils import patch_attr
from loguru import logger


__all__ = ["to_accelerate", "to_accelerate_module"]


def to_accelerate(model: torch.nn.Module) -> dict[str, str]:
    """
    Convert a model from `compressed_tensors` offloading to `accelerate` offloading.
    This is is often called before `PreTrainedModel.save_pretrained`, as without this
    conversion, `save_pretrained` will use excessive memory and device movement.

    :param model: model dispatched with `compressed_tensors` offloading
    :return: accelerate-style device map

    Note: if NO module in the model actually has an `OffloadCache` (i.e. the
    model was loaded via HF `accelerate` / `device_map="auto"` rather than
    via `compressed_tensors` offloading), this function leaves the existing
    `hf_device_map` untouched. The previous behaviour — stamping every
    named submodule with `str(DEFAULT_OFFLOAD_DEVICE)` ("cpu") — tricked
    `transformers.PreTrainedModel.save_pretrained` into the offloaded-save
    path, which then triggered a VLM key-remap bug (module_map uses
    pre-remap keys while state_dict uses post-remap keys → KeyError during
    shard save, e.g. "visual.patch_embed.proj.weight" on Qwen2.5-VL).
    """
    # Fast path: model is not compressed_tensors-offloaded at all. Just return
    # the existing hf_device_map unchanged.
    has_any_offload_cache = any(
        isinstance(getattr(m, "_parameters", None), OffloadCache)
        for m in model.modules()
    )
    if not has_any_offload_cache:
        return dict(getattr(model, "hf_device_map", {}) or {})

    hf_device_map = {}
    hf_disk_index = _to_accelerate_disk_index(model, DiskCache.index)

    for name, module in model.named_modules():
        # Only modules that actually had an OffloadCache carry meaningful
        # device information; for everything else (container modules without
        # their own parameters), skip the entry rather than writing the
        # DEFAULT_OFFLOAD_DEVICE ("cpu") sentinel. That sentinel would
        # otherwise propagate into hf_device_map and trip the offloaded-save
        # path in transformers.PreTrainedModel.save_pretrained even when all
        # real parameters are on GPU.
        had_offload_cache = isinstance(
            getattr(module, "_parameters", None), OffloadCache
        )
        offload_device_str = to_accelerate_module(module, name, hf_disk_index)
        if had_offload_cache:
            hf_device_map[name] = offload_device_str

    setattr(model, "hf_device_map", hf_device_map)
    return hf_device_map


def to_accelerate_module(
    module: torch.nn.Module,
    name: Optional[str] = None,
    hf_disk_index: Optional[dict[str, dict[str, str]]] = None,
) -> str:
    """
    Convert a module from `compressed_tensors` offloading to `accelerate` offloading

    :param module: module to convert to accelerate offloading
    :param name: name of module in model
    :param hf_disk_index: accelerate-style disk index to attach to weight loaders
    :return: str of offloaded device. Defaults to cpu if module does not have parameters
    """
    has_accelerate = True
    try:
        from accelerate.hooks import AlignDevicesHook, add_hook_to_module
        from accelerate.utils import OffloadedWeightsLoader, PrefixedDataset
    except ImportError:
        logger.warning(
            "Cannot convert module without `accelerate` installed. This may result "
            "in high memory usage during saving and tied tensors being saved twice",
            log_once=True,
        )
        has_accelerate = False

    offload_device = DEFAULT_OFFLOAD_DEVICE
    if has_accelerate and isinstance(module._parameters, OffloadCache):
        cache = module._parameters
        offload_device = norm_device(cache.offload_device)
        remove_module_offload(module, onload_tensors=False)

        # create weights map
        if isinstance(cache, DiskCache):
            if name is None or hf_disk_index is None:
                raise ValueError(
                    "Must provide `name` and `hf_disk_index` "
                    "to convert disk offloaded module"
                )

            weights_map = PrefixedDataset(
                prefix=f"{name}.",
                dataset=OffloadedWeightsLoader(
                    index=hf_disk_index,
                    save_folder=cache.offload_dir,
                ),
            )
        else:
            weights_map = dict(get_tensors(module, recurse=False))

        # create hook
        hook = AlignDevicesHook(
            execution_device=cache.onload_device,
            offload=True,
            io_same_device=True,
            weights_map=weights_map,
            offload_buffers=True,
            place_submodules=False,
        )

        # add hook (skip onloading)
        with patch_attr(AlignDevicesHook, "init_hook", lambda self, module: module):
            add_hook_to_module(module, hook)

        # skipping init hook => need to populate `original_devices`
        hook.original_devices = {
            name: offload_device if offload_device != "disk" else torch.device("cpu")
            for name, _ in get_tensors(module, recurse=False)
        }

    return str(offload_device)


def _to_accelerate_disk_index(
    model: torch.nn.Module, index: dict[torch.Tensor, dict[str, str]]
) -> dict[str, dict[str, str]]:
    from compressed_tensors.offload import disable_onloading  # circular dependency

    with disable_onloading():
        offloaded_to_key = _invert_dict(model.state_dict(keep_vars=True))

    return {
        key: weight_info
        for offloaded, weight_info in index.items()
        for key in offloaded_to_key[offloaded]
    }


def _invert_dict(d: dict) -> dict:
    inverted = defaultdict(list)
    for key, value in d.items():
        inverted[value].append(key)
    return inverted
