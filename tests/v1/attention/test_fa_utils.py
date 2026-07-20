# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace

import pytest

from vllm import config
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends import fa_utils
from vllm.v1.attention.backends import flash_attn as flash_attn_backend
from vllm.v1.kv_cache_interface import KVQuantMode


def _resolve(
    monkeypatch,
    *,
    major: int,
    supported: set[int],
    override: int | None = None,
    batch_invariant: bool = False,
    **kwargs,
) -> int | None:
    interface = types.ModuleType("vllm.vllm_flash_attn.flash_attn_interface")
    interface.__dict__.update(
        is_fa_version_supported=lambda version: version in supported,
        fa_version_unsupported_reason=lambda version: "unsupported",
    )
    monkeypatch.setitem(
        sys.modules, "vllm.vllm_flash_attn.flash_attn_interface", interface
    )
    platform = SimpleNamespace(
        is_xpu=lambda: False,
        is_rocm=lambda: False,
        get_device_capability=lambda: SimpleNamespace(major=major),
        is_device_capability_family=lambda family: family == major * 10,
    )
    monkeypatch.setattr(fa_utils, "current_platform", platform)
    monkeypatch.setattr(fa_utils.envs, "VLLM_BATCH_INVARIANT", batch_invariant)
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(flash_attn_version=override),
        model_config=SimpleNamespace(is_diffusion=False),
    )
    monkeypatch.setattr(config, "get_current_vllm_config_or_none", lambda: vllm_config)
    return fa_utils.get_flash_attn_version(**kwargs)


@pytest.mark.parametrize(
    ("major", "supported", "override", "batch_invariant", "kwargs", "expected"),
    [
        (9, {2, 3, 4}, None, False, {}, 4),
        (9, {2, 3}, None, False, {}, 3),
        (9, {2}, None, False, {}, 2),
        (9, {2, 3, 4}, 3, False, {}, 3),
        (9, {2, 3, 4}, None, False, {"head_size": 40, "is_paged": True}, 3),
        (9, {2, 3, 4}, None, False, {"head_size": 40}, 4),
        (9, {2, 3, 4}, None, False, {"head_size": 264, "is_paged": True}, 4),
        (
            9,
            {2, 3, 4},
            None,
            False,
            {"head_size": 40, "head_size_v": 72, "is_paged": True},
            3,
        ),
        (
            9,
            {2, 3, 4},
            None,
            False,
            {"head_size": 128, "head_size_v": 72, "is_paged": True},
            4,
        ),
        (9, {2, 3, 4}, None, True, {}, 3),
        (9, {2, 3, 4}, None, True, {"head_size": 264}, 2),
        (
            9,
            {2, 3, 4},
            None,
            True,
            {"head_size": 40, "head_size_v": 72, "has_sinks": True},
            2,
        ),
        (
            9,
            {2, 3, 4},
            None,
            True,
            {"kv_cache_dtype": "fp8", "head_size": 40},
            2,
        ),
        (9, {2, 3, 4}, None, False, {"requires_alibi": True}, 2),
        (10, {2, 3, 4}, None, False, {}, 4),
    ],
)
def test_flash_attn_version_resolution(
    monkeypatch,
    major,
    supported,
    override,
    batch_invariant,
    kwargs,
    expected,
):
    assert (
        _resolve(
            monkeypatch,
            major=major,
            supported=supported,
            override=override,
            batch_invariant=batch_invariant,
            **kwargs,
        )
        == expected
    )


def test_hopper_fp8_kv_cache_support_uses_request_dimensions(monkeypatch):
    assert (
        _resolve(
            monkeypatch,
            major=9,
            supported={2, 3, 4},
            kv_cache_dtype="fp8",
            head_size=128,
        )
        == 3
    )
    assert fa_utils.flash_attn_supports_kv_cache_dtype("fp8", head_size=128)
    assert not fa_utils.flash_attn_supports_kv_cache_dtype("fp8", head_size=40)


@pytest.mark.parametrize(
    ("version", "cudagraph_support"),
    [
        (3, AttentionCGSupport.ALWAYS),
        (4, AttentionCGSupport.UNIFORM_BATCH),
    ],
)
@pytest.mark.parametrize(
    ("kv_quant_mode", "resolved_dtype"),
    [
        (KVQuantMode.FP8_PER_TENSOR, "fp8"),
        (KVQuantMode.NONE, "auto"),
    ],
)
def test_metadata_builder_uses_request_version(
    monkeypatch, version, cudagraph_support, kv_quant_mode, resolved_dtype
):
    calls = []

    def resolve(**kwargs):
        assert config.get_current_vllm_config_or_none() is vllm_config
        calls.append(kwargs)
        return version

    monkeypatch.setattr(flash_attn_backend, "get_flash_attn_version", resolve)

    def no_dcp_group():
        raise AssertionError

    monkeypatch.setattr("vllm.distributed.parallel_state.get_dcp_group", no_dcp_group)
    kv_cache_spec = SimpleNamespace(
        head_size=40,
        head_size_v=72,
        block_size=16,
        dtype=object(),
        kv_quant_mode=kv_quant_mode,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            get_num_attention_heads=lambda parallel_config: 8,
            get_num_kv_heads=lambda parallel_config: 2,
            get_head_size=lambda: 40,
            rswa_window=None,
        ),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=None,
        ),
        attention_config=SimpleNamespace(flash_attn_max_num_splits_for_cuda_graph=32),
    )

    builder_cls = flash_attn_backend.FlashAttentionMetadataBuilder
    assert (
        builder_cls.get_cudagraph_support(vllm_config, kv_cache_spec)
        == cudagraph_support
    )
    builder = builder_cls(kv_cache_spec, [], vllm_config, device=None)
    assert builder.aot_schedule is (version == 3)
    assert (
        calls
        == [
            {
                "head_size": 40,
                "head_size_v": 72,
                "kv_cache_dtype": resolved_dtype,
                "is_paged": True,
            }
        ]
        * 2
    )
