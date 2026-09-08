# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from relax.backends.megatron.weight_update.validation import validate_rollout_engine_weight_versions


def _build_actor(*, rank: int = 0, versions: list[str] | None = None):
    args = SimpleNamespace(
        is_sync_dedicated=True,
        weight_version_validation_timeout_s=10.0,
        rollout_http_timeout=4.0,
        rollout_num_gpus=2,
        rollout_num_gpus_per_engine=1,
    )
    engines = [MagicMock(name=f"engine-{index}") for index in range(2)]
    for engine in engines:
        engine.get_weight_version.remote.return_value = MagicMock()
    rollout_manager = MagicMock()
    rollout_manager.get_rollout_engines_and_lock.remote.return_value = MagicMock()
    rollout_state = (engines, MagicMock(), 0, [], [])
    return args, rollout_manager, rank, versions or ["7", "7"], rollout_state, engines


def _validate(args, rollout_manager):
    validate_rollout_engine_weight_versions(
        args=args,
        rollout_manager=rollout_manager,
        expected_version="7",
        process_group="gloo",
    )


def test_sync_weight_validation_queries_all_engines_only_on_rank_zero():
    args, rollout_manager, rank, versions, rollout_state, engines = _build_actor()
    with (
        patch("relax.backends.megatron.weight_update.validation.dist.get_rank", return_value=rank),
        patch("relax.backends.megatron.weight_update.validation.dist.broadcast_object_list") as mock_broadcast,
        patch(
            "relax.backends.megatron.weight_update.validation.ray.get", side_effect=[rollout_state, versions]
        ) as mock_get,
        patch("relax.backends.megatron.weight_update.validation.time.monotonic", side_effect=[10.0, 11.0]),
    ):
        _validate(args, rollout_manager)

    assert mock_get.call_count == 2
    assert all(engine.get_weight_version.remote.call_count == 1 for engine in engines)
    for engine in engines:
        engine.get_weight_version.remote.assert_called_once_with(timeout=4.0)
    mock_broadcast.assert_called_once_with([None], src=0, group="gloo")


def test_sync_weight_validation_nonzero_rank_only_participates_in_broadcast():
    args, rollout_manager, _, _, _, engines = _build_actor(rank=1)
    with (
        patch("relax.backends.megatron.weight_update.validation.dist.get_rank", return_value=1),
        patch("relax.backends.megatron.weight_update.validation.dist.broadcast_object_list") as mock_broadcast,
        patch("relax.backends.megatron.weight_update.validation.ray.get") as mock_get,
    ):
        _validate(args, rollout_manager)

    mock_get.assert_not_called()
    assert all(engine.get_weight_version.remote.call_count == 0 for engine in engines)
    mock_broadcast.assert_called_once_with([None], src=0, group="gloo")


def test_sync_weight_validation_broadcasts_mismatch_before_raising():
    args, rollout_manager, _, _, rollout_state, _ = _build_actor(versions=["7", "6"])
    broadcast_payloads = []

    def _capture(payload, **_kwargs):
        broadcast_payloads.append(list(payload))

    with (
        patch("relax.backends.megatron.weight_update.validation.dist.get_rank", return_value=0),
        patch("relax.backends.megatron.weight_update.validation.dist.broadcast_object_list", side_effect=_capture),
        patch(
            "relax.backends.megatron.weight_update.validation.ray.get",
            side_effect=[rollout_state, ["7", "6"]],
        ),
        patch("relax.backends.megatron.weight_update.validation.time.monotonic", side_effect=[10.0, 11.0]),
        pytest.raises(RuntimeError, match="mismatches"),
    ):
        _validate(args, rollout_manager)

    assert len(broadcast_payloads) == 1
    assert "mismatches" in broadcast_payloads[0][0]


def test_sync_weight_validation_broadcasts_ray_timeout_before_raising():
    args, rollout_manager, _, _, _, _ = _build_actor()
    broadcast_payloads = []

    def _capture(payload, **_kwargs):
        broadcast_payloads.append(list(payload))

    with (
        patch("relax.backends.megatron.weight_update.validation.dist.get_rank", return_value=0),
        patch("relax.backends.megatron.weight_update.validation.dist.broadcast_object_list", side_effect=_capture),
        patch(
            "relax.backends.megatron.weight_update.validation.ray.get",
            side_effect=TimeoutError("manager timeout"),
        ),
        patch("relax.backends.megatron.weight_update.validation.time.monotonic", return_value=10.0),
        pytest.raises(RuntimeError, match="manager timeout"),
    ):
        _validate(args, rollout_manager)

    assert len(broadcast_payloads) == 1
    assert "TimeoutError" in broadcast_payloads[0][0]


def test_sglang_weight_version_request_has_http_timeout():
    pytest.importorskip("sglang_router")
    pytest.importorskip("sglang.srt.server_args")
    from relax.backends.sglang.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine.args = SimpleNamespace(rollout_http_timeout=120.0)
    response = MagicMock()
    response.json.return_value = {"weight_version": "7"}

    with patch("relax.backends.sglang.sglang_engine.requests.get", return_value=response) as mock_get:
        assert engine.get_weight_version(timeout=4.0) == "7"

    mock_get.assert_called_once_with("http://127.0.0.1:30000/model_info", timeout=4.0)


def test_sglang_weight_version_falls_back_for_legacy_endpoint():
    pytest.importorskip("sglang_router")
    pytest.importorskip("sglang.srt.server_args")
    from relax.backends.sglang.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.node_rank = 0
    engine.server_host = "127.0.0.1"
    engine.server_port = 30000
    engine.args = SimpleNamespace(rollout_http_timeout=120.0)
    not_found = MagicMock(status_code=404)
    legacy = MagicMock(status_code=200)
    legacy.json.return_value = {"weight_version": "7"}

    with patch("relax.backends.sglang.sglang_engine.requests.get", side_effect=[not_found, legacy]) as mock_get:
        assert engine.get_weight_version(timeout=4.0) == "7"

    assert mock_get.call_args_list == [
        (("http://127.0.0.1:30000/model_info",), {"timeout": 4.0}),
        (("http://127.0.0.1:30000/weight_version",), {"timeout": 4.0}),
    ]
