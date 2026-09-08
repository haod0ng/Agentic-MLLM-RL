# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Post-publication validation for synchronous disaggregated rollout
engines."""

import time
from argparse import Namespace
from typing import Any

import ray
import torch.distributed as dist


def validate_rollout_engine_weight_versions(
    *,
    args: Namespace,
    rollout_manager: Any,
    expected_version: str,
    process_group: Any,
) -> None:
    """Check all engines on rank 0 and broadcast one success/failure result."""
    validation_error = [None]
    if dist.get_rank(group=process_group) == 0:
        timeout_s = float(args.weight_version_validation_timeout_s)
        deadline = time.monotonic() + timeout_s
        try:
            rollout_state = ray.get(
                rollout_manager.get_rollout_engines_and_lock.remote(),
                timeout=timeout_s,
            )
            rollout_engines = rollout_state[0]
            if not rollout_engines:
                raise RuntimeError("rollout manager returned no engines")
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise TimeoutError("rollout-manager lookup consumed the validation timeout")
            expected_engine_count = args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            if len(rollout_engines) != expected_engine_count:
                raise RuntimeError(f"expected {expected_engine_count} rollout engines, got {len(rollout_engines)}")
            http_timeout_s = min(remaining_s, float(args.rollout_http_timeout))
            engine_versions = ray.get(
                [engine.get_weight_version.remote(timeout=http_timeout_s) for engine in rollout_engines],
                timeout=remaining_s,
            )
            mismatches = [
                (engine_index, str(version))
                for engine_index, version in enumerate(engine_versions)
                if str(version) != expected_version
            ]
            if mismatches:
                raise RuntimeError(f"expected rollout weight version {expected_version}, mismatches={mismatches}")
        except Exception as exc:
            validation_error[0] = f"{type(exc).__name__}: {exc}"

    dist.broadcast_object_list(validation_error, src=0, group=process_group)
    if validation_error[0] is not None:
        raise RuntimeError(f"Rollout weight-version validation failed: {validation_error[0]}")
