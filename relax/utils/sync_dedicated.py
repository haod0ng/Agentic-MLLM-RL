# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validation helpers for the synchronous disaggregated execution mode."""

from argparse import Namespace


SYNC_DEDICATED_RESOURCES = {
    "actor": (1, 4),
    "rollout": (1, 12),
    "judge_accuracy": (1, 4),
    "judge_multiturn_vlm": (1, 4),
}


def validate_sync_dedicated_args(args: Namespace) -> None:
    """Validate the single supported synchronous disaggregated topology."""
    if not getattr(args, "is_sync_dedicated", False):
        return
    if args.weight_version_validation_timeout_s <= 0:
        raise ValueError("--weight-version-validation-timeout-s must be > 0.")
    incompatible_modes = [name for name in ("fully_async", "colocate", "hybrid") if bool(getattr(args, name, False))]
    if incompatible_modes:
        raise ValueError(
            "--is-sync-dedicated cannot be combined with "
            + ", ".join(f"--{name.replace('_', '-')}" for name in incompatible_modes)
            + "."
        )
    if args.max_staleness != 0:
        raise ValueError("--is-sync-dedicated requires --max-staleness 0.")
    if args.offload_train or args.offload_rollout:
        raise ValueError("--is-sync-dedicated requires --no-offload-train and --no-offload-rollout.")
    resources = args.resource or {}
    if set(resources) != set(SYNC_DEDICATED_RESOURCES):
        raise ValueError(
            "--is-sync-dedicated supports exactly actor, rollout, judge_accuracy, and "
            f"judge_multiturn_vlm resources; configured roles: {sorted(resources)}."
        )
    for role, expected in SYNC_DEDICATED_RESOURCES.items():
        try:
            actual = tuple(int(value) for value in resources[role])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"--is-sync-dedicated has invalid resource entry for {role}: {resources[role]!r}"
            ) from exc
        if actual != expected:
            raise ValueError(f"--is-sync-dedicated requires resource {role}={list(expected)}, got {list(actual)}.")

    expected_topology = {
        "num_gpus_per_node": 4,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 4,
        "rollout_num_gpus": 12,
        "rollout_num_gpus_per_engine": 1,
    }
    mismatches = {
        name: getattr(args, name, None)
        for name, expected in expected_topology.items()
        if getattr(args, name, None) != expected
    }
    if mismatches:
        raise ValueError(
            f"--is-sync-dedicated requires fixed topology {expected_topology}; mismatched values: {mismatches}."
        )
