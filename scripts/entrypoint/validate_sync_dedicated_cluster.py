# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Validate the fixed six-node Ray topology before model deployment."""

import argparse
import inspect
import json
from collections import Counter
from typing import Any

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


EXPECTED_ROLE_COUNTS = {
    "actor": 1,
    "rollout": 3,
    "judge_accuracy": 1,
    "judge_multiturn_vlm": 1,
}

PROBE_GPU_COUNTS = {
    "actor": 4,
    "rollout": 12,
    "judge_accuracy": 4,
    "judge_multiturn_vlm": 4,
}


@ray.remote(num_cpus=0, num_gpus=1)
def _probe_gpu_assignment() -> dict[str, Any]:
    """Report the node and visible GPU assigned to one placement bundle."""
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "gpu_ids": [str(gpu_id) for gpu_id in ray.get_gpu_ids()],
    }


def validate_sync_dedicated_nodes(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a compact topology summary or raise on any placement
    ambiguity."""
    if "bundle_label_selector" not in inspect.signature(placement_group).parameters:
        raise RuntimeError(
            "--is-sync-dedicated requires Ray placement_group(bundle_label_selector=...), "
            f"but Ray {getattr(ray, '__version__', 'unknown')} does not expose it"
        )

    alive_nodes = [node for node in nodes if bool(node.get("Alive", node.get("alive", False)))]
    expected_nodes = sum(EXPECTED_ROLE_COUNTS.values())
    if len(alive_nodes) != expected_nodes:
        raise RuntimeError(f"expected {expected_nodes} alive Ray nodes, got {len(alive_nodes)}")

    roles: Counter[str] = Counter()
    summary_nodes = []
    for node in alive_nodes:
        labels = node.get("Labels", node.get("labels", {})) or {}
        resources = node.get("Resources", node.get("resources", {})) or {}
        role = labels.get("relax_role")
        gpu_count = float(resources.get("GPU", 0))
        if role not in EXPECTED_ROLE_COUNTS:
            raise RuntimeError(f"Ray node has missing or unknown relax_role label: {role!r}")
        if gpu_count != 4.0:
            raise RuntimeError(f"Ray node role={role!r} must expose exactly 4 GPUs, got {gpu_count}")
        roles[role] += 1
        summary_nodes.append(
            {
                "node_id": node.get("NodeID", node.get("node_id")),
                "node_ip": node.get("NodeManagerAddress", node.get("node_ip")),
                "role": role,
                "gpus": gpu_count,
            }
        )

    if dict(roles) != EXPECTED_ROLE_COUNTS:
        raise RuntimeError(f"sync-dedicated Ray role distribution mismatch: {dict(roles)}")
    return {"ray_version": getattr(ray, "__version__", "unknown"), "roles": dict(roles), "nodes": summary_nodes}


def validate_probe_assignments(assignments: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Validate that all 24 probe actors occupy the intended disjoint nodes."""
    expected_roles = set(PROBE_GPU_COUNTS)
    if set(assignments) != expected_roles:
        raise RuntimeError(f"sync-dedicated probe roles mismatch: got {sorted(assignments)}")

    role_nodes: dict[str, set[str]] = {}
    physical_assignments: set[tuple[str, str]] = set()
    summary: dict[str, Any] = {}
    for role, expected_gpus in PROBE_GPU_COUNTS.items():
        probes = assignments[role]
        if len(probes) != expected_gpus:
            raise RuntimeError(f"sync-dedicated probe role={role!r} expected {expected_gpus} GPUs, got {len(probes)}")
        nodes: Counter[str] = Counter()
        for probe in probes:
            gpu_ids = probe.get("gpu_ids", [])
            if len(gpu_ids) != 1:
                raise RuntimeError(f"sync-dedicated probe role={role!r} must see exactly one GPU, got {gpu_ids}")
            node_id = str(probe.get("node_id", ""))
            if not node_id:
                raise RuntimeError(f"sync-dedicated probe role={role!r} returned an empty node id")
            assignment = (node_id, str(gpu_ids[0]))
            if assignment in physical_assignments:
                raise RuntimeError(f"sync-dedicated probe reused physical assignment {assignment}")
            physical_assignments.add(assignment)
            nodes[node_id] += 1

        expected_node_count = EXPECTED_ROLE_COUNTS[role]
        if len(nodes) != expected_node_count or any(gpu_count != 4 for gpu_count in nodes.values()):
            raise RuntimeError(
                f"sync-dedicated probe role={role!r} expected {expected_node_count} nodes x 4 GPUs, got {dict(nodes)}"
            )
        role_nodes[role] = set(nodes)
        summary[role] = {"gpu_count": len(probes), "nodes": dict(nodes)}

    roles = list(PROBE_GPU_COUNTS)
    for index, role in enumerate(roles):
        for other_role in roles[index + 1 :]:
            overlap = role_nodes[role] & role_nodes[other_role]
            if overlap:
                raise RuntimeError(f"sync-dedicated probe roles {role!r} and {other_role!r} share nodes: {overlap}")
    return {"gpu_count": len(physical_assignments), "roles": summary}


def probe_sync_dedicated_placement(timeout_s: float) -> dict[str, Any]:
    """Allocate all production-sized role PGs together, then verify each
    bundle."""
    if timeout_s <= 0:
        raise ValueError(f"probe timeout must be positive, got {timeout_s}")

    placement_groups: dict[str, Any] = {}
    probe_refs: list[Any] = []
    probe_succeeded = False
    try:
        # Submit every PG before awaiting readiness. This verifies that the full
        # 24-GPU topology is jointly feasible, rather than checking each role
        # against the same free GPUs one at a time.
        for role, gpu_count in PROBE_GPU_COUNTS.items():
            bundles = [{"GPU": 1, "CPU": 1} for _ in range(gpu_count)]
            strategy = "PACK" if role == "rollout" else "STRICT_PACK"
            placement_groups[role] = placement_group(
                bundles,
                strategy=strategy,
                bundle_label_selector=[{"relax_role": role} for _ in bundles],
            )
        ray.get([pg.ready() for pg in placement_groups.values()], timeout=timeout_s)

        refs: dict[str, list[Any]] = {}
        for role, gpu_count in PROBE_GPU_COUNTS.items():
            role_refs = [
                _probe_gpu_assignment.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=placement_groups[role],
                        placement_group_bundle_index=bundle_index,
                    )
                ).remote()
                for bundle_index in range(gpu_count)
            ]
            probe_refs.extend(role_refs)
            refs[role] = role_refs

        all_results = ray.get(probe_refs, timeout=timeout_s)
        assignments: dict[str, list[dict[str, Any]]] = {}
        result_offset = 0
        for role, gpu_count in PROBE_GPU_COUNTS.items():
            assignments[role] = all_results[result_offset : result_offset + gpu_count]
            result_offset += gpu_count
        report = validate_probe_assignments(assignments)
        probe_succeeded = True
        return report
    finally:
        for ref in probe_refs:
            try:
                ray.cancel(ref, force=True)
            except Exception:
                pass
        cleanup_errors = []
        for pg in placement_groups.values():
            try:
                remove_placement_group(pg)
            except Exception as exc:
                cleanup_errors.append(f"{type(exc).__name__}: {exc}")
        if probe_succeeded and cleanup_errors:
            raise RuntimeError(f"sync-dedicated probe placement-group cleanup failed: {cleanup_errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="auto")
    parser.add_argument("--probe-timeout-s", type=float, default=120.0)
    args = parser.parse_args()
    ray.init(address=args.address, ignore_reinit_error=True)
    topology = validate_sync_dedicated_nodes(ray.nodes())
    topology["placement_probe"] = probe_sync_dedicated_placement(args.probe_timeout_s)
    print(json.dumps(topology, sort_keys=True))


if __name__ == "__main__":
    main()
