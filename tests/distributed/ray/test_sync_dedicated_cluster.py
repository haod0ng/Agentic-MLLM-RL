# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest


MODULE_PATH = Path("scripts/entrypoint/validate_sync_dedicated_cluster.py")
SPEC = importlib.util.spec_from_file_location("validate_sync_dedicated_cluster", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
cluster_validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cluster_validator)


def _nodes(role_counts: dict[str, int], *, gpus: float = 4.0):
    nodes = []
    for role, count in role_counts.items():
        for index in range(count):
            nodes.append(
                {
                    "Alive": True,
                    "NodeID": f"{role}-{index}",
                    "NodeManagerAddress": f"10.0.0.{len(nodes) + 1}",
                    "Labels": {"relax_role": role},
                    "Resources": {"GPU": gpus},
                }
            )
    return nodes


def _probe_assignments():
    assignments = {}
    node_index = 0
    for role, gpu_count in cluster_validator.PROBE_GPU_COUNTS.items():
        node_count = cluster_validator.EXPECTED_ROLE_COUNTS[role]
        role_assignments = []
        for role_gpu_index in range(gpu_count):
            role_assignments.append(
                {
                    "node_id": f"node-{node_index + role_gpu_index // 4}",
                    "gpu_ids": [str(role_gpu_index % 4)],
                }
            )
        assignments[role] = role_assignments
        node_index += node_count
    return assignments


def test_sync_dedicated_cluster_accepts_exact_six_node_role_layout():
    report = cluster_validator.validate_sync_dedicated_nodes(_nodes(cluster_validator.EXPECTED_ROLE_COUNTS))

    assert report["roles"] == cluster_validator.EXPECTED_ROLE_COUNTS
    assert len(report["nodes"]) == 6


def test_sync_dedicated_cluster_rejects_wrong_role_distribution():
    with pytest.raises(RuntimeError, match="role distribution mismatch"):
        cluster_validator.validate_sync_dedicated_nodes(
            _nodes(
                {
                    "actor": 1,
                    "rollout": 2,
                    "judge_accuracy": 2,
                    "judge_multiturn_vlm": 1,
                }
            )
        )


def test_sync_dedicated_cluster_rejects_wrong_gpu_count():
    with pytest.raises(RuntimeError, match="exactly 4 GPUs"):
        cluster_validator.validate_sync_dedicated_nodes(_nodes(cluster_validator.EXPECTED_ROLE_COUNTS, gpus=3.0))


def test_sync_dedicated_cluster_rejects_ray_without_label_selector():
    def _legacy_placement_group(bundles, strategy="PACK"):
        return None

    with (
        patch.object(cluster_validator, "placement_group", _legacy_placement_group),
        pytest.raises(RuntimeError, match="bundle_label_selector"),
    ):
        cluster_validator.validate_sync_dedicated_nodes(_nodes(cluster_validator.EXPECTED_ROLE_COUNTS))


def test_sync_dedicated_probe_accepts_disjoint_full_24_gpu_layout():
    report = cluster_validator.validate_probe_assignments(_probe_assignments())

    assert report["gpu_count"] == 24
    assert report["roles"]["rollout"]["gpu_count"] == 12
    assert len(report["roles"]["rollout"]["nodes"]) == 3


def test_sync_dedicated_probe_rejects_cross_role_node_overlap():
    assignments = _probe_assignments()
    assignments["judge_accuracy"][0]["node_id"] = assignments["actor"][0]["node_id"]

    with pytest.raises(RuntimeError, match="reused physical assignment|expected 1 nodes x 4 GPUs|share nodes"):
        cluster_validator.validate_probe_assignments(assignments)


def test_sync_dedicated_probe_rejects_duplicate_physical_gpu():
    assignments = _probe_assignments()
    assignments["rollout"][1] = dict(assignments["rollout"][0])

    with pytest.raises(RuntimeError, match="reused physical assignment"):
        cluster_validator.validate_probe_assignments(assignments)
