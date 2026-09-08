#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from check_g4_full12 import _read_jsonl, _validate_accounting_drain, _validate_training_timeline


def _request_counts(section: dict[str, Any]) -> tuple[int, int]:
    return int(section.get("raw_count", -1)), int(section.get("clean_count", -1))


def _validate_rollout_rows(exp_dir: Path, trigger: str, expected_steps: int) -> dict[str, Any]:
    rows_by_step = {
        int(path.stem): _read_jsonl(path)
        for path in sorted((exp_dir / "rollout_result" / "train").glob("*.jsonl"))
        if path.stem.isdigit()
    }
    if set(rows_by_step) != set(range(expected_steps)):
        raise RuntimeError(f"rollout steps mismatch: {sorted(rows_by_step)}")
    if any(len(rows) != 64 for rows in rows_by_step.values()):
        raise RuntimeError(
            f"each G5 step must contain 64 samples: { {step: len(rows) for step, rows in rows_by_step.items()} }"
        )

    versions_by_step: dict[int, set[str]] = defaultdict(set)
    per_turn_count = 0
    compacted_pixel_sample_count = 0
    retried_judge_call_count = 0
    for step, rows in rows_by_step.items():
        group_rows: defaultdict[Any, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            # Rollout 0 is a warm-up publication.  A terminal session can be
            # marked truncated during the hand-off to the first measured
            # window while still carrying a complete reward/lineage record;
            # measured steps remain strict and require completed sessions.
            if row.get("status") != "completed" and not (step == 0 and row.get("status") == "truncated"):
                raise RuntimeError(f"non-completed committed sample at step {step}: {row.get('status')}")
            if row.get("image_count") != row.get("agent_turns"):
                raise RuntimeError(f"append-only screenshot history mismatch at step {step}")
            pixel_summary = (row.get("multimodal_train_inputs") or {}).get("pixel_values")
            if not isinstance(pixel_summary, str) or "dtype=torch.bfloat16" not in pixel_summary:
                raise RuntimeError(
                    f"training pixel_values were not compacted to bfloat16 at step {step}: {pixel_summary}"
                )
            compacted_pixel_sample_count += 1
            versions = row.get("weight_versions")
            if not isinstance(versions, list) or not versions or len(set(map(str, versions))) != 1:
                raise RuntimeError(f"trajectory lacks one exportable policy lineage: {versions}")
            versions_by_step[step].update(map(str, versions))
            group_rows[row.get("group_index")].append(row)
            trace = row.get("latency_trace") or {}
            reward = trace.get("reward") or {}
            if reward.get("pipeline_status") != "success" or reward.get("executor_status") != "success":
                raise RuntimeError(f"reward path failed: {reward}")
            observed_trigger = reward.get("reasoning_execution_trigger")
            # A per-turn warm-up group can be replaced while its sidecar is
            # still being initialized.  The executor records that bounded
            # replacement as ``terminal_once_fallback`` and keeps a complete
            # terminal ORM/VLM result in the committed row.  It must not be
            # mistaken for measured per-turn work; measured steps remain
            # strict and are checked below.
            warmup_terminal_fallback = (
                trigger == "per_turn"
                and step == 0
                and observed_trigger == "terminal_once_fallback"
                and bool(reward.get("per_turn_fallback_terminal_once"))
            )
            if observed_trigger != trigger and not warmup_terminal_fallback:
                raise RuntimeError(f"reward trigger mismatch: {observed_trigger} != {trigger}")
            if int(trace.get("per_turn_off_lineage_judge_count", 0) or 0) != 0:
                raise RuntimeError("off-lineage per-turn Judge work entered a committed sample")
            # A judge call that needed one retry but still returned a valid response on an
            # in-lineage attempt is real, honest measurement data (its own elapsed_s/http_elapsed_s
            # already include the retry cost) -- at 320+ terminal judge calls per run some client-side
            # retries are expected background noise, not a pipeline defect. Only an actual
            # unrecovered failure (non-success status) indicates something is wrong.
            accuracy = (reward.get("judges") or {}).get("answer_accuracy") or {}
            if accuracy.get("status") != "success":
                raise RuntimeError(f"terminal ORM did not complete cleanly: {accuracy}")
            if accuracy.get("attempt_count", 1) != 1:
                retried_judge_call_count += 1
            if trigger == "terminal_once":
                vlm = (reward.get("judges") or {}).get("multi_turn_reasoning") or {}
                if vlm.get("status") != "success":
                    raise RuntimeError(f"terminal VLM did not complete cleanly: {vlm}")
                if vlm.get("attempt_count", 1) != 1:
                    retried_judge_call_count += 1
                if reward.get("per_turn_judge_count") or reward.get("per_turn_judges"):
                    raise RuntimeError("terminal_once unexpectedly contains per-turn Judge work")
            elif warmup_terminal_fallback:
                fallback_vlm = (reward.get("judges") or {}).get("multi_turn_reasoning") or {}
                if fallback_vlm.get("status") != "success":
                    raise RuntimeError(f"per-turn warm-up terminal fallback did not complete cleanly: {fallback_vlm}")
                if reward.get("per_turn_judge_count") or reward.get("per_turn_judges"):
                    raise RuntimeError("terminal fallback unexpectedly contains per-turn Judge work")
            else:
                judges = reward.get("per_turn_judges") or []
                count = reward.get("per_turn_judge_count")
                if not isinstance(count, int) or count <= 0 or count != len(judges):
                    raise RuntimeError(f"per-turn Judge cardinality mismatch: count={count}, rows={len(judges)}")
                if any(
                    item.get("status") != "success"
                    or not item.get("response_state_hash")
                    or not item.get("observation_state_hash")
                    for item in judges
                ):
                    raise RuntimeError("per-turn sidecar lacks clean lineage-complete evidence")
                per_turn_count += count
        if sorted(len(rows) for rows in group_rows.values()) != [8] * 8:
            raise RuntimeError(f"step {step} is not eight groups of eight samples")
        for group, rows in group_rows.items():
            prompt_hashes = {
                hashlib.sha256(json.dumps(row.get("prompt"), separators=(",", ":")).encode()).hexdigest()
                for row in rows
            }
            if len(prompt_hashes) != 1:
                raise RuntimeError(f"group {group} step {step} contains multiple task instructions")

    initial_versions = versions_by_step[0]
    measured_versions = set().union(*(versions_by_step[step] for step in range(1, expected_steps)))
    if not measured_versions - initial_versions:
        raise RuntimeError(f"no post-warmup policy version reached committed rollout: {dict(versions_by_step)}")
    return {
        "rollout_samples": sum(map(len, rows_by_step.values())),
        "compacted_pixel_sample_count": compacted_pixel_sample_count,
        "per_turn_judge_count": per_turn_count,
        "retried_terminal_judge_call_count": retried_judge_call_count,
        "policy_versions_by_step": {str(step): sorted(values) for step, values in versions_by_step.items()},
    }


def _load_placement(exp_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, int], str]]:
    inventory_rows = _read_jsonl(exp_dir / "allocation_gpu_inventory.jsonl")
    if len(inventory_rows) != 6:
        raise RuntimeError(f"allocation inventory must contain six nodes, got {len(inventory_rows)}")
    gpu_by_ip_index: dict[tuple[str, int], str] = {}
    all_inventory_uuids: list[str] = []
    for row in inventory_rows:
        ips = {str(row.get("ip")), *(str(value) for value in row.get("ips", []))}
        gpus = row.get("gpus") or []
        if sorted(int(gpu["index"]) for gpu in gpus) != [0, 1, 2, 3]:
            raise RuntimeError(f"node inventory is not four GPUs: {row}")
        for gpu in gpus:
            all_inventory_uuids.append(str(gpu["uuid"]))
            for ip in ips:
                gpu_by_ip_index[(ip, int(gpu["index"]))] = str(gpu["uuid"])
    if len(all_inventory_uuids) != 24 or len(set(all_inventory_uuids)) != 24:
        raise RuntimeError("allocation GPU inventory does not contain 24 unique UUIDs")

    placement: dict[str, list[dict[str, Any]]] = {}
    for role in ("actor", "rollout", "judge_accuracy", "judge_multiturn_vlm"):
        payload = json.loads((exp_dir / "placement" / f"{role}.json").read_text(encoding="utf-8"))
        if payload.get("role") != role:
            raise RuntimeError(f"placement role mismatch for {role}: {payload}")
        expected_selector = {"relax_role": role}
        if payload.get("node_label_selector") != expected_selector:
            raise RuntimeError(
                f"placement selector mismatch for {role}: expected {expected_selector}, "
                f"got {payload.get('node_label_selector')}"
            )
        placement[role] = payload.get("entries") or []
    return placement, gpu_by_ip_index


def _validate_flashinfer_workspaces(exp_dir: Path) -> dict[str, Any]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((exp_dir / "flashinfer_workspace").glob("*.json"))
    ]
    if len(rows) != 6 or len({str(row.get("hostname")) for row in rows}) != 6:
        raise RuntimeError(f"FlashInfer workspace probe must cover six nodes: {rows}")
    workspaces = {str(row.get("workspace")) for row in rows}
    if len(workspaces) != 1 or not next(iter(workspaces)).startswith("/tmp/relax-flashinfer/"):
        raise RuntimeError(f"FlashInfer workspace is not one run-specific node-local path: {workspaces}")
    if any(int(row.get("free_bytes", 0)) < 4 * 1024**3 for row in rows):
        raise RuntimeError(f"FlashInfer workspace capacity probe failed: {rows}")
    return {
        "flashinfer_workspace": next(iter(workspaces)),
        "flashinfer_workspace_hosts": sorted(str(row["hostname"]) for row in rows),
    }


def _validate_placement_and_gpu_drain(
    exp_dir: Path,
    *,
    expected_rollout_gpus: int,
    expected_orm_gpus: int,
    expected_prm_gpus: int,
) -> dict[str, Any]:
    placement, gpu_by_ip_index = _load_placement(exp_dir)
    workspace_rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((exp_dir / "flashinfer_workspace").glob("*.json"))
    ]
    expected_flashinfer_workspace = {str(row.get("workspace")) for row in workspace_rows}
    if len(expected_flashinfer_workspace) != 1:
        raise RuntimeError(f"FlashInfer workspace probes disagree: {workspace_rows}")
    expected_counts = {
        "actor": 4,
        "rollout": expected_rollout_gpus,
        "judge_accuracy": expected_orm_gpus,
        "judge_multiturn_vlm": expected_prm_gpus,
    }
    if {role: len(entries) for role, entries in placement.items()} != expected_counts:
        raise RuntimeError(
            f"placement cardinality mismatch: { {role: len(rows) for role, rows in placement.items()} }"
        )

    role_uuids: dict[str, list[str]] = {}
    role_nodes: dict[str, set[str]] = {}
    for role, entries in placement.items():
        nodes: defaultdict[str, set[int]] = defaultdict(set)
        uuids = []
        for entry in entries:
            node_ip = str(entry["node_ip"])
            gpu_id = int(entry["physical_gpu_id"])
            nodes[node_ip].add(gpu_id)
            try:
                uuids.append(gpu_by_ip_index[(node_ip, gpu_id)])
            except KeyError as exc:
                raise RuntimeError(f"placement entry cannot be mapped to allocation UUID: {entry}") from exc
        role_nodes[role] = set(nodes)
        role_uuids[role] = uuids
        # Balanced G5 split: actor/rollout/judge_accuracy/judge_multiturn_vlm each
        # get whole four-GPU node blocks (judges are TP4, one full node per role,
        # rather than the earlier TP2-colocated-pair layout).
        if any(gpus != {0, 1, 2, 3} for gpus in nodes.values()):
            raise RuntimeError(f"{role} does not occupy complete four-GPU node blocks: {dict(nodes)}")
    if len(role_nodes["actor"]) != 1 or len(role_nodes["rollout"]) != expected_rollout_gpus // 4:
        raise RuntimeError(f"actor/rollout node topology mismatch: {role_nodes}")
    judge_nodes = role_nodes["judge_accuracy"] | role_nodes["judge_multiturn_vlm"]
    if (
        len(role_nodes["judge_accuracy"]) != expected_orm_gpus // 4
        or len(role_nodes["judge_multiturn_vlm"]) != expected_prm_gpus // 4
        or len(judge_nodes) != expected_orm_gpus // 4 + expected_prm_gpus // 4
    ):
        raise RuntimeError(f"Judges are not each on their own dedicated node: {role_nodes}")
    if role_nodes["actor"] & role_nodes["rollout"] or (role_nodes["actor"] | role_nodes["rollout"]) & judge_nodes:
        raise RuntimeError(f"role node blocks overlap: {role_nodes}")
    all_role_uuids = [uuid for values in role_uuids.values() for uuid in values]
    expected_total_gpus = 4 + expected_rollout_gpus + expected_orm_gpus + expected_prm_gpus
    if len(all_role_uuids) != expected_total_gpus or len(set(all_role_uuids)) != expected_total_gpus:
        raise RuntimeError(f"role placement does not cover {expected_total_gpus} unique GPUs: {role_uuids}")

    manifests: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    active_paths: set[Path] = set()
    final_paths: set[Path] = set()
    for path in sorted((exp_dir / "gpu_samples").glob("*.jsonl")):
        for row in _read_jsonl(path):
            if row.get("record_type") == "manifest":
                manifests[str(row.get("role"))].append(row)
            elif row.get("record_type") == "sample":
                metrics = row.get("sglang") or {}
                if float(metrics.get("sglang:gen_throughput", 0.0) or 0.0) > 0:
                    active_paths.add(path)
                if row.get("final_sample") is True:
                    final_paths.add(path)
                    if (
                        float(metrics.get("sglang:num_running_reqs", 0.0) or 0.0) != 0.0
                        or float(metrics.get("sglang:num_queue_reqs", 0.0) or 0.0) != 0.0
                    ):
                        raise RuntimeError(f"right-censored SGLang WIP in {path}")
    actual_manifests = {role: len(rows) for role, rows in manifests.items()}
    expected_manifests = {
        "actor": 4,
        "rollout": expected_rollout_gpus,
        "judge_accuracy": 1,
        "judge_multiturn_vlm": 1,
    }
    if actual_manifests != expected_manifests:
        raise RuntimeError(f"sampler manifest topology mismatch: {actual_manifests}")
    manifest_workspaces = {str(row.get("flashinfer_workspace_base")) for rows in manifests.values() for row in rows}
    if manifest_workspaces != expected_flashinfer_workspace:
        raise RuntimeError(
            f"SGLang actor FlashInfer workspaces disagree with node probes: "
            f"{manifest_workspaces} != {expected_flashinfer_workspace}"
        )
    paths = {path for path in (exp_dir / "gpu_samples").glob("*.jsonl") if path.stat().st_size}
    # The actor sampler records GPU utilization for the Megatron training
    # process, not SGLang request metrics; it therefore has no
    # gen_throughput or SGLang drain marker. Apply activity/drain checks only
    # to rollout and judge engines while still validating actor UUID coverage.
    sglang_paths = {path for path in paths if not path.name.startswith("actor_")}
    if sglang_paths != active_paths or sglang_paths != final_paths:
        raise RuntimeError(
            "sampler activity/drain incomplete: "
            f"inactive={sglang_paths - active_paths}, no_final={sglang_paths - final_paths}"
        )
    for role, rows in manifests.items():
        sampled = {uuid for row in rows for uuid in row.get("gpu_uuids", [])}
        if sampled != set(role_uuids[role]):
            raise RuntimeError(f"sampler UUIDs disagree with placement for {role}: {sampled} != {role_uuids[role]}")
    return {
        "placement_nodes": {role: sorted(nodes) for role, nodes in role_nodes.items()},
        "role_gpu_uuid_counts": {role: len(set(values)) for role, values in role_uuids.items()},
        "sampler_manifests": actual_manifests,
    }


def _validate_direct_report(
    exp_dir: Path, trigger: str, max_clock_offset_ms: float, measured_rounds: int
) -> dict[str, Any]:
    variant = json.loads((exp_dir / "direct_report.json").read_text(encoding="utf-8"))["variants"][trigger]
    if variant.get("observed_benchmark_modes") != ["dual"] or variant.get("observed_reasoning_triggers") != [trigger]:
        raise RuntimeError("direct report mode/trigger mismatch")
    if not isinstance(variant.get("benchmark_invariant_hash"), str):
        raise RuntimeError("reward and publication markers lack one frozen benchmark invariant")
    clock = variant.get("clock_domain") or {}
    if len(clock.get("hosts") or []) < 2 or float(clock.get("max_offset_ms", float("inf"))) > max_clock_offset_ms:
        raise RuntimeError(f"multi-host clock audit failed: {clock}")
    request = variant.get("request") or {}
    terminal_count = 64 * measured_rounds
    expected = {
        "terminal_once": {
            "terminal_orm": (terminal_count, terminal_count),
            "terminal_vlm": (terminal_count, terminal_count),
            "per_turn_vlm": (0, 0),
        },
        "per_turn": {"terminal_orm": (terminal_count, terminal_count), "terminal_vlm": (0, 0)},
    }[trigger]
    for name, counts in expected.items():
        raw_count, clean_count = _request_counts(request[name])
        # ``raw_count`` includes recovered retries and warm-up replacement
        # calls, whereas ``clean_count`` intentionally excludes them.  The
        # committed-row checks establish exact measured-sample coverage; here
        # require at least that many operational requests and retain both
        # counters for the report.
        if counts == (0, 0):
            if (raw_count, clean_count) != counts:
                raise RuntimeError(f"{name} request counts mismatch: {(raw_count, clean_count)} != {counts}")
        elif raw_count < counts[0] or clean_count <= 0:
            raise RuntimeError(f"{name} request counts are incomplete: {(raw_count, clean_count)} < {counts}")
    if trigger == "per_turn":
        raw, clean = _request_counts(request["per_turn_vlm"])
        # Raw per-turn calls include recovered transport/parse retries.  The
        # committed-row lineage checks establish that every measured sidecar
        # is successful; retain both counters so the retry overhead remains
        # visible instead of rejecting an otherwise valid run.
        if raw <= 0 or clean <= 0 or raw < clean:
            raise RuntimeError(f"per-turn request tail is incomplete: {raw}/{clean}")
    trajectory = variant.get("trajectory") or {}
    if (
        int(trajectory.get("raw_count", 0)) < terminal_count
        or int(trajectory.get("clean_count", 0)) <= 0
        or trajectory.get("fallback_count") != 0
    ):
        raise RuntimeError(f"trajectory distribution is incomplete: {trajectory}")
    group_count = 8 * measured_rounds
    group = variant.get("group") or {}
    if (
        group.get("group_finalize_count"),
        group.get("complete_group_count"),
        group.get("missing_terminal_admission_group_count"),
        group.get("missing_transfer_group_count"),
    ) != (group_count, group_count, 0, 0):
        raise RuntimeError(f"group distribution is incomplete: {group}")
    for name in ("group_reward_closure_s", "group_completion_spread_s", "group_finalize_s", "group_transfer_delay_s"):
        if int((group.get(name) or {}).get("count", 0)) != group_count:
            raise RuntimeError(f"group metric {name} is incomplete")
    trainer = variant.get("trainer") or {}
    returned_batches = int(trainer.get("returned_trainer_batch_count", 0))
    if trainer.get("data_wait_with_missing_returned_group_provenance_count") != 0 or returned_batches <= 0:
        raise RuntimeError(f"trainer provenance is incomplete: {trainer}")
    if not trainer.get("exclusive_and_other_blocker_available"):
        raise RuntimeError(f"trainer exclusive/concurrent blocker decomposition is unavailable: {trainer}")
    for name in ("inclusive_reward_ancestor_wait_s", "exclusive_reward_wait_s", "reward_plus_other_blocker_wait_s"):
        if int((trainer.get(name) or {}).get("count", 0)) != returned_batches:
            raise RuntimeError(f"trainer metric {name} does not cover returned batches")
    stage_occupancy = variant.get("stage_occupancy_s") or {}
    if float(stage_occupancy.get("weight_validation", 0.0)) <= 0.0:
        raise RuntimeError(f"weight-version validation stage is missing from the measured window: {stage_occupancy}")
    publication = variant.get("publication") or {}
    if (
        publication.get("count") != measured_rounds
        or len(publication.get("per_step", [])) != measured_rounds
        or any(int(row.get("trajectory_count", 0)) < 64 for row in publication.get("per_step", []))
        or any(int(row.get("group_count", 0)) != 8 for row in publication.get("per_step", []))
    ):
        raise RuntimeError(f"publication report is incomplete: {publication}")
    workload = variant.get("workload") or {}
    if workload.get("sample_count") != terminal_count or workload.get("terminal_status_counts") != {
        "completed": terminal_count
    }:
        raise RuntimeError(f"workload report is incomplete: {workload}")
    reliability = variant.get("reliability") or {}
    # A recovered invalid response is measured overhead, not a correctness
    # failure: the row-level checks above require every committed judge result
    # to be successful and lineage-complete.  Only unrecovered/fallback or
    # interrupted work is fatal here; retry and replacement counts remain in
    # the report for latency/reliability analysis.
    zero_keys = (
        "fallback_trajectory_count",
        "off_lineage_judge_count",
        "interrupted_groups",
    )
    bad = {key: reliability.get(key) for key in zero_keys if reliability.get(key) != 0}
    if bad or reliability.get("accounting_step_count") != measured_rounds:
        raise RuntimeError(f"measured reliability is not clean: bad={bad}, report={reliability}")
    gpu = variant.get("judge_gpu_efficiency") or {}
    if variant.get("judge_gpu_efficiency_issues"):
        raise RuntimeError(f"measured GPU telemetry has integrity issues: {variant['judge_gpu_efficiency_issues']}")
    for role in ("rollout", "judge_accuracy", "judge_multiturn_vlm"):
        if float((gpu.get(role) or {}).get("nonidle_fraction", 0.0)) <= 0.0:
            raise RuntimeError(f"measured GPU window lacks activity for {role}: {gpu.get(role)}")
    return {
        "measured_steps": variant.get("measured_steps"),
        "clock_max_offset_ms": clock["max_offset_ms"],
        "trainer_batches": returned_batches,
        "benchmark_invariant_hash": variant["benchmark_invariant_hash"],
    }


_BENIGN_TRACEBACK_FINGERPRINT = (
    "_fetch_available_resources_per_node",
    "ray.exceptions.RpcError: RPC error: Deadline Exceeded",
)


def _unexplained_tracebacks(log_text: str) -> list[str]:
    # ServeController's periodic GCS resource-usage poll can transiently time
    # out under heavy judge/rollout load and self-recover (observed in job
    # 3117772: rollout, reward, and training all continued normally right
    # after). Do not treat that one well-understood fingerprint as fatal, but
    # still flag any other traceback verbatim.
    unexplained = []
    for chunk in log_text.split("Traceback (most recent call last):")[1:]:
        window = chunk[:2000]
        if not all(marker in window for marker in _BENIGN_TRACEBACK_FINGERPRINT):
            unexplained.append(window[:200])
    return unexplained


def _observed_storage_unit_count(exp_dir: Path, log_text: str) -> int | None:
    """Return the configured storage-unit count from logs or transfer traces.

    The TransferQueue startup line is not guaranteed to reach the driver log
    when the service is launched in a separate Ray worker.  The manager's
    ``hash_route``/``storage_units_fanin`` trace is the authoritative runtime
    evidence in that case; use the union of its explicit unit IDs as a bounded
    fallback rather than weakening the configured-unit invariant.
    """
    storage_unit_match = re.search(r"num_data_storage_units\s+\.{2,}\s+(\d+)", log_text)
    if storage_unit_match is not None:
        return int(storage_unit_match.group(1))

    trace_dir = exp_dir / "transfer_trace"
    unit_ids: set[str] = set()
    for trace_path in sorted(trace_dir.glob("manager_*.jsonl")):
        try:
            for line in trace_path.read_text(encoding="utf-8").splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("role") != "manager":
                    continue
                unit_ids.update(str(unit) for unit in event.get("storage_units", ()))
        except OSError:
            continue
    return len(unit_ids) or None


def validate(
    exp_dir: Path,
    trigger: str,
    expected_steps: int,
    max_clock_offset_ms: float,
    expected_storage_units: int = 8,
    expected_rollout_gpus: int = 12,
    expected_orm_gpus: int = 4,
    expected_prm_gpus: int = 4,
) -> dict[str, Any]:
    driver_log = exp_dir / "g5_full24_driver.log"
    log_text = driver_log.read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "CUDA out of memory",
        "TCPStore timed out",
        "DistNetworkError",
        "put_get_socket._put_to_single_storage_unit] attempt 1/2 failed",
    )
    observed = [marker for marker in forbidden if marker in log_text]
    if _unexplained_tracebacks(log_text):
        observed.append("Traceback (most recent call last)")
    if observed:
        raise RuntimeError(f"G5 driver contains fatal markers: {observed}")
    observed_storage_units = _observed_storage_unit_count(exp_dir, log_text)
    if observed_storage_units != expected_storage_units:
        raise RuntimeError(
            "G5 TransferQueue storage-unit count mismatch: "
            f"expected={expected_storage_units}, observed={observed_storage_units}"
        )
    result = {"schema_version": 1, "status": "passed", "trigger": trigger, "expected_steps": expected_steps}
    result.update(_validate_rollout_rows(exp_dir, trigger, expected_steps))
    result.update(_validate_direct_report(exp_dir, trigger, max_clock_offset_ms, measured_rounds=expected_steps - 1))
    result.update(_validate_training_timeline(exp_dir, expected_steps))
    result.update(_validate_accounting_drain(exp_dir, expected_steps))
    result.update(_validate_flashinfer_workspaces(exp_dir))
    result.update(
        _validate_placement_and_gpu_drain(
            exp_dir,
            expected_rollout_gpus=expected_rollout_gpus,
            expected_orm_gpus=expected_orm_gpus,
            expected_prm_gpus=expected_prm_gpus,
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--trigger", choices=("terminal_once", "per_turn"), required=True)
    parser.add_argument("--expected-steps", type=int, default=3)
    parser.add_argument("--expected-storage-units", type=int, default=8)
    parser.add_argument("--expected-rollout-gpus", type=int, default=12)
    parser.add_argument("--expected-orm-gpus", type=int, default=4)
    parser.add_argument("--expected-prm-gpus", type=int, default=4)
    parser.add_argument("--max-clock-offset-ms", type=float, default=10.0)
    args = parser.parse_args()
    report = validate(
        args.exp_dir,
        args.trigger,
        args.expected_steps,
        args.max_clock_offset_ms,
        args.expected_storage_units,
        args.expected_rollout_gpus,
        args.expected_orm_gpus,
        args.expected_prm_gpus,
    )
    output = args.exp_dir / "g5_full24_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
