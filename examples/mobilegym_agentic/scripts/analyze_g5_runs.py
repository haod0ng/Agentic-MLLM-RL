#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Produce reproducible post-run reports for the two G5 dual-judge modes.

The reports deliberately separate measured spans from inferred quantities.  All
wall-clock joins use the run's audited timestamps; process-local monotonic
transfer spans are never subtracted across processes.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_ANALYZER_DIR = Path(__file__).resolve().parents[2] / "agentic_dual_judge"
sys.path.insert(0, str(_ANALYZER_DIR))
import analyze_latency  # noqa: E402


def _union(intervals: Iterable[tuple[float, float]]) -> float:
    return analyze_latency._union_duration(intervals)


def _stats(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not finite:
        return {"count": 0, "mean_s": None, "p50_s": None, "p90_s": None, "max_s": None}
    return {
        "count": len(finite),
        "mean_s": statistics.mean(finite),
        "p50_s": finite[(len(finite) - 1) // 2],
        "p90_s": finite[min(len(finite) - 1, math.floor(0.9 * (len(finite) - 1)))],
        "max_s": finite[-1],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _load_run(
    exp: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    tuple[float, float],
    list[dict[str, Any]],
]:
    marker = exp / "latency_markers" / "weight_serving_ready.jsonl"
    events, source = analyze_latency.load_variant_events(
        exp / "timeline", include_rollout_with_timeline=False, ready_marker_path=marker
    )
    if source != "timeline+ready_markers":
        raise ValueError(f"{exp}: unexpected timeline source {source!r}")
    report = json.loads((exp / "direct_report.json").read_text(encoding="utf-8"))
    variants = report.get("variants", {})
    if len(variants) != 1:
        raise ValueError(f"{exp}: expected one direct-report variant")
    window_raw = next(iter(variants.values())).get("window_s")
    if not isinstance(window_raw, list) or len(window_raw) != 2:
        raise ValueError(f"{exp}: direct report has no fixed measurement window")
    rows = []
    for path in sorted((exp / "rollout_result" / "train").glob("*.jsonl")):
        rows.extend(_read_jsonl(path))
    request_events, request_source = analyze_latency.load_variant_events(
        exp / "rollout_result", include_rollout_with_timeline=False
    )
    if request_source != "rollout_jsonl":
        raise ValueError(f"{exp}: unexpected request-event source {request_source!r}")
    request_events = [event for event in request_events if event["name"] == "critical_path.judge_request"]
    return events, rows, (float(window_raw[0]), float(window_raw[1])), request_events


def _window_events(events: list[dict[str, Any]], window: tuple[float, float]) -> list[dict[str, Any]]:
    start, end = window
    return [event for event in events if event["end_s"] >= start and event["start_s"] <= end]


def _stage_overlap(events: list[dict[str, Any]], window: tuple[float, float]) -> dict[str, float]:
    start, end = window
    return {
        stage: _union(
            (max(start, event["start_s"]), min(end, event["end_s"])) for event in events if event["stage"] == stage
        )
        for stage in analyze_latency.STAGE_ORDER
    }


def _latency_overlap(exp: Path, events: list[dict[str, Any]], window: tuple[float, float]) -> dict[str, Any]:
    duration = window[1] - window[0]
    occupancy = _stage_overlap(events, window)
    overlap = analyze_latency._active_set_durations(events, window)
    report = json.loads((exp / "direct_report.json").read_text(encoding="utf-8"))
    variant = next(iter(report["variants"].values()))
    return {
        "schema_version": 1,
        "measurement_window_s": list(window),
        "window_duration_s": duration,
        "inclusive_stage_occupancy_s": occupancy,
        "inclusive_stage_occupancy_pct": {key: 100.0 * value / duration for key, value in occupancy.items()},
        "overlap_partition_pct": {key: 100.0 * value / duration for key, value in sorted(overlap.items())},
        "reward_gpu_efficiency": variant.get("judge_gpu_efficiency"),
        "caveat": (
            "Inclusive stage percentages can exceed 100 because concurrent spans overlap. "
            "The overlap partition sums to 100 but is coincidence-based, not proof of the binding bottleneck."
        ),
    }


def _transfer_breakdown(exp: Path) -> dict[str, Any]:
    trace_dir = exp / "transfer_trace"
    manager = [row for path in trace_dir.glob("manager_*.jsonl") for row in _read_jsonl(path)]
    producer = [row for path in trace_dir.glob("producer_*.jsonl") for row in _read_jsonl(path)]
    storage = [row for path in trace_dir.glob("storage_unit_*.jsonl") for row in _read_jsonl(path)]
    partitions = sorted(
        {str(row.get("partition_id")) for row in producer if str(row.get("partition_id", "")).startswith("train_")},
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    fanin_by_partition: dict[str, dict[str, Any]] = {}
    fanin_starts = {
        partition: min(
            row["started_monotonic_s"]
            for row in producer
            if row.get("partition_id") == partition and row.get("event") == "storage_manager_put"
        )
        for partition in partitions
    }

    def nearest_partition(timestamp: Any) -> str | None:
        if not isinstance(timestamp, (int, float)) or not fanin_starts:
            return None
        return min(fanin_starts, key=lambda part: abs(float(timestamp) - fanin_starts[part]))

    # Manager and producer traces are emitted by the same process, so their
    # monotonic clocks are comparable.  Manager events do not carry a
    # partition_id; associate each event with the closest producer PUT wave.
    manager_by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manager:
        partition = nearest_partition(row.get("started_monotonic_s"))
        if partition is not None:
            manager_by_partition[partition].append(row)

    # Storage-unit clocks are process-local and cannot be joined to the
    # producer clock by timestamp.  Each storage unit receives one PUT per
    # partition in order; use the per-unit PUT ordinal to associate it with
    # the producer waves.  The report records this as an ordinal join rather
    # than pretending it is a cross-host timestamp measurement.
    storage_by_partition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for storage_unit in sorted({row.get("storage_unit_id") for row in storage if row.get("storage_unit_id")}):
        operation_rows = {
            "put": sorted(
                (
                    row
                    for row in storage
                    if row.get("storage_unit_id") == storage_unit
                    and row.get("event") == "server_receive_deserialize"
                    and row.get("operation") == "ZMQRequestType.PUT_DATA"
                ),
                key=lambda row: float(row.get("started_monotonic_s", 0.0)),
            ),
            "get": sorted(
                (
                    row
                    for row in storage
                    if row.get("storage_unit_id") == storage_unit
                    and row.get("event") == "server_receive_deserialize"
                    and row.get("operation") == "ZMQRequestType.GET_DATA"
                ),
                key=lambda row: float(row.get("started_monotonic_s", 0.0)),
            ),
            "clear": sorted(
                (
                    row
                    for row in storage
                    if row.get("storage_unit_id") == storage_unit
                    and row.get("event") == "server_receive_deserialize"
                    and row.get("operation") == "ZMQRequestType.CLEAR_DATA"
                ),
                key=lambda row: float(row.get("started_monotonic_s", 0.0)),
            ),
            "store": sorted(
                (
                    row
                    for row in storage
                    if row.get("storage_unit_id") == storage_unit and row.get("event") == "server_store"
                ),
                key=lambda row: float(row.get("started_monotonic_s", 0.0)),
            ),
        }
        for rows_for_operation in operation_rows.values():
            for index, row in enumerate(rows_for_operation):
                if index < len(partitions):
                    storage_by_partition[partitions[index]].append(row)

    for partition in partitions:
        manager_rows = manager_by_partition.get(partition, [])
        producer_rows = [row for row in producer if row.get("partition_id") == partition]
        storage_rows = storage_by_partition.get(partition, [])
        by_event = defaultdict(list)
        for row in manager_rows:
            by_event[row.get("event")].append(row)
        fanin = next((row for row in by_event["storage_units_fanin"]), None)
        roundtrips = by_event["zmq_roundtrip"]
        serializes = by_event["serialize"]
        put_deser = [
            row
            for row in storage_rows
            if row.get("event") == "server_receive_deserialize" and row.get("operation") == "ZMQRequestType.PUT_DATA"
        ]
        put_store = [row for row in storage_rows if row.get("event") == "server_store"]
        get_deser = [
            row
            for row in storage_rows
            if row.get("event") == "server_receive_deserialize" and row.get("operation") == "ZMQRequestType.GET_DATA"
        ]
        clear_deser = [
            row
            for row in storage_rows
            if row.get("event") == "server_receive_deserialize" and row.get("operation") == "ZMQRequestType.CLEAR_DATA"
        ]

        def elapsed(row: dict[str, Any]) -> float:
            return max(0.0, float(row["ended_monotonic_s"]) - float(row["started_monotonic_s"]))

        max_rtt = max((elapsed(row) for row in roundtrips), default=0.0)
        fanin_s = elapsed(fanin) if fanin else None
        last_rtt_end = max((float(row["ended_monotonic_s"]) for row in roundtrips), default=0.0)
        residual = max(0.0, float(fanin["ended_monotonic_s"]) - last_rtt_end) if fanin else None
        async_put = next((row for row in producer_rows if row.get("event") == "async_put_total"), None)
        fanin_by_partition[partition] = {
            "sample_count": next(
                (row.get("sample_count") for row in producer_rows if row.get("event") == "async_put_total"), None
            ),
            "wire_bytes": sum(int(row.get("wire_bytes", 0) or 0) for row in serializes),
            "producer_async_put_s": elapsed(async_put) if async_put else None,
            "manager_fanin_s": fanin_s,
            "manager_serialize_sum_s": sum(elapsed(row) for row in serializes),
            "manager_max_zmq_roundtrip_s": max_rtt,
            "manager_fanin_tail_after_last_ack_s": residual,
            "storage_put_deserialize_sum_s": sum(elapsed(row) for row in put_deser),
            "storage_put_store_sum_s": sum(elapsed(row) for row in put_store),
            "storage_get_deserialize_sum_s": sum(elapsed(row) for row in get_deser),
            "storage_clear_deserialize_sum_s": sum(elapsed(row) for row in clear_deser),
            "storage_put_count": len(put_deser),
            "storage_get_count": len(get_deser),
            "storage_clear_count": len(clear_deser),
            "attribution": (
                "The long interval is in manager ZMQ roundtrip/fan-in. Serialization and storage-side "
                "deserialize/store are measured separately and are not the source of the 100-second tail."
            ),
        }
    all_fanin = [row["manager_fanin_s"] for row in fanin_by_partition.values() if row["manager_fanin_s"] is not None]
    return {
        "schema_version": 1,
        "partitions": fanin_by_partition,
        "fanin_summary": _stats(all_fanin),
        "storage_unit_count": len({row.get("storage_unit_id") for row in storage if row.get("storage_unit_id")}),
        "cross_process_clock_caveat": (
            "Manager events are joined to producer waves by same-process monotonic proximity; storage-unit events "
            "are joined by per-unit PUT ordinal because cross-host monotonic subtraction is invalid. "
            "Use clock_sync_audit.json for wall-clock joins."
        ),
    }


def _admission_breakdown(exp: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    names = {
        "admission_setup": "critical_path.rollout_admission_setup",
        "admission_wait": "critical_path.rollout_admission_wait",
        "dispatch": "critical_path.rollout_dispatch",
        "rollout_queue": "critical_path.rollout_queue",
        "generation": "critical_path.rollout_generation",
        "post_generation": "critical_path.rollout_post_generation",
    }
    by_step: dict[str, dict[str, Any]] = {}
    for step in sorted({event.get("step") for event in events if isinstance(event.get("step"), int)}):
        step_rows = [event for event in events if event.get("step") == step]
        row: dict[str, Any] = {"step": step}
        for label, name in names.items():
            values = [event["end_s"] - event["start_s"] for event in step_rows if event["name"] == name]
            row[f"{label}_sum_s"] = sum(values)
            row[f"{label}_distribution"] = _stats(values)
            row[f"{label}_union_s"] = _union(
                (event["start_s"], event["end_s"]) for event in step_rows if event["name"] == name
            )
        by_step[str(step)] = row
    ready = _read_jsonl(exp / "latency_markers" / "weight_serving_ready.jsonl")
    admission = [event for event in events if event["name"] == names["admission_setup"]]
    first_request = min((event["start_s"] for event in admission), default=None)
    first_ready = min(
        (float(row["wall_time_s"]) for row in ready if isinstance(row.get("wall_time_s"), (int, float))), default=None
    )
    startup_gap = first_ready - first_request if first_ready is not None and first_request is not None else None
    return {
        "schema_version": 1,
        "per_step": by_step,
        "startup_gap_first_request_to_first_ready_s": startup_gap,
        "known_components": [
            "per-session admission setup",
            "per-session admission gate wait",
            "dispatch and rollout queue",
            "policy generation",
        ],
        "unattributed_or_unknown": [
            "MobileGym process/browser launch before the first request (no per-process launch span)",
            "model-weight load and Ray Serve deployment internals unless present in service logs",
            "host-level scheduler/network delay outside the recorded span boundaries",
        ],
        "interpretation": "Admission sums are per-session work; union/makespan is the appropriate wall-clock quantity for the critical path.",
    }


def _row_turn_intervals(rows: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        trace = row.get("latency_trace") or {}
        reward = trace.get("reward") if isinstance(trace, dict) else None
        sample_key = reward.get("sample_key") if isinstance(reward, dict) else None
        session_id = sample_key[0] if isinstance(sample_key, list) and sample_key else row.get("sample_index")
        for turn in trace.get("turns", []) if isinstance(trace, dict) else []:
            events = turn.get("events") if isinstance(turn, dict) else None
            if not isinstance(events, dict):
                continue
            start, end = events.get("chat_request_arrive_at"), events.get("chat_end_at")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
                intervals[str(session_id)].append((float(start), float(end)))
    return intervals


def _env_cpu_bubble(exp: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    request_intervals = _row_turn_intervals(rows)
    per_session: list[dict[str, Any]] = []
    observed_samples = 0
    unknown_sessions = 0
    hz_default = 100.0
    for path in sorted((exp / "env_cpu").glob("*.jsonl")):
        records = _read_jsonl(path)
        if len(records) < 2:
            unknown_sessions += 1
            continue
        session = str(records[0].get("session_id", path.stem))
        intervals = request_intervals.get(session, [])
        if not intervals:
            unknown_sessions += 1
            continue
        records.sort(key=lambda row: float(row.get("sampled_at", 0.0)))
        idle_wait = 0.0
        observed = 0.0
        cpu_s = 0.0
        browser_cpu_s = 0.0
        idle_samples = 0
        browser_counts: list[int] = []
        hz = float(records[0].get("clock_ticks_per_second") or hz_default)
        for current, nxt in zip(records, records[1:]):
            start = float(current.get("sampled_at", 0.0))
            end = float(nxt.get("sampled_at", start))
            if end <= start:
                continue
            overlap = _union((max(start, left), min(end, right)) for left, right in intervals)
            if overlap <= 0:
                continue
            processes = current.get("processes") or []
            next_processes = {
                int(item.get("pid")): item for item in (nxt.get("processes") or []) if item.get("pid") is not None
            }
            current_root = next((item for item in processes if item.get("role") == "bench_env"), None)
            if not current_root:
                continue
            root_pid = int(current_root.get("pid"))
            root_next = next_processes.get(root_pid, {})
            root_delta = max(
                0,
                int(root_next.get("utime_ticks", current_root.get("utime_ticks", 0)) or 0)
                + int(root_next.get("stime_ticks", current_root.get("stime_ticks", 0)) or 0)
                - int(current_root.get("utime_ticks", 0) or 0)
                - int(current_root.get("stime_ticks", 0) or 0),
            )
            aggregate_delta = 0
            browser_delta = 0
            browser_count = 0
            for process in processes:
                pid = process.get("pid")
                if pid is None:
                    continue
                pid = int(pid)
                next_process = next_processes.get(pid, {})
                delta = max(
                    0,
                    int(next_process.get("utime_ticks", process.get("utime_ticks", 0)) or 0)
                    + int(next_process.get("stime_ticks", process.get("stime_ticks", 0)) or 0)
                    - int(process.get("utime_ticks", 0) or 0)
                    - int(process.get("stime_ticks", 0) or 0),
                )
                aggregate_delta += delta
                role = str(process.get("role", ""))
                command = str(process.get("cmdline", ""))
                if role == "playwright" or "chrom" in command.lower():
                    browser_count += 1
                    browser_delta += delta
            dt_s = end - start
            aggregate_cpu = aggregate_delta / hz
            browser_cpu = browser_delta / hz
            observed += overlap
            cpu_s += aggregate_cpu * overlap / dt_s
            browser_cpu_s += browser_cpu * overlap / dt_s
            browser_counts.append(browser_count)
            state = str(current_root.get("state", ""))[:1]
            if state in {"S", "D", "I"} and root_delta == 0:
                idle_wait += overlap
                idle_samples += 1
            observed_samples += 1
        if observed > 0:
            per_session.append(
                {
                    "session_id": session,
                    "task_id": records[0].get("task_id"),
                    "policy_wait_observed_s": observed,
                    "idle_wait_s": idle_wait,
                    "idle_wait_fraction": idle_wait / observed,
                    "aggregate_process_cpu_s": cpu_s,
                    "browser_cpu_s": browser_cpu_s,
                    "idle_wait_cpu_s": 0.0 if idle_wait else None,
                    "idle_sample_count": idle_samples,
                    "mean_browser_process_count": statistics.mean(browser_counts) if browser_counts else 0.0,
                }
            )
    per_session.sort(key=lambda row: row["idle_wait_s"], reverse=True)
    idle_values = [row["idle_wait_s"] for row in per_session]
    observed_values = [row["policy_wait_observed_s"] for row in per_session]
    return {
        "schema_version": 1,
        "classification": "root bench_env state S/D/I with zero root CPU ticks while a policy request interval was in flight",
        "session_count_with_evidence": len(per_session),
        "session_count_unknown": unknown_sessions,
        "sample_intervals_used": observed_samples,
        "aggregate": {
            "policy_wait_s": _stats(observed_values),
            "idle_wait_s": _stats(idle_values),
            "idle_fraction_of_observed": sum(idle_values) / sum(observed_values) if sum(observed_values) else None,
            "total_process_cpu_s": sum(row["aggregate_process_cpu_s"] for row in per_session),
            "total_browser_cpu_s": sum(row["browser_cpu_s"] for row in per_session),
        },
        "top_idle_sessions": per_session[:30],
        "caveats": [
            "This is an observational interval classification, not proof that a sleeping environment was the throughput bottleneck.",
            "Sampling gaps and missing process descendants are unknown, not idle; per-session and whole-job CPU must not be conflated.",
            "A policy request in flight is used as the proxy for waiting for the rollout action; environment-side work outside that interval is excluded.",
        ],
    }


def _api_log_text(path: Path) -> str:
    if not path.exists():
        return ""
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Ray Jobs API exports may wrap the multiline driver log in a JSON
    # envelope; raw text exports are accepted too.
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, dict) and isinstance(envelope.get("logs"), str):
        raw = envelope["logs"]
    return raw


def _parse_perf_log(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    raw = _api_log_text(path)
    pattern = re.compile(r"perf\s+(\d+):\s+(\{.*\})")
    for line in raw.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            payload = ast.literal_eval(match.group(2))
        except (SyntaxError, ValueError):
            continue
        if isinstance(payload, dict):
            result[match.group(1)] = {
                str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))
            }
    return result


def _repeated_computation(
    exp: Path,
    events: list[dict[str, Any]],
    request_events: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    api_log: Path,
) -> dict[str, Any]:
    measured_steps = sorted(
        {int(row.get("step")) for row in _read_jsonl(exp / "latency_markers" / "weight_serving_ready.jsonl")}
    )[1:]
    request_names = {
        "terminal_orm": "orm",
        "terminal_vlm": "terminal_prm",
        "per_turn_vlm": "per_turn_prm",
    }
    requests = [
        event
        for event in request_events
        if event.get("attributes", {}).get("request_kind") in request_names and event.get("step") in measured_steps
    ]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in requests:
        by_kind[request_names[event["attributes"]["request_kind"]]].append(event)

    def request_role(kind: str, values: list[dict[str, Any]]) -> dict[str, Any]:
        attrs = [event.get("attributes", {}) for event in values]
        retry_events = [item for item in attrs if int(item.get("attempt_count", 1) or 1) > 1]
        return {
            "logical_request_count": len(values),
            "retry_event_count": len(retry_events),
            "extra_attempts": sum(max(0, int(item.get("attempt_count", 1) or 1) - 1) for item in attrs),
            "invalid_response_count": sum(int(item.get("invalid_response_count", 0) or 0) for item in attrs),
            "queue_s": sum(float(item.get("queue_elapsed_s", 0.0) or 0.0) for item in attrs),
            "http_s": sum(float(item.get("http_elapsed_s", 0.0) or 0.0) for item in attrs),
            "payload_prep_s": sum(float(item.get("payload_prep_elapsed_s", 0.0) or 0.0) for item in attrs),
            "parse_s": sum(float(item.get("parse_elapsed_s", 0.0) or 0.0) for item in attrs),
            "backoff_s": sum(float(item.get("backoff_elapsed_s", 0.0) or 0.0) for item in attrs),
            "logical_span_s": sum(event["end_s"] - event["start_s"] for event in values),
            "server_tokenize_s": sum(
                float((item.get("server") or {}).get("tokenize_elapsed_s", 0.0) or 0.0) for item in attrs
            ),
            "server_engine_http_s": sum(
                float((item.get("server") or {}).get("engine_http_elapsed_s", 0.0) or 0.0) for item in attrs
            ),
            "server_media_restore_s": sum(
                float((item.get("server") or {}).get("media_restore_elapsed_s", 0.0) or 0.0) for item in attrs
            ),
            "server_engine_call_s": sum(
                float((item.get("server") or {}).get("engine_call_elapsed_s", 0.0) or 0.0) for item in attrs
            ),
            "server_request_total_s": sum(
                float((item.get("server") or {}).get("request_total_elapsed_s", 0.0) or 0.0) for item in attrs
            ),
            "attribution": "Measured logical request events; non-clean events retain retry/invalid latency instead of being discarded.",
        }

    turn_judge_events = [
        event
        for event in events
        if event["name"] == "critical_path.turn_judge" and event.get("step") in measured_steps
    ]
    turn_judge_by_step = defaultdict(list)
    for event in turn_judge_events:
        turn_judge_by_step[str(event.get("step"))].append(event)

    turns = [int(row.get("agent_turns")) for row in rows if isinstance(row.get("agent_turns"), int)]
    image_counts = [int(row.get("image_count")) for row in rows if isinstance(row.get("image_count"), int)]
    image_tokens = [int(row.get("image_token_count")) for row in rows if isinstance(row.get("image_token_count"), int)]
    perf = _parse_perf_log(api_log)
    validation_events = [
        event
        for event in events
        if event["name"] == "critical_path.weight_version_validation" and event.get("step") in measured_steps
    ]
    validation_union = _union((event["start_s"], event["end_s"]) for event in validation_events)
    return {
        "schema_version": 1,
        "measured_steps": measured_steps,
        "judge_retries_and_repeated_calls": {
            kind: request_role(kind, values) for kind, values in sorted(by_kind.items())
        },
        "policy_visual_history": {
            "measured_trajectory_count": len(rows),
            "agent_turns": _stats(turns),
            "image_count": _stats(image_counts),
            "image_token_count": _stats(image_tokens),
            "inferred_repeated_screenshot_count_sum": sum(
                max(0, turns_count * (turns_count - 1) // 2) for turns_count in turns
            ),
            "inference_note": "Assumes one newly captured screenshot per turn and history replay on each subsequent policy request; validate against per-turn payloads before calling this measured.",
        },
        "serialization_and_media": {
            "policy_media_encode_s": sum(
                float((turn.get("events") or {}).get("media_encode_elapsed_s", 0.0) or 0.0)
                for row in rows
                for turn in (row.get("latency_trace") or {}).get("turns", [])
            ),
            "policy_processor_s": sum(
                float((turn.get("events") or {}).get("processor_elapsed_s", 0.0) or 0.0)
                for row in rows
                for turn in (row.get("latency_trace") or {}).get("turns", [])
            ),
            "transfer_manager_serialize_is_reported_separately": True,
        },
        "weight_version_validation": {
            "rank_event_count": len(validation_events),
            "wall_union_s": validation_union,
            "attribution": "Measured rank-local validation stage; rank event count is not additive GPU wall time.",
        },
        "per_turn_vlm_judge_spans": {
            "request_count": len(turn_judge_events),
            "logical_span_sum_s": sum(event["end_s"] - event["start_s"] for event in turn_judge_events),
            "span_union_s": _union((event["start_s"], event["end_s"]) for event in turn_judge_events),
            "by_step": {
                step: {
                    "request_count": len(step_events),
                    "logical_span_sum_s": sum(event["end_s"] - event["start_s"] for event in step_events),
                    "span_union_s": _union((event["start_s"], event["end_s"]) for event in step_events),
                }
                for step, step_events in sorted(turn_judge_by_step.items())
            },
            "attribution": "turn_judge spans include per-turn VLM client and server work; nested queue/http fields are not available on this event.",
        },
        "trainer_perf": perf,
        "labels": {
            "measured": [
                "judge queue/http/tokenize/media fields",
                "policy media/processor elapsed fields",
                "weight validation timeline",
            ],
            "inferred": ["repeated screenshot count from triangular history assumption"],
            "unknown": ["any hidden framework cache recomputation not represented by these spans"],
        },
    }


def _request_wip_category(
    request_events: list[dict[str, Any]],
    request_kind: str,
    window: tuple[float, float],
    measured_steps: list[int],
) -> dict[str, Any]:
    start, end = window
    selected = [
        event
        for event in request_events
        if event.get("step") in measured_steps
        and event.get("attributes", {}).get("request_kind") == request_kind
        and event.get("end_s", start) >= start
        and event.get("start_s", end) <= end
    ]
    clipped: list[tuple[float, float]] = []
    client_sums = {
        "queue_s": 0.0,
        "payload_prep_s": 0.0,
        "http_s": 0.0,
        "parse_s": 0.0,
        "backoff_s": 0.0,
        "client_unattributed_s": 0.0,
    }
    server_sums = {
        "server_queue_elapsed_s": 0.0,
        "media_restore_elapsed_s": 0.0,
        "tokenize_elapsed_s": 0.0,
        "engine_address_elapsed_s": 0.0,
        "engine_http_elapsed_s": 0.0,
        "engine_backoff_elapsed_s": 0.0,
        "engine_call_elapsed_s": 0.0,
        "request_total_elapsed_s": 0.0,
    }
    retry_count = 0
    extra_attempts = 0
    invalid_count = 0
    clean_count = 0
    by_step: dict[str, int] = defaultdict(int)
    for event in selected:
        event_start = max(start, float(event["start_s"]))
        event_end = min(end, float(event["end_s"]))
        if event_end <= event_start:
            continue
        clipped.append((event_start, event_end))
        attrs = event.get("attributes", {})
        attempt_count = int(attrs.get("attempt_count", 1) or 1)
        if attempt_count > 1:
            retry_count += 1
            extra_attempts += attempt_count - 1
        invalid_count += int(attrs.get("invalid_response_count", 0) or 0)
        clean_count += int(analyze_latency._direct_clean(attrs))
        by_step[str(event.get("step"))] += 1
        known_client = 0.0
        for key in ("queue_s", "payload_prep_s", "http_s", "parse_s", "backoff_s"):
            value = float(attrs.get(f"{key.replace('_s', '')}_elapsed_s", 0.0) or 0.0)
            client_sums[key] += value
            known_client += value
        client_sums["client_unattributed_s"] += max(0.0, event_end - event_start - known_client)
        server = attrs.get("server") or {}
        for key in server_sums:
            server_sums[key] += float(server.get(key, 0.0) or 0.0)

    boundaries: dict[float, int] = defaultdict(int)
    for interval_start, interval_end in clipped:
        boundaries[interval_start] += 1
        boundaries[interval_end] -= 1
    wip_area = 0.0
    max_wip = 0
    wip_time: dict[str, float] = defaultdict(float)
    current = 0
    previous = start
    for timestamp in sorted({start, end, *boundaries}):
        if timestamp > previous:
            duration = timestamp - previous
            wip_area += current * duration
            wip_time[str(current)] += duration
            max_wip = max(max_wip, current)
        current += boundaries.get(timestamp, 0)
        previous = timestamp
    interval_sum = sum(right - left for left, right in clipped)
    client_total = sum(client_sums.values())
    client_share = {key: 100.0 * value / client_total if client_total else 0.0 for key, value in client_sums.items()}
    request_total = server_sums["request_total_elapsed_s"]
    server_share = {
        key: 100.0 * server_sums[key] / request_total if request_total else 0.0
        for key in (
            "server_queue_elapsed_s",
            "media_restore_elapsed_s",
            "tokenize_elapsed_s",
            "engine_address_elapsed_s",
            "engine_call_elapsed_s",
        )
    }
    return {
        "request_kind": request_kind,
        "raw_count": len(selected),
        "clock_host_count": len(
            {
                event.get("attributes", {}).get("clock_host")
                for event in selected
                if event.get("attributes", {}).get("clock_host")
            }
        ),
        "clock_host_request_counts": {
            str(host): sum(1 for event in selected if event.get("attributes", {}).get("clock_host") == host)
            for host in sorted(
                {
                    event.get("attributes", {}).get("clock_host")
                    for event in selected
                    if event.get("attributes", {}).get("clock_host")
                }
            )
        },
        "configured_client_concurrency": 24 if request_kind == "per_turn_vlm" else 8,
        "client_instance_count": None,
        "clean_count": clean_count,
        "retry_event_count": retry_count,
        "extra_attempts": extra_attempts,
        "invalid_response_count": invalid_count,
        "request_count_by_step": dict(sorted(by_step.items(), key=lambda item: int(item[0]))),
        "wip": {
            "logical_span_sum_s": interval_sum,
            "logical_span_union_s": _union(clipped),
            "mean_inflight": wip_area / (end - start) if end > start else 0.0,
            "max_inflight": max_wip,
            "wip_time_by_concurrency_s": dict(sorted(wip_time.items(), key=lambda item: int(item[0]))),
        },
        "client_components_s": client_sums,
        "client_component_share_pct": client_share,
        "server_nested_components_s": server_sums,
        "server_component_relative_to_request_total_pct": server_share,
        "semantics": (
            "Client components are a diagnostic decomposition of each logical request span. HTTP contains nested "
            "server work, so client and server fields must not be added as independent wall-clock stages."
        ),
    }


def _reward_wip_breakdown(
    request_events: list[dict[str, Any]], window: tuple[float, float], measured_steps: list[int], mode: str
) -> dict[str, Any]:
    orm = _request_wip_category(request_events, "terminal_orm", window, measured_steps)
    prm_kind = "terminal_vlm" if mode == "terminal_once" else "per_turn_vlm"
    prm = _request_wip_category(request_events, prm_kind, window, measured_steps)
    return {
        "schema_version": 1,
        "measurement_window_s": list(window),
        "window_duration_s": window[1] - window[0],
        "measured_steps": measured_steps,
        "mode": mode,
        "categories": {"orm": orm, "prm": {"branch": mode, **prm}},
        "subcomponent_semantics": (
            "WIP is calculated from logical judge-request intervals and includes operational retries/invalid calls. "
            "The PRM category is terminal_vlm for terminal_once and per_turn_vlm for per_turn. "
            "Clock-synchronized fixed-window inclusion is used; missing intervals are not imputed. "
            "The observed WIP is aggregated across clock hosts; configured client concurrency is per rollout "
            "client, not a global PRM-server concurrency limit. clock_host is not a client-instance identifier, "
            "so an observed max above the configured value can reflect multiple actor clients sharing one host."
        ),
    }


def _tpdp_replay(
    exp: Path,
    api_log: Path,
    mode: str,
    request_events: list[dict[str, Any]],
    measured_steps: list[int],
) -> dict[str, Any]:
    config_path = Path(__file__).resolve().parents[1] / (
        "judge_services_e2e_g5_qwen3vl8_prm3b_terminal_once.json"
        if mode == "terminal_once"
        else "judge_services_e2e_g5_qwen3vl8_prm3b_per_turn.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prm_config = config.get("multi_turn_reasoning") or {}
    engine_config = prm_config.get("engine_config") or {}
    expected_model = str(prm_config.get("model_path", "")).split("/")[-1]
    expected = {
        "tp_size": engine_config.get("tp_size"),
        "dp_size": engine_config.get("dp_size"),
        "num_gpus_per_engine": prm_config.get("num_gpus_per_engine"),
        "max_concurrency": prm_config.get("max_concurrency"),
        "model_basename": expected_model,
    }

    placement_path = exp / "placement" / "judge_multiturn_vlm.json"
    placement = json.loads(placement_path.read_text(encoding="utf-8")) if placement_path.exists() else {}
    entries = placement.get("entries") if isinstance(placement, dict) else []
    physical_gpu_ids = [str(item.get("physical_gpu_id")) for item in entries if isinstance(item, dict)]
    placement_checks = {
        "role": placement.get("role") == "judge_multiturn_vlm",
        "strategy": placement.get("strategy") == "STRICT_PACK",
        "entry_count": len(entries) == 4,
        "unique_physical_gpu_ids": len(set(physical_gpu_ids)) == 4,
        "physical_gpu_ids": physical_gpu_ids,
        "same_node": len({str(item.get("node_ip")) for item in entries if isinstance(item, dict)}) == 1,
    }

    manifests = [
        row
        for path in sorted((exp / "gpu_samples").glob("judge_multiturn_vlm_*.jsonl"))
        for row in _read_jsonl(path)
        if row.get("record_type") == "manifest"
    ]
    manifest = manifests[0] if manifests else {}
    manifest_uuids = [str(value) for value in manifest.get("gpu_uuids", [])]
    manifest_checks = {
        "manifest_count": len(manifests),
        "role": manifest.get("role") == "judge_multiturn_vlm",
        "num_gpus_per_engine": manifest.get("num_gpus_per_engine") == expected["num_gpus_per_engine"],
        "unique_gpu_uuids": len(manifest_uuids) == 4 and len(set(manifest_uuids)) == 4,
        "model_basename": str(manifest.get("model_path", "")).split("/")[-1] == expected_model,
        "gpu_uuids": manifest_uuids,
    }

    log_text = _api_log_text(api_log)
    launch_pairs = [
        {"dp_rank": int(rank), "gpu_start": int(gpu)}
        for rank, gpu in re.findall(r"Launch DP(\d+) starting at GPU #(\d+)", log_text)
    ]
    engine_lines = [
        line
        for line in log_text.splitlines()
        if "server_args=ServerArgs" in line and "tp_size=" in line and "dp_size=" in line and expected_model in line
    ]
    engine_tp = [int(value) for line in engine_lines for value in re.findall(r"(?<![A-Za-z0-9_])tp_size=(\d+)", line)]
    engine_dp = [int(value) for line in engine_lines for value in re.findall(r"(?<![A-Za-z0-9_])dp_size=(\d+)", line)]
    launch_checks = {
        "engine_args_line_count": len(engine_lines),
        "engine_tp_values": engine_tp,
        "engine_dp_values": engine_dp,
        "dp_launches": launch_pairs,
        "tp_dp_in_log": expected["tp_size"] in engine_tp and expected["dp_size"] in engine_dp,
        "dp_launch_count": len(launch_pairs) == expected["dp_size"],
        "dp_ranks": sorted(item["dp_rank"] for item in launch_pairs),
        "dp_gpu_starts": sorted(item["gpu_start"] for item in launch_pairs),
        "dp_gpu_partitioning": sorted(item["gpu_start"] for item in launch_pairs)
        == [index * int(expected["tp_size"]) for index in range(int(expected["dp_size"]))],
    }

    replay_kind = "terminal_vlm" if mode == "terminal_once" else "per_turn_vlm"
    replay_events = [
        event
        for event in request_events
        if event.get("step") in measured_steps and event.get("attributes", {}).get("request_kind") == replay_kind
    ]
    status_counts: dict[str, int] = defaultdict(int)
    for event in replay_events:
        status_counts[str(event.get("attributes", {}).get("request_status"))] += 1
    replay = {
        "request_kind": replay_kind,
        "measured_steps": measured_steps,
        "request_count": len(replay_events),
        "request_count_by_step": {
            str(step): sum(1 for event in replay_events if event.get("step") == step) for step in measured_steps
        },
        "status_counts": dict(sorted(status_counts.items())),
        "clean_count": sum(analyze_latency._direct_clean(event.get("attributes", {})) for event in replay_events),
    }
    checks = {
        **{f"placement_{key}": value for key, value in placement_checks.items() if isinstance(value, bool)},
        **{f"manifest_{key}": value for key, value in manifest_checks.items() if isinstance(value, bool)},
        "tp_dp_in_log": launch_checks["tp_dp_in_log"],
        "dp_launch_count": launch_checks["dp_launch_count"],
        "dp_gpu_partitioning": launch_checks["dp_gpu_partitioning"],
        "engine_args_line_count": launch_checks["engine_args_line_count"] == 1,
        "replay_nonempty": replay["request_count"] > 0,
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "mode": mode,
        "replay_kind": "trace_replay",
        "expected": expected,
        "config_path": str(config_path),
        "placement_evidence": placement_checks,
        "manifest_evidence": manifest_checks,
        "launch_log_evidence": launch_checks,
        "request_trace_replay": replay,
        "checks": checks,
        "limitations": [
            "This is a post-run trace replay plus launch-log/placement audit; it is not a fresh GPU inference benchmark.",
            "The manifest proves the four allocated GPU UUIDs and engine width, while launch logs prove DP0/DP1 startup; they do not expose per-rank NCCL internals.",
        ],
    }


def _write_plot(path: Path, title: str, values: dict[str, dict[str, float]], ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    labels = list(values)
    keys = list(next(iter(values.values()))) if values else []
    x = list(range(len(labels)))
    width = 0.8 / max(1, len(keys))
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, key in enumerate(keys):
        ax.bar(
            [item + (index - (len(keys) - 1) / 2) * width for item in x],
            [values[label][key] for label in labels],
            width,
            label=key,
        )
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def analyze_variant(name: str, exp: Path, api_log: Path) -> dict[str, Any]:
    events, rows, window, request_events = _load_run(exp)
    measured_steps = sorted(
        {int(row.get("step")) for row in _read_jsonl(exp / "latency_markers" / "weight_serving_ready.jsonl")}
    )[1:]
    overlap = _latency_overlap(exp, events, window)
    transfer = _transfer_breakdown(exp)
    admission = _admission_breakdown(exp, events)
    env = _env_cpu_bubble(exp, rows)
    repeated = _repeated_computation(exp, events, request_events, rows, api_log)
    reward_wip = _reward_wip_breakdown(request_events, window, measured_steps, name)
    tpdp = _tpdp_replay(exp, api_log, name, request_events, measured_steps)
    reward_gpu = {
        "schema_version": 1,
        "measurement_window_s": list(window),
        "roles": overlap.get("reward_gpu_efficiency") or {},
        "semantics": (
            "NVML util_percent is a sampled hardware-busy indicator, not SM occupancy or model FLOP efficiency. "
            "Coverage and zero-request fractions are retained to prevent low utilization from being overinterpreted."
        ),
    }
    analysis_dir = exp / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "latency_overlap_exposure": overlap,
        "reward_wip_breakdown": reward_wip,
        "transfer_breakdown": transfer,
        "admission_breakdown": admission,
        "env_cpu_bubble": env,
        "environment_bubble": env,
        "repeated_computation": repeated,
        "repeated_compute": repeated,
        "reward_gpu_utilization": reward_gpu,
        "tpdp_replay": tpdp,
    }
    for stem, payload in outputs.items():
        (analysis_dir / f"{stem}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    stage_values = {
        name: {
            "rollout_admission_wait": overlap["inclusive_stage_occupancy_pct"].get("rollout_admission_wait", 0.0),
            "generation": overlap["inclusive_stage_occupancy_pct"].get("generation", 0.0),
            "reward": overlap["inclusive_stage_occupancy_pct"].get("reward", 0.0),
            "turn_judge": overlap["inclusive_stage_occupancy_pct"].get("turn_judge", 0.0),
            "transfer": overlap["inclusive_stage_occupancy_pct"].get("transfer", 0.0),
            "training": overlap["inclusive_stage_occupancy_pct"].get("training", 0.0),
            "weight_update": overlap["inclusive_stage_occupancy_pct"].get("weight_update", 0.0),
        }
    }
    _write_plot(
        analysis_dir / "latency_overlap_exposure.png",
        f"G5 latency overlap ({name})",
        stage_values,
        "inclusive occupancy (%)",
    )
    gpu = reward_gpu["roles"]
    gpu_values = {
        name: {
            role: float((gpu.get(role) or {}).get("util_percent", {}).get("mean_percent", 0.0) or 0.0)
            for role in ("judge_accuracy", "judge_multiturn_vlm")
        }
    }
    _write_plot(
        analysis_dir / "reward_gpu_utilization.png",
        f"Reward GPU utilization ({name})",
        gpu_values,
        "NVML util mean (%)",
    )
    prm = reward_wip["categories"]["prm"]
    wip_values = {
        "ORM": reward_wip["categories"]["orm"]["client_component_share_pct"],
        "PRM": prm["client_component_share_pct"],
    }
    _write_plot(
        analysis_dir / "reward_wip_breakdown.png",
        f"Reward request WIP decomposition ({name})",
        wip_values,
        "diagnostic share (%)",
    )
    server_values = {
        "ORM": {
            key.removeprefix("server_").removesuffix("_elapsed_s"): value
            for key, value in reward_wip["categories"]["orm"]["server_component_relative_to_request_total_pct"].items()
        },
        "PRM": {
            key.removeprefix("server_").removesuffix("_elapsed_s"): value
            for key, value in prm["server_component_relative_to_request_total_pct"].items()
        },
    }
    _write_plot(
        analysis_dir / "reward_wip_server_breakdown.png",
        f"Reward server-side components ({name})",
        server_values,
        "relative to server request-total (%)",
    )
    bubble_values = {name: {"idle_fraction": float(env["aggregate"]["idle_fraction_of_observed"] or 0.0) * 100.0}}
    _write_plot(
        analysis_dir / "environment_bubble.png",
        f"Environment CPU bubble ({name})",
        bubble_values,
        "idle fraction (%)",
    )
    # Keep concise aliases matching the experiment report checklist while
    # preserving the original names consumed by earlier analysis notebooks.
    (analysis_dir / "repeated_compute.json").write_text(
        json.dumps(repeated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (analysis_dir / "environment_bubble.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"name": name, "exp_dir": str(exp), **outputs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-exp", type=Path, required=True)
    parser.add_argument("--per-turn-exp", type=Path, required=True)
    parser.add_argument("--terminal-api-log", type=Path, required=True)
    parser.add_argument("--per-turn-api-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": 1,
        "terminal_once": analyze_variant("terminal_once", args.terminal_exp, args.terminal_api_log),
        "per_turn": analyze_variant("per_turn", args.per_turn_exp, args.per_turn_api_log),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
