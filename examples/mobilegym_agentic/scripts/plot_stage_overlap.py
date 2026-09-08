#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Plot exact stage overlap, exclusive wall time, and pipeline bubbles.

The input is a completed MobileGym experiment directory containing ``timeline/``,
``rollout_result/``, ``latency_markers/weight_serving_ready.jsonl``, and a
``clock_sync_audit.json``.  Unlike a min/max or percentile envelope, this tool
partitions the ready-to-ready measurement window at every observed span boundary.
Each resulting slice therefore has one exact set of active selected stages.

The figure answers three descriptive questions:

1. Where are the per-stage inactive gaps (lane-local pipeline bubbles)?
2. Which parts of a stage overlap another selected stage?
3. Which parts are observed with no other selected stage active (exclusive
   wall time, shown as an exposure candidate)?

Exclusive wall time is not, by itself, a causal critical-path contribution.  A
true critical-path claim additionally requires dependency lineage or a controlled
counterfactual.  Reward lanes are canonical request-WIP intervals (client queue
plus HTTP and client overhead), not reward-GPU kernel activity.  The figure also
plots the corresponding request-concurrency functions ``N_ORM(t)`` and
``N_PRM(t)``.  ``data_wait`` and
``transfer_buffer_wait`` are plotted as waits and deliberately do not count as
stage execution.

Example:
    python examples/mobilegym_agentic/scripts/plot_stage_overlap.py \
        --variant terminal_once=/path/to/exp/3124850 \
        --variant per_turn=/path/to/exp/3124159 \
        --warmup-updates 1 --measure-updates 8 \
        --output-prefix examples/mobilegym_agentic/latency_overlap_exposure
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


_AGENTIC_DUAL_JUDGE_DIR = Path(__file__).resolve().parents[2] / "agentic_dual_judge"
sys.path.insert(0, str(_AGENTIC_DUAL_JUDGE_DIR))
import analyze_latency  # noqa: E402  (the sibling example is not a Python package)


Interval = tuple[float, float]
ActiveSlice = tuple[float, float, tuple[str, ...]]
ConcurrencySlice = tuple[float, float, int]

COMPONENT_ORDER = (
    "Rollout generation",
    "Reward requests active (N>0)",
    "Transfer execution",
    "Training + optimizer",
    "Weight update",
)
COMPONENT_COLORS = {
    "Rollout generation": "#2389c9",
    "Reward requests active (N>0)": "#d95f02",
    "Transfer execution": "#4c78a8",
    "Training + optimizer": "#2ca25f",
    "Weight update": "#8c6bb1",
}

CANONICAL_REWARD_REQUEST_NAME = "critical_path.judge_request"
DIAGNOSTIC_ORDER = (
    "Outcome RM request WIP",
    "Process RM request WIP",
    "Session gate/admission wait",
    "Transfer buffer wait",
    "Trainer data_wait",
)
TRAINER_TRACE_UNAVAILABLE = "Trainer trace unavailable"
DIAGNOSTIC_COLORS = {
    "Outcome RM request WIP": "#e6ab02",
    "Process RM request WIP": "#a63603",
    "Session gate/admission wait": "#8c9699",
    "Transfer buffer wait": "#9da7a9",
    "Trainer data_wait": "#6f7c7d",
    TRAINER_TRACE_UNAVAILABLE: "#c44e52",
}


def _parse_named_path(spec: str) -> tuple[str, Path]:
    name, separator, raw_path = spec.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, Path(raw_path)


def _matches_component(event: dict[str, Any], component: str) -> bool:
    """Select non-wait stage intervals without accepting prefix-matched
    waits."""
    name = event["name"]
    if component == "Rollout generation":
        return name == "critical_path.rollout_generation"
    if component == "Reward requests active (N>0)":
        return name == CANONICAL_REWARD_REQUEST_NAME and _reward_service(event) is not None
    if component == "Transfer execution":
        return name == "critical_path.transfer"
    if component == "Training + optimizer":
        return event["stage"] == "training" and name != "critical_path.training_schedule"
    if component == "Weight update":
        return name == "critical_path.weight_update"
    raise ValueError(f"unknown component: {component}")


def _reward_service(event: dict[str, Any]) -> str | None:
    """Map one canonical queue-enter-to-request-end span to its RM lane."""
    if event["name"] != CANONICAL_REWARD_REQUEST_NAME:
        return None
    component = event.get("attributes", {}).get("component")
    if component == "answer_accuracy":
        return "Outcome RM request WIP"
    if component == "multi_turn_reasoning":
        return "Process RM request WIP"
    return None


def concurrency_slices(intervals: Iterable[Interval], window: Interval) -> list[ConcurrencySlice]:
    """Return the exact number of concurrently active intervals over
    ``window``."""
    window_start, window_end = window
    if window_end <= window_start:
        raise ValueError(f"measurement window must be positive, got {window}")
    deltas: defaultdict[float, int] = defaultdict(int)
    deltas[window_start] = 0
    deltas[window_end] = 0
    for start, end in intervals:
        clipped_start = max(window_start, float(start))
        clipped_end = min(window_end, float(end))
        if clipped_end <= clipped_start:
            continue
        deltas[clipped_start] += 1
        deltas[clipped_end] -= 1

    slices: list[ConcurrencySlice] = []
    active = 0
    previous = window_start
    for timestamp in sorted(deltas):
        if timestamp > previous:
            slices.append((previous, timestamp, active))
        active += deltas[timestamp]
        if active < 0:
            raise ValueError(f"request concurrency became negative at {timestamp}")
        previous = timestamp
    if previous < window_end:
        slices.append((previous, window_end, active))
    if active != 0:
        raise ValueError(f"request concurrency did not close at the window end: {active}")
    return slices


def merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Return the interval union as sorted, non-overlapping spans."""
    ordered = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[Interval] = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def clip_intervals(intervals: Iterable[Interval], window: Interval) -> list[Interval]:
    window_start, window_end = window
    return merge_intervals(
        (max(window_start, start), min(window_end, end))
        for start, end in intervals
        if min(window_end, end) > max(window_start, start)
    )


def partition_active_sets(component_intervals: dict[str, list[Interval]], window: Interval) -> list[ActiveSlice]:
    """Partition ``window`` into slices with an exact productive active set."""
    window_start, window_end = window
    boundaries: list[tuple[float, str, int]] = []
    for component in COMPONENT_ORDER:
        for start, end in clip_intervals(component_intervals.get(component, []), window):
            boundaries.append((start, component, 1))
            boundaries.append((end, component, -1))
    boundaries.sort(key=lambda item: item[0])

    active: Counter[str] = Counter()
    slices: list[ActiveSlice] = []
    previous = window_start
    index = 0
    while index < len(boundaries):
        timestamp = boundaries[index][0]
        if timestamp > previous:
            active_set = tuple(component for component in COMPONENT_ORDER if active[component] > 0)
            slices.append((previous, timestamp, active_set))
        while index < len(boundaries) and boundaries[index][0] == timestamp:
            _timestamp, component, delta = boundaries[index]
            active[component] += delta
            index += 1
        previous = timestamp
    if window_end > previous:
        active_set = tuple(component for component in COMPONENT_ORDER if active[component] > 0)
        slices.append((previous, window_end, active_set))

    coalesced: list[ActiveSlice] = []
    for start, end, active_set in slices:
        if coalesced and coalesced[-1][1] == start and coalesced[-1][2] == active_set:
            coalesced[-1] = (coalesced[-1][0], end, active_set)
        else:
            coalesced.append((start, end, active_set))
    return coalesced


def _duration(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in intervals)


def summarize_active_slices(
    component_intervals: dict[str, list[Interval]], active_slices: list[ActiveSlice], window: Interval
) -> dict[str, Any]:
    """Build additive exclusive/overlap summaries from an active-set
    partition."""
    total_s = window[1] - window[0]
    if total_s <= 0:
        raise ValueError(f"measurement window must be positive, got {window}")

    exclusive_by_component: defaultdict[str, float] = defaultdict(float)
    overlap_by_component: defaultdict[str, float] = defaultdict(float)
    active_set_s: defaultdict[str, float] = defaultdict(float)
    pairwise_s: dict[str, dict[str, float]] = {
        left: {right: 0.0 for right in COMPONENT_ORDER} for left in COMPONENT_ORDER
    }
    global_bubble_s = 0.0
    for start, end, active_set in active_slices:
        duration_s = end - start
        label = "+".join(active_set) if active_set else "no_productive_stage"
        active_set_s[label] += duration_s
        if not active_set:
            global_bubble_s += duration_s
            continue
        if len(active_set) == 1:
            exclusive_by_component[active_set[0]] += duration_s
        else:
            for component in active_set:
                overlap_by_component[component] += duration_s
        for left in active_set:
            for right in active_set:
                pairwise_s[left][right] += duration_s

    components = {}
    for component in COMPONENT_ORDER:
        active_s = _duration(clip_intervals(component_intervals.get(component, []), window))
        exclusive_s = exclusive_by_component[component]
        overlapped_s = overlap_by_component[component]
        if abs(active_s - exclusive_s - overlapped_s) > 1e-6:
            raise ValueError(
                f"active-set accounting mismatch for {component}: "
                f"active={active_s}, exclusive={exclusive_s}, overlap={overlapped_s}"
            )
        bubble_s = total_s - active_s
        components[component] = {
            "active_s": active_s,
            "exclusive_s": exclusive_s,
            "overlapped_s": overlapped_s,
            "bubble_s": bubble_s,
            "active_percent_of_window": 100.0 * active_s / total_s,
            "exclusive_percent_of_window": 100.0 * exclusive_s / total_s,
            "overlapped_percent_of_window": 100.0 * overlapped_s / total_s,
            "bubble_percent_of_window": 100.0 * bubble_s / total_s,
            "exclusive_percent_of_active": 100.0 * exclusive_s / active_s if active_s else None,
        }

    return {
        "window_s": total_s,
        "components": components,
        "global_bubble_s": global_bubble_s,
        "global_bubble_percent_of_window": 100.0 * global_bubble_s / total_s,
        "active_set_s": dict(sorted(active_set_s.items())),
        "pairwise_overlap_s": pairwise_s,
    }


def _clock_audit(exp_dir: Path, max_clock_offset_ms: float) -> dict[str, Any]:
    path = exp_dir / "clock_sync_audit.json"
    if not path.is_file():
        raise ValueError(f"missing cross-host clock audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    offset_ms = audit.get("max_pairwise_offset_ms")
    if isinstance(offset_ms, bool) or not isinstance(offset_ms, (int, float)):
        raise ValueError(f"clock audit has no numeric max_pairwise_offset_ms: {path}")
    if float(offset_ms) > max_clock_offset_ms:
        raise ValueError(f"clock offset {float(offset_ms):.3f} ms exceeds limit {max_clock_offset_ms:.3f} ms: {path}")
    return {
        "path": str(path),
        "max_pairwise_offset_ms": float(offset_ms),
        "limit_ms": max_clock_offset_ms,
    }


def build_variant_report(
    name: str,
    exp_dir: Path,
    *,
    warmup_updates: int,
    measure_updates: int | None,
    max_clock_offset_ms: float,
) -> tuple[dict[str, Any], list[ActiveSlice], dict[str, list[Interval]], dict[str, list[Interval]]]:
    """Load one real run and return its plot-ready interval accounting."""
    if not exp_dir.is_dir():
        raise ValueError(f"experiment directory does not exist: {exp_dir}")
    ready_marker_path = exp_dir / "latency_markers" / "weight_serving_ready.jsonl"
    if not ready_marker_path.is_file():
        raise ValueError(f"missing ready-marker file: {ready_marker_path}")
    clock_audit = _clock_audit(exp_dir, max_clock_offset_ms)

    events, source = analyze_latency.load_variant_events(
        exp_dir,
        include_rollout_with_timeline=True,
        ready_marker_path=ready_marker_path,
    )
    if "timeline" not in source:
        raise ValueError(f"{exp_dir}: exact stage overlap requires timeline input, got {source}")
    ready_boundaries = analyze_latency._ready_boundaries(events)
    resolved_measure_updates = measure_updates
    if resolved_measure_updates is None:
        resolved_measure_updates = len(ready_boundaries) - warmup_updates
    window, measured_steps, boundary_step = analyze_latency._fixed_k_window(
        events,
        warmup_updates=warmup_updates,
        measure_updates=resolved_measure_updates,
    )

    events_without_training_wait = analyze_latency._remove_stream_wait_from_training(events)
    component_intervals = {
        component: clip_intervals(
            (
                (event["start_s"], event["end_s"])
                for event in events_without_training_wait
                if _matches_component(event, component)
            ),
            window,
        )
        for component in COMPONENT_ORDER
    }
    request_intervals = {
        service: [
            (event["start_s"], event["end_s"])
            for event in events_without_training_wait
            if _reward_service(event) == service and event["end_s"] > window[0] and event["start_s"] < window[1]
        ]
        for service in DIAGNOSTIC_ORDER[:2]
    }
    diagnostic_intervals = {
        service: clip_intervals(request_intervals[service], window) for service in DIAGNOSTIC_ORDER[:2]
    }
    diagnostic_intervals["Session gate/admission wait"] = clip_intervals(
        (
            (event["start_s"], event["end_s"])
            for event in events_without_training_wait
            if event["name"] == "critical_path.rollout_admission_wait"
        ),
        window,
    )
    diagnostic_intervals["Transfer buffer wait"] = clip_intervals(
        (
            (event["start_s"], event["end_s"])
            for event in events_without_training_wait
            if event["name"] == "critical_path.transfer_buffer_wait"
        ),
        window,
    )
    diagnostic_intervals["Trainer data_wait"] = clip_intervals(
        (
            (event["start_s"], event["end_s"])
            for event in events_without_training_wait
            if event["name"] == "critical_path.data_wait"
        ),
        window,
    )
    trainer_event_counts = {
        step: {
            event_name: sum(1 for event in events if event.get("step") == step and event["name"] == event_name)
            for event_name in (
                "critical_path.training_schedule",
                "critical_path.optimizer_step",
                "critical_path.weight_update",
            )
        }
        for step in measured_steps
    }
    expected_trainer_rank_count = max(
        (count for counts in trainer_event_counts.values() for count in counts.values()), default=0
    )
    missing_trainer_steps = [
        step
        for step, counts in trainer_event_counts.items()
        if expected_trainer_rank_count == 0 or any(count < expected_trainer_rank_count for count in counts.values())
    ]
    ready_interval_by_step = {
        step: (ready_boundaries[previous_step], ready_boundaries[step])
        for previous_step, step in zip([boundary_step, *measured_steps[:-1]], measured_steps)
    }
    diagnostic_intervals[TRAINER_TRACE_UNAVAILABLE] = clip_intervals(
        (ready_interval_by_step[step] for step in missing_trainer_steps), window
    )
    active_slices = partition_active_sets(component_intervals, window)
    summary = summarize_active_slices(component_intervals, active_slices, window)
    total_s = summary["window_s"]
    diagnostic_summary = {
        diagnostic: {
            "active_s": (active_s := _duration(diagnostic_intervals[diagnostic])),
            "active_percent_of_window": 100.0 * active_s / total_s,
        }
        for diagnostic in (*DIAGNOSTIC_ORDER, TRAINER_TRACE_UNAVAILABLE)
    }
    request_concurrency = {}
    for service in DIAGNOSTIC_ORDER[:2]:
        slices = concurrency_slices(request_intervals[service], window)
        request_seconds = sum((end - start) * count for start, end, count in slices)
        request_concurrency[service] = {
            "definition": "canonical critical_path.judge_request: queue-enter to request-end",
            "request_count": len(request_intervals[service]),
            "peak_concurrency": max((count for _start, _end, count in slices), default=0),
            "mean_concurrency": request_seconds / total_s,
            "request_seconds": request_seconds,
            "slices": [
                {"start_s": start - window[0], "end_s": end - window[0], "concurrency": count}
                for start, end, count in slices
            ],
        }
    data_wait_s = diagnostic_summary["Trainer data_wait"]["active_s"]
    selected_ready_steps = [boundary_step, *measured_steps]
    relative_ready_boundaries_s = [ready_boundaries[step] - window[0] for step in selected_ready_steps]
    report = {
        "name": name,
        "exp_dir": str(exp_dir),
        "source": source,
        "event_count": len(events),
        "clock_audit": clock_audit,
        "window": {
            "start_s": window[0],
            "end_s": window[1],
            "duration_s": total_s,
            "boundary_step": boundary_step,
            "measured_steps": measured_steps,
            "relative_ready_boundaries_s": relative_ready_boundaries_s,
        },
        **summary,
        "diagnostics": diagnostic_summary,
        "request_concurrency": request_concurrency,
        "trainer_trace_audit": {
            "complete": not missing_trainer_steps,
            "expected_rank_count": expected_trainer_rank_count,
            "missing_steps": missing_trainer_steps,
            "event_counts_by_step": trainer_event_counts,
            "accounting_valid": not missing_trainer_steps,
        },
        "trainer_data_wait_s": data_wait_s,
        "trainer_data_wait_percent_of_window": 100.0 * data_wait_s / total_s,
    }
    return report, active_slices, component_intervals, diagnostic_intervals


def _relative_intervals(intervals: Iterable[Interval], origin: float) -> list[Interval]:
    return [(start - origin, end - origin) for start, end in intervals]


def _segments_for_component(active_slices: list[ActiveSlice], component: str, *, exclusive: bool) -> list[Interval]:
    return merge_intervals(
        (start, end)
        for start, end, active_set in active_slices
        if component in active_set and ((len(active_set) == 1) == exclusive)
    )


def plot_reports(
    plot_data: list[tuple[dict[str, Any], list[ActiveSlice], dict[str, list[Interval]], dict[str, list[Interval]]]],
    *,
    output_prefix: Path,
    title: str,
    dpi: int,
) -> tuple[Path, Path]:
    """Render PDF and PNG versions of the overlap/exposure figure."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("plot_stage_overlap.py requires matplotlib") from exc

    column_count = len(plot_data)
    figure = plt.figure(figsize=(8.0 * column_count, 13.0), constrained_layout=False)
    grid = figure.add_gridspec(3, column_count, height_ratios=(2.2, 0.82, 1.35), hspace=0.34, wspace=0.22)
    figure.suptitle(title, fontsize=16, fontweight="bold", y=0.985)

    for column, (report, active_slices, _component_intervals, diagnostic_intervals) in enumerate(plot_data):
        window_start = report["window"]["start_s"]
        total_s = report["window_s"]
        ready_boundaries_s = report["window"]["relative_ready_boundaries_s"]

        gantt = figure.add_subplot(grid[0, column])
        row_labels = [
            COMPONENT_ORDER[0],
            "Session gate/admission wait",
            COMPONENT_ORDER[1],
            COMPONENT_ORDER[2],
            "Transfer buffer wait",
            COMPONENT_ORDER[3],
            COMPONENT_ORDER[4],
            "Trainer data_wait",
            TRAINER_TRACE_UNAVAILABLE,
            "Observed state",
        ]
        row_positions = {label: len(row_labels) - index - 1 for index, label in enumerate(row_labels)}
        row_height = 0.66

        for component in COMPONENT_ORDER:
            y = row_positions[component]
            gantt.broken_barh([(0.0, total_s)], (y - row_height / 2, row_height), facecolors="#eeeeee")
            overlapped = _relative_intervals(
                _segments_for_component(active_slices, component, exclusive=False), window_start
            )
            exclusive = _relative_intervals(
                _segments_for_component(active_slices, component, exclusive=True), window_start
            )
            if overlapped:
                gantt.broken_barh(
                    [(start, end - start) for start, end in overlapped],
                    (y - row_height / 2, row_height),
                    facecolors=COMPONENT_COLORS[component],
                    edgecolors=COMPONENT_COLORS[component],
                    alpha=0.34,
                    hatch="////",
                    linewidth=0.3,
                )
            if exclusive:
                gantt.broken_barh(
                    [(start, end - start) for start, end in exclusive],
                    (y - row_height / 2, row_height),
                    facecolors=COMPONENT_COLORS[component],
                    edgecolors=COMPONENT_COLORS[component],
                    linewidth=0.3,
                )

        for diagnostic in (*DIAGNOSTIC_ORDER[2:], TRAINER_TRACE_UNAVAILABLE):
            y = row_positions[diagnostic]
            gantt.broken_barh([(0.0, total_s)], (y - row_height / 2, row_height), facecolors="#f7f7f7")
            relative_intervals = _relative_intervals(diagnostic_intervals[diagnostic], window_start)
            if not relative_intervals:
                continue
            gantt.broken_barh(
                [(start, end - start) for start, end in relative_intervals],
                (y - row_height / 2, row_height),
                facecolors=DIAGNOSTIC_COLORS[diagnostic],
                edgecolors=DIAGNOSTIC_COLORS[diagnostic],
                alpha=0.76,
                hatch="xxxx" if diagnostic == TRAINER_TRACE_UNAVAILABLE else None,
                linewidth=0.3,
            )

        state_y = row_positions["Observed state"]
        for start, end, active_set in active_slices:
            relative_start = start - window_start
            if not active_set:
                color, alpha, hatch = "#444444", 0.90, None
            elif len(active_set) == 1:
                color, alpha, hatch = COMPONENT_COLORS[active_set[0]], 1.0, None
            else:
                color, alpha, hatch = "#7a5195", 0.65, "////"
            gantt.broken_barh(
                [(relative_start, end - start)],
                (state_y - row_height / 2, row_height),
                facecolors=color,
                edgecolors=color,
                alpha=alpha,
                hatch=hatch,
                linewidth=0.2,
            )

        ready_steps = [report["window"]["boundary_step"], *report["window"]["measured_steps"]]
        for marker_index, boundary in enumerate(ready_boundaries_s):
            gantt.axvline(boundary, color="#555555", linestyle=":", linewidth=0.75, alpha=0.75)
            horizontal_alignment = "center"
            if marker_index == 0:
                horizontal_alignment = "left"
            elif marker_index == len(ready_boundaries_s) - 1:
                horizontal_alignment = "right"
            gantt.annotate(
                f"R{ready_steps[marker_index]}",
                xy=(boundary, 0.995),
                xycoords=("data", "axes fraction"),
                xytext=(0, -1),
                textcoords="offset points",
                ha=horizontal_alignment,
                va="top",
                fontsize=7,
                color="#555555",
            )
        gantt.set_xlim(0.0, total_s)
        gantt.set_ylim(-0.65, len(row_labels) - 0.35)
        gantt.set_yticks([row_positions[label] for label in row_labels], row_labels)
        if column > 0:
            gantt.tick_params(axis="y", labelleft=False)
        gantt.set_xlabel("Elapsed time inside selected ready-to-ready window (s)")
        gantt.grid(axis="x", color="#dddddd", linewidth=0.5, alpha=0.6)
        gantt.set_axisbelow(True)
        gantt.set_title(
            f"{report['name']}  |  T={total_s:.1f}s  |  ready steps "
            f"{report['window']['boundary_step']}→{report['window']['measured_steps'][-1]}",
            fontsize=12,
            fontweight="bold",
        )

        concurrency_axis = figure.add_subplot(grid[1, column])
        for service, short_label in (
            ("Outcome RM request WIP", "N_ORM(t)"),
            ("Process RM request WIP", "N_PRM(t)"),
        ):
            values = report["request_concurrency"][service]
            slices = values["slices"]
            edges = [slices[0]["start_s"], *(item["end_s"] for item in slices)]
            counts = [item["concurrency"] for item in slices]
            concurrency_axis.stairs(
                counts,
                edges,
                label=(f"{short_label}: peak={values['peak_concurrency']}, mean={values['mean_concurrency']:.1f}"),
                color=DIAGNOSTIC_COLORS[service],
                linewidth=1.0,
            )
        for boundary in ready_boundaries_s:
            concurrency_axis.axvline(boundary, color="#777777", linestyle=":", linewidth=0.55, alpha=0.55)
        concurrency_axis.set_xlim(0.0, total_s)
        concurrency_axis.set_ylim(bottom=0)
        concurrency_axis.set_ylabel("In-flight\nrequests")
        concurrency_axis.set_xlabel("Elapsed time inside selected ready-to-ready window (s)")
        concurrency_axis.grid(axis="both", color="#e1e1e1", linewidth=0.45, alpha=0.65)
        concurrency_axis.set_axisbelow(True)
        concurrency_axis.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2)
        concurrency_axis.set_title("Canonical request concurrency: queue-enter → request-end", fontsize=10)

        summary_axis = figure.add_subplot(grid[2, column])
        trace_audit = report["trainer_trace_audit"]
        if not trace_audit["accounting_valid"]:
            summary_axis.set_axis_off()
            missing = ", ".join(str(step) for step in trace_audit["missing_steps"])
            summary_axis.text(
                0.5,
                0.58,
                "Observed wall-time accounting suppressed",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="#9c2f2f",
                transform=summary_axis.transAxes,
            )
            summary_axis.text(
                0.5,
                0.40,
                f"trainer critical-path trace is incomplete for steps: {missing}\n"
                "blank intervals are unknown, not idle/bubble",
                ha="center",
                va="center",
                fontsize=10,
                color="#6b2929",
                transform=summary_axis.transAxes,
            )
            continue
        summary_labels = [
            *COMPONENT_ORDER,
            "Session gate/admission wait",
            "Transfer buffer wait",
            "Trainer data_wait",
            "Global no-stage bubble",
        ]
        summary_positions = {label: len(summary_labels) - index - 1 for index, label in enumerate(summary_labels)}
        for component in COMPONENT_ORDER:
            values = report["components"][component]
            y = summary_positions[component]
            exclusive_pct = values["exclusive_percent_of_window"]
            overlap_pct = values["overlapped_percent_of_window"]
            bubble_pct = values["bubble_percent_of_window"]
            summary_axis.barh(y, exclusive_pct, color=COMPONENT_COLORS[component], height=0.62)
            summary_axis.barh(
                y,
                overlap_pct,
                left=exclusive_pct,
                color=COMPONENT_COLORS[component],
                alpha=0.34,
                hatch="////",
                edgecolor=COMPONENT_COLORS[component],
                height=0.62,
            )
            summary_axis.barh(
                y,
                bubble_pct,
                left=exclusive_pct + overlap_pct,
                color="#eeeeee",
                edgecolor="#dddddd",
                height=0.62,
            )
            summary_axis.text(
                min(99.0, exclusive_pct + overlap_pct + 0.8),
                y,
                f"S {exclusive_pct:.1f}%  O {overlap_pct:.1f}%",
                va="center",
                fontsize=8,
            )

        for diagnostic in ("Session gate/admission wait", "Transfer buffer wait", "Trainer data_wait"):
            diagnostic_pct = report["diagnostics"][diagnostic]["active_percent_of_window"]
            y = summary_positions[diagnostic]
            summary_axis.barh(y, diagnostic_pct, color=DIAGNOSTIC_COLORS[diagnostic], height=0.62)
            summary_axis.barh(y, 100.0 - diagnostic_pct, left=diagnostic_pct, color="#f7f7f7", height=0.62)
            summary_axis.text(min(99.0, diagnostic_pct + 0.8), y, f"{diagnostic_pct:.1f}%", va="center", fontsize=8)

        global_bubble_pct = report["global_bubble_percent_of_window"]
        y = summary_positions["Global no-stage bubble"]
        summary_axis.barh(y, global_bubble_pct, color="#444444", height=0.62)
        summary_axis.barh(y, 100.0 - global_bubble_pct, left=global_bubble_pct, color="#f7f7f7", height=0.62)
        summary_axis.text(min(99.0, global_bubble_pct + 0.8), y, f"{global_bubble_pct:.1f}%", va="center", fontsize=8)

        summary_axis.set_xlim(0.0, 100.0)
        summary_axis.set_yticks([summary_positions[label] for label in summary_labels], summary_labels)
        if column > 0:
            summary_axis.tick_params(axis="y", labelleft=False)
        summary_axis.set_xlabel("Share of selected window (%)")
        summary_axis.grid(axis="x", color="#dddddd", linewidth=0.5, alpha=0.7)
        summary_axis.set_axisbelow(True)
        summary_axis.set_title("Observed wall-time accounting", fontsize=11, fontweight="bold")

    legend_handles = [
        Patch(facecolor="#555555", label="solo-active: exactly one selected stage active"),
        Patch(facecolor="#999999", alpha=0.34, hatch="////", label="overlapped: ≥2 selected stages active"),
        Patch(facecolor="#eeeeee", edgecolor="#dddddd", label="stage inactive / lane-local bubble"),
        Patch(facecolor="#7f8c8d", alpha=0.70, label="diagnostic WIP/wait lane (not in active-set count)"),
        Patch(facecolor="#444444", label="global: no selected stage span observed"),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", ncol=3, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.015)
    )
    figure.text(
        0.5,
        0.003,
        "Solo-active wall time is an observed exposure candidate, not causal critical-path attribution. "
        "Reward is request WIP (queue + HTTP + client overhead), not GPU compute; transfer-buffer/data waits are "
        "excluded. Session-gate/admission wait is diagnostic residence, not execution. R0→R2 includes every "
        "observed physical span in the window, including step-0 warmup spill; measured lineage is steps 1 and 2. "
        "Global bubble is not system idle because environment execution lacks a leaf span.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )
    figure.subplots_adjust(top=0.94, bottom=0.105, left=0.10, right=0.98)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_prefix.with_suffix(".pdf")
    png_path = output_prefix.with_suffix(".png")
    for target, image_format, save_dpi in ((pdf_path, "pdf", None), (png_path, "png", dpi)):
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.stem}.",
            suffix=f".{image_format}",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            figure.savefig(temporary_path, format=image_format, dpi=save_dpi, bbox_inches="tight")
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variant",
        action="append",
        type=_parse_named_path,
        required=True,
        metavar="NAME=EXP_DIR",
        help="completed experiment directory; repeat to draw variants side by side",
    )
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=1,
        help="number of ready boundaries before the selected measurement window (minimum 1; default: 1)",
    )
    parser.add_argument(
        "--measure-updates",
        type=int,
        default=None,
        help="number of ready-to-ready intervals; default uses every complete interval after warmup",
    )
    parser.add_argument("--max-clock-offset-ms", type=float, default=10.0)
    parser.add_argument("--output-prefix", type=Path, required=True, help="writes PREFIX.pdf, PREFIX.png, PREFIX.json")
    parser.add_argument(
        "--title", default="Agentic MLLM RL: observed stage overlap, solo-active candidates, and bubbles"
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    names = [name for name, _path in args.variant]
    if len(set(names)) != len(names):
        parser.error(f"duplicate variant names are not allowed: {names}")
    if args.warmup_updates < 1:
        parser.error("--warmup-updates must be at least 1 for a ready-to-ready window")
    if args.measure_updates is not None and args.measure_updates < 1:
        parser.error("--measure-updates must be positive")
    if args.max_clock_offset_ms < 0:
        parser.error("--max-clock-offset-ms must be non-negative")

    plot_data = [
        build_variant_report(
            name,
            exp_dir,
            warmup_updates=args.warmup_updates,
            measure_updates=args.measure_updates,
            max_clock_offset_ms=args.max_clock_offset_ms,
        )
        for name, exp_dir in args.variant
    ]
    pdf_path, png_path = plot_reports(plot_data, output_prefix=args.output_prefix, title=args.title, dpi=args.dpi)
    report_path = args.output_prefix.with_suffix(".json")
    output = {
        "schema_version": 2,
        "semantics": {
            "selected_components": {
                "Rollout generation": "critical_path.rollout_generation",
                "Reward requests active (N>0)": (
                    "union of canonical outcome/process critical_path.judge_request parents from queue-enter to "
                    "request-end, projected to a Boolean active indicator for overlap accounting; includes client "
                    "semaphore queue, HTTP, and client/server overhead, so it is not reward-GPU compute"
                ),
                "Transfer execution": "critical_path.transfer only; transfer_buffer_wait is excluded",
                "Training + optimizer": (
                    "training spans after rank-local data_wait subtraction, plus optimizer; nested spans are unioned"
                ),
                "Weight update": "critical_path.weight_update only; weight_gate_wait is excluded",
            },
            "exclusive": "component active while no other selected component is active",
            "overlapped": "component active while at least one other selected component is active",
            "lane_bubble": "component inactive inside the selected ready-to-ready window",
            "global_bubble": (
                "no selected component has an observed span; this is not system idle because environment execution "
                "does not have a separate leaf span in these traces"
            ),
            "diagnostic_lanes": (
                "the Gantt displays rollout_admission_wait, transfer_buffer_wait, and trainer data_wait; per-role "
                "Boolean request unions are retained in JSON diagnostics but not plotted as WIP lanes; all are "
                "excluded from the selected-component active-set partition"
            ),
            "window_lineage": (
                "the R0-to-R2 plot includes every observed physical span intersecting the window, including any "
                "step-0 warmup spill; the formal measured update lineage is steps 1 and 2"
            ),
            "request_concurrency": (
                "N_ORM(t) and N_PRM(t) count canonical queue-enter-to-request-end spans and preserve request "
                "multiplicity"
            ),
            "trainer_data_wait": (
                "union of observed rank-local trainer wait spans; excluded from the selected-component active set "
                "and not proof that every trainer rank or GPU is idle"
            ),
            "trace_completeness": (
                "wall-time accounting is suppressed for a variant when any measured step lacks the expected "
                "rank coverage for training_schedule, optimizer_step, or weight_update"
            ),
            "clock_uncertainty": (
                "cross-host overlaps near the clock-audit bound should be treated as uncertain; the per-run bound "
                "is recorded under clock_audit"
            ),
            "critical_path_caveat": (
                "solo-active observed wall time is an exposure candidate, not causal critical-path contribution; "
                "dependency lineage or a controlled counterfactual is still required"
            ),
        },
        "variants": [item[0] for item in plot_data],
    }
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
