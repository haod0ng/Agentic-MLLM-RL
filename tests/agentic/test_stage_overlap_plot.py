# Copyright (c) 2026 Relax Authors. All Rights Reserved.

from examples.mobilegym_agentic.scripts.plot_stage_overlap import (
    COMPONENT_ORDER,
    _matches_component,
    _reward_service,
    concurrency_slices,
    merge_intervals,
    partition_active_sets,
    summarize_active_slices,
)


def test_merge_intervals_unions_touching_and_overlapping_spans():
    assert merge_intervals([(3.0, 4.0), (0.0, 2.0), (1.0, 3.0), (7.0, 7.0)]) == [(0.0, 4.0)]


def test_active_set_partition_accounts_for_exclusive_overlap_and_bubble():
    intervals = {component: [] for component in COMPONENT_ORDER}
    intervals["Rollout generation"] = [(0.0, 4.0)]
    intervals["Reward requests active (N>0)"] = [(2.0, 6.0)]
    intervals["Training + optimizer"] = [(7.0, 9.0)]

    slices = partition_active_sets(intervals, (0.0, 10.0))

    assert slices == [
        (0.0, 2.0, ("Rollout generation",)),
        (2.0, 4.0, ("Rollout generation", "Reward requests active (N>0)")),
        (4.0, 6.0, ("Reward requests active (N>0)",)),
        (6.0, 7.0, ()),
        (7.0, 9.0, ("Training + optimizer",)),
        (9.0, 10.0, ()),
    ]

    report = summarize_active_slices(intervals, slices, (0.0, 10.0))
    assert report["components"]["Rollout generation"]["exclusive_s"] == 2.0
    assert report["components"]["Rollout generation"]["overlapped_s"] == 2.0
    assert report["components"]["Reward requests active (N>0)"]["exclusive_s"] == 2.0
    assert report["components"]["Reward requests active (N>0)"]["overlapped_s"] == 2.0
    assert report["components"]["Training + optimizer"]["exclusive_s"] == 2.0
    assert report["components"]["Training + optimizer"]["overlapped_s"] == 0.0
    assert report["global_bubble_s"] == 2.0
    assert report["pairwise_overlap_s"]["Rollout generation"]["Reward requests active (N>0)"] == 2.0
    assert sum(report["active_set_s"].values()) == 10.0
    assert (
        report["pairwise_overlap_s"]["Rollout generation"]["Reward requests active (N>0)"]
        == report["pairwise_overlap_s"]["Reward requests active (N>0)"]["Rollout generation"]
    )


def test_component_matching_keeps_transfer_wait_out_of_execution():
    transfer = {"name": "critical_path.transfer", "stage": "transfer"}
    buffer_wait = {"name": "critical_path.transfer_buffer_wait", "stage": "transfer"}

    assert _matches_component(transfer, "Transfer execution")
    assert not _matches_component(buffer_wait, "Transfer execution")


def test_reward_matching_uses_only_canonical_request_parent():
    request = {
        "name": "critical_path.judge_request",
        "stage": "request",
        "attributes": {"component": "answer_accuracy"},
    }
    outer_turn_operation = {
        "name": "critical_path.turn_judge",
        "stage": "turn_judge",
        "attributes": {"component": "multi_turn_reasoning"},
    }

    assert _reward_service(request) == "Outcome RM request WIP"
    assert _matches_component(request, "Reward requests active (N>0)")
    assert _reward_service(outer_turn_operation) is None
    assert not _matches_component(outer_turn_operation, "Reward requests active (N>0)")


def test_concurrency_slices_preserve_request_multiplicity():
    slices = concurrency_slices([(1.0, 4.0), (2.0, 3.0), (4.0, 5.0)], (0.0, 6.0))

    assert slices == [
        (0.0, 1.0, 0),
        (1.0, 2.0, 1),
        (2.0, 3.0, 2),
        (3.0, 4.0, 1),
        (4.0, 5.0, 1),
        (5.0, 6.0, 0),
    ]
