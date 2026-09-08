# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import importlib.util
from decimal import Decimal
from pathlib import Path

import pytest


MODULE_PATH = Path("examples/mobilegym_agentic/scripts/gpu_hour_budget.py")
SPEC = importlib.util.spec_from_file_location("gpu_hour_budget", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gpu_hour_budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gpu_hour_budget)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("00:50:00", 3000), ("50:00", 3000), ("50", 3000), ("1-00:00:00", 86400)],
)
def test_parse_slurm_time_limit(value: str, expected: int) -> None:
    assert gpu_hour_budget.parse_slurm_time_limit(value) == expected


def test_budget_reserves_from_allocation_and_rejects_total_over_requested_cap(tmp_path: Path) -> None:
    ledger = tmp_path / "gpu_hours.tsv"
    first = gpu_hour_budget.reserve_gpu_hours(
        ledger_path=ledger,
        job_id="1",
        budget="120",
        allocated_gpus=24,
        time_limit="00:50:00",
        trigger="terminal_once",
        max_job_gpu_hours="20",
    )
    second = gpu_hour_budget.reserve_gpu_hours(
        ledger_path=ledger,
        job_id="2",
        budget="120",
        allocated_gpus=24,
        time_limit="00:50:00",
        trigger="per_turn",
        max_job_gpu_hours="20",
    )
    retry = gpu_hour_budget.reserve_gpu_hours(
        ledger_path=ledger,
        job_id="3",
        budget="120",
        allocated_gpus=24,
        time_limit="00:40:00",
        trigger="retry",
        max_job_gpu_hours="20",
    )

    assert (first, second, retry) == (Decimal("20"), Decimal("20"), Decimal("16"))
    with pytest.raises(RuntimeError, match="budget exhausted"):
        gpu_hour_budget.reserve_gpu_hours(
            ledger_path=ledger,
            job_id="4",
            budget="120",
            allocated_gpus=24,
            time_limit="03:00:00",
            trigger="extra",
        )


def test_budget_rejects_overstated_cap_and_sync_job_over_30_hours(tmp_path: Path) -> None:
    ledger = tmp_path / "gpu_hours.tsv"
    with pytest.raises(ValueError, match="cannot exceed"):
        gpu_hour_budget.reserve_gpu_hours(
            ledger_path=ledger,
            job_id="1",
            budget="261",
            allocated_gpus=24,
            time_limit="00:50:00",
            trigger="terminal_once",
        )
    with pytest.raises(ValueError, match="per-job cap"):
        gpu_hour_budget.reserve_gpu_hours(
            ledger_path=ledger,
            job_id="2",
            budget="120",
            allocated_gpus=24,
            time_limit="01:16:00",
            trigger="terminal_once",
            max_job_gpu_hours="30",
        )

    assert not Path(f"{ledger}.lock").exists()


def test_budget_fails_closed_on_duplicate_jobs_and_malformed_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "gpu_hours.tsv"
    gpu_hour_budget.reserve_gpu_hours(
        ledger_path=ledger,
        job_id="1",
        budget="120",
        allocated_gpus=24,
        time_limit="00:50:00",
        trigger="terminal_once",
    )
    with pytest.raises(ValueError, match="already has"):
        gpu_hour_budget.reserve_gpu_hours(
            ledger_path=ledger,
            job_id="1",
            budget="120",
            allocated_gpus=24,
            time_limit="00:50:00",
            trigger="terminal_once",
        )

    ledger.write_text("malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid ledger row"):
        gpu_hour_budget.reserve_gpu_hours(
            ledger_path=ledger,
            job_id="2",
            budget="120",
            allocated_gpus=24,
            time_limit="00:50:00",
            trigger="per_turn",
        )


def test_budget_rejects_tsv_control_characters(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single TSV field"):
        gpu_hour_budget.reserve_gpu_hours(
            ledger_path=tmp_path / "gpu_hours.tsv",
            job_id="1",
            budget="120",
            allocated_gpus=24,
            time_limit="00:50:00",
            trigger="per_turn\nforged",
        )


def test_budget_settlement_reclaims_only_measured_hours_and_preserves_audit(tmp_path: Path) -> None:
    ledger = tmp_path / "gpu_hours.tsv"
    gpu_hour_budget.reserve_gpu_hours(
        ledger_path=ledger,
        job_id="debug-1",
        budget="260",
        allocated_gpus=24,
        time_limit="01:15:00",
        trigger="terminal_once",
        max_job_gpu_hours="30",
    )
    settled = gpu_hour_budget.settle_gpu_hours(
        ledger_path=ledger,
        job_id="debug-1",
        actual_gpu_hours="3.806667",
    )
    assert settled == Decimal("3.806667")
    row = ledger.read_text(encoding="utf-8").strip().split("\t")
    assert row[:3] == ["debug-1", "3.806667", "settled"]
    assert row[6] == "30.000000"
    with pytest.raises(ValueError, match="already settled"):
        gpu_hour_budget.settle_gpu_hours(
            ledger_path=ledger,
            job_id="debug-1",
            actual_gpu_hours="1",
        )


def test_budget_settlement_rejects_measurement_above_reservation(tmp_path: Path) -> None:
    ledger = tmp_path / "gpu_hours.tsv"
    gpu_hour_budget.reserve_gpu_hours(
        ledger_path=ledger,
        job_id="debug-1",
        budget="260",
        allocated_gpus=24,
        time_limit="01:15:00",
        trigger="terminal_once",
        max_job_gpu_hours="30",
    )
    with pytest.raises(ValueError, match="exceeds reservation"):
        gpu_hour_budget.settle_gpu_hours(
            ledger_path=ledger,
            job_id="debug-1",
            actual_gpu_hours="30.000001",
        )
