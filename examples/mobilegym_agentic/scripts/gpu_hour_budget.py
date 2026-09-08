# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Conservative GPU-hour admission for the MobileGym sync experiment."""

import argparse
import os
import signal
import time
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator


# The initial campaign consumed 140 GPU-hours. The user authorized the planned
# additional 60 GPU-hours and a further 40 GPU-hours for bounded retries and
# causal replay, so the hard cap is 260 GPU-hours for this checkout. Callers
# cannot bypass the cap through a larger --budget value.  A cancelled/failed
# reservation may be settled once with measured Slurm usage; the original
# reservation is retained in an audit column and active jobs remain
# conservatively accounted for at their full TimeLimit.
MAX_EXPERIMENT_GPU_HOURS = Decimal("260")


def parse_slurm_time_limit(value: str) -> int:
    """Convert a Slurm ``TimeLimit`` value to seconds."""
    normalized = value.strip()
    if not normalized or normalized.upper() in {"UNLIMITED", "PARTITION_LIMIT", "NOT_SET"}:
        raise ValueError(f"finite Slurm TimeLimit required, got {value!r}")

    days = 0
    clock = normalized
    if "-" in normalized:
        day_text, clock = normalized.split("-", 1)
        days = int(day_text)

    fields = clock.split(":")
    if len(fields) == 3:
        hours, minutes, seconds = (int(field) for field in fields)
    elif len(fields) == 2:
        hours = 0
        minutes, seconds = (int(field) for field in fields)
    elif len(fields) == 1:
        hours = 0
        minutes = int(fields[0])
        seconds = 0
    else:
        raise ValueError(f"invalid Slurm TimeLimit {value!r}")
    if days < 0 or hours < 0 or not 0 <= minutes < 60 or not 0 <= seconds < 60:
        raise ValueError(f"invalid Slurm TimeLimit {value!r}")
    total_seconds = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total_seconds <= 0:
        raise ValueError(f"positive Slurm TimeLimit required, got {value!r}")
    return total_seconds


def _positive_decimal(value: str, *, name: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return parsed


def _validate_ledger_field(value: str, *, name: str) -> None:
    if not value or any(character in value for character in "\t\r\n"):
        raise ValueError(f"{name} must be a non-empty single TSV field, got {value!r}")


def _read_ledger(ledger_path: Path) -> tuple[list[list[str]], Decimal, set[str]]:
    """Read and validate the ledger, returning rows, used hours, and IDs."""
    rows: list[list[str]] = []
    used = Decimal(0)
    seen_job_ids: set[str] = set()
    if not ledger_path.exists():
        return rows, used, seen_job_ids
    for line_number, line in enumerate(ledger_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) not in {6, 7} or fields[2] not in {"reserved", "settled"}:
            raise ValueError(f"invalid ledger row at {ledger_path}:{line_number}")
        if fields[0] in seen_job_ids:
            raise ValueError(f"duplicate job id in GPU-hour ledger at {ledger_path}:{line_number}")
        seen_job_ids.add(fields[0])
        try:
            accounted_hours = Decimal(fields[1])
        except InvalidOperation as exc:
            raise ValueError(f"invalid reservation at {ledger_path}:{line_number}") from exc
        if not accounted_hours.is_finite() or accounted_hours < 0:
            raise ValueError(f"invalid reservation at {ledger_path}:{line_number}")
        if len(fields) == 7:
            try:
                original_reservation = Decimal(fields[6])
            except InvalidOperation as exc:
                raise ValueError(f"invalid original reservation at {ledger_path}:{line_number}") from exc
            if not original_reservation.is_finite() or original_reservation < accounted_hours:
                raise ValueError(f"invalid original reservation at {ledger_path}:{line_number}")
        rows.append(fields)
        used += accounted_hours
    return rows, used, seen_job_ids


@contextmanager
def _directory_lock(lock_path: Path, timeout_s: float = 60.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout_s
    acquired = False
    while not acquired:
        try:
            lock_path.mkdir()
            acquired = True
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"could not acquire GPU-hour ledger lock {lock_path}")
            time.sleep(0.25)

    owner_path = lock_path / "owner"
    try:
        owner_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
        yield
    finally:
        owner_path.unlink(missing_ok=True)
        try:
            lock_path.rmdir()
        except FileNotFoundError:
            pass


def reserve_gpu_hours(
    *,
    ledger_path: Path,
    job_id: str,
    budget: str,
    allocated_gpus: int,
    time_limit: str,
    trigger: str,
    max_job_gpu_hours: str | None = None,
) -> Decimal:
    """Atomically reserve the allocation's worst-case GPU hours."""
    _validate_ledger_field(job_id, name="job_id")
    _validate_ledger_field(trigger, name="trigger")
    requested_budget = _positive_decimal(budget, name="budget")
    if requested_budget > MAX_EXPERIMENT_GPU_HOURS:
        raise ValueError(
            f"budget cannot exceed the experiment cap of {MAX_EXPERIMENT_GPU_HOURS} GPU-hours, got {requested_budget}"
        )
    if allocated_gpus <= 0:
        raise ValueError(f"allocated_gpus must be positive, got {allocated_gpus}")

    time_limit_s = parse_slurm_time_limit(time_limit)
    reservation = Decimal(allocated_gpus * time_limit_s) / Decimal(3600)
    if max_job_gpu_hours is not None:
        job_cap = _positive_decimal(max_job_gpu_hours, name="max_job_gpu_hours")
        if reservation > job_cap:
            raise ValueError(f"job reserves {reservation:.6f} GPU-hours, exceeding the per-job cap of {job_cap}")

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with _directory_lock(Path(f"{ledger_path}.lock")):
        _, used, seen_job_ids = _read_ledger(ledger_path)
        if job_id in seen_job_ids:
            raise ValueError(f"job id {job_id!r} already has a GPU-hour reservation")
        if used + reservation > requested_budget:
            raise RuntimeError(
                f"GPU-hour budget exhausted: used={used:.6f}, requested={reservation:.6f}, "
                f"budget={requested_budget:.6f}"
            )
        with ledger_path.open("a", encoding="utf-8") as ledger:
            ledger.write(f"{job_id}\t{reservation:.6f}\treserved\t{trigger}\t{allocated_gpus}\t{time_limit}\n")
            ledger.flush()
            os.fsync(ledger.fileno())
    return reservation


def settle_gpu_hours(*, ledger_path: Path, job_id: str, actual_gpu_hours: str) -> Decimal:
    """Replace one completed reservation with measured Slurm GPU-hours.

    Settlement is deliberately explicit and idempotency-safe: only an existing
    ``reserved`` row can be settled, and a second attempt fails instead of
    silently changing accounting.  The seventh TSV field retains the original
    worst-case reservation for auditability.
    """
    _validate_ledger_field(job_id, name="job_id")
    measured = _positive_decimal(actual_gpu_hours, name="actual_gpu_hours")
    if not ledger_path.exists():
        raise ValueError(f"job id {job_id!r} is not present in GPU-hour ledger")
    with _directory_lock(Path(f"{ledger_path}.lock")):
        rows, _, _ = _read_ledger(ledger_path)
        matching: list[str] | None = None
        for row in rows:
            if row[0] == job_id:
                matching = row
                break
        if matching is None:
            raise ValueError(f"job id {job_id!r} is not present in GPU-hour ledger")
        if matching[2] != "reserved":
            raise ValueError(f"job id {job_id!r} is already {matching[2]}")
        original = matching[1]
        if measured > Decimal(original):
            raise ValueError(f"actual_gpu_hours {measured} exceeds reservation {original} for job {job_id!r}")
        matching[1] = f"{measured:.6f}"
        matching[2] = "settled"
        if len(matching) == 6:
            matching.append(f"{Decimal(original):.6f}")
        ledger_path.write_text(
            "".join("\t".join(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    return measured


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--job-id")
    parser.add_argument("--budget")
    parser.add_argument("--allocated-gpus", type=int)
    parser.add_argument("--time-limit")
    parser.add_argument("--trigger")
    parser.add_argument("--max-job-gpu-hours")
    parser.add_argument("--settle-job-id")
    parser.add_argument("--actual-gpu-hours")
    args = parser.parse_args()

    def _terminate(_signum: int, _frame: object) -> None:
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, _terminate)
    if args.settle_job_id is not None:
        if args.actual_gpu_hours is None:
            parser.error("--actual-gpu-hours is required with --settle-job-id")
        settled = settle_gpu_hours(
            ledger_path=args.ledger,
            job_id=args.settle_job_id,
            actual_gpu_hours=args.actual_gpu_hours,
        )
        print(f"{settled:.6f}")
        return
    required = {
        "--job-id": args.job_id,
        "--budget": args.budget,
        "--allocated-gpus": args.allocated_gpus,
        "--time-limit": args.time_limit,
        "--trigger": args.trigger,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    reservation = reserve_gpu_hours(
        ledger_path=args.ledger,
        job_id=args.job_id,
        budget=args.budget,
        allocated_gpus=args.allocated_gpus,
        time_limit=args.time_limit,
        trigger=args.trigger,
        max_job_gpu_hours=args.max_job_gpu_hours,
    )
    print(f"{reservation:.6f}")


if __name__ == "__main__":
    main()
