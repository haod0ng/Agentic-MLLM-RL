#!/usr/bin/env python3

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Export a public, privacy-preserving MobileGym performance bundle.

The exporter keeps raw *performance telemetry* required to reproduce the
latency analysis (timeline spans, GPU samples, TransferQueue traces, and the
rollout latency fields).  It intentionally excludes agent prompts, responses,
screenshots, browser/session files, and cluster-identifying fields.  This lets
an experiment be reviewed without publishing benchmark interaction content or
site-specific infrastructure identifiers.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_DROP_VALUE_KEYS = frozenset(
    {
        "call_stack",
        "command",
        "dist_init_addr",
        "flashinfer_workspace_base",
        "gpu_uuids",
        "model_path",
        "server_host",
    }
)
_NODE_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:nid\d+|clariden(?:-[\w-]+)?)(?![A-Za-z0-9])", flags=re.IGNORECASE)
_PATH_PATTERN = re.compile(r"/(?:iopsstor|capstor|users|opt)/[^\s\"']+")
_IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


@dataclass
class Sanitizer:
    """Remove infrastructure identity while retaining telemetry shape."""

    node_aliases: dict[str, str] = field(default_factory=dict)

    def value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {
                self.value(key): "<redacted>" if key in _DROP_VALUE_KEYS else self.value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if not isinstance(value, str):
            return value
        value = _PATH_PATTERN.sub("<path>", value)
        value = _IPV4_PATTERN.sub("<ip>", value)

        def replace_node(match: re.Match[str]) -> str:
            node = match.group(0).lower()
            if node not in self.node_aliases:
                self.node_aliases[node] = f"node-{len(self.node_aliases) + 1}"
            return self.node_aliases[node]

        return _NODE_PATTERN.sub(replace_node, value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_text_for_write(path) as output_file:
        output_file.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _open_text_for_write(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def _compressed(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".gz")


def _export_json(source: Path, destination: Path, sanitizer: Sanitizer) -> None:
    _write_json(destination, sanitizer.value(json.loads(source.read_text(encoding="utf-8", errors="replace"))))


def _export_jsonl(source: Path, destination: Path, sanitizer: Sanitizer) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open(encoding="utf-8", errors="replace") as input_file,
        _open_text_for_write(destination) as output_file,
    ):
        for line in input_file:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            output_file.write(json.dumps(sanitizer.value(value), sort_keys=True) + "\n")


def _export_rollout_latency(source_dir: Path, destination_dir: Path, sanitizer: Sanitizer) -> None:
    """Retain timing fields but never export interaction text or visual
    inputs."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    allowed = (
        "agent_turns",
        "group_index",
        "image_count",
        "image_token_count",
        "latency_trace",
        "multimodal_token_count",
        "prompt_length",
        "prompt_token_count",
        "response_length",
        "response_token_count",
        "rollout_id",
        "sample_index",
        "status",
        "total_length",
        "total_token_count",
        "weight_versions",
    )
    row_count = 0
    output_file = None
    try:
        for source in sorted(source_dir.glob("*.jsonl")):
            with source.open(encoding="utf-8", errors="replace") as input_file:
                for line in input_file:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row_count % 96 == 0:
                        if output_file is not None:
                            output_file.close()
                        output_file = _open_text_for_write(
                            destination_dir / f"rollout_latency_part{row_count // 96:02d}.jsonl.gz"
                        )
                    output_file.write(
                        json.dumps(
                            sanitizer.value({key: row.get(key) for key in allowed if key in row}), sort_keys=True
                        )
                        + "\n"
                    )
                    row_count += 1
    finally:
        if output_file is not None:
            output_file.close()


def _copy_figures(source_dir: Path, destination_dir: Path) -> Iterable[Path]:
    for suffix in ("*.png", "*.pdf"):
        for source in sorted(source_dir.glob(suffix)):
            destination = destination_dir / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            yield destination


def _manifest(bundle_dir: Path) -> dict[str, Any]:
    records = []
    for path in sorted(candidate for candidate in bundle_dir.rglob("*") if candidate.is_file()):
        if path.name == "manifest.json":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"path": str(path.relative_to(bundle_dir)), "bytes": path.stat().st_size, "sha256": digest})
    return {
        "schema_version": 1,
        "artifact_scope": "public raw performance telemetry and derived analysis",
        "excluded": [
            "prompts, model responses, screenshots, and multimodal inputs",
            "MobileGym browser/session files",
            "cluster node names, IP addresses, absolute paths, model paths, GPU UUIDs, and call stacks",
        ],
        "files": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.experiment_dir
    output = args.output_dir
    if output.exists() and any(path.name != "README.md" for path in output.iterdir()):
        raise ValueError(f"output directory must be empty: {output}")
    sanitizer = Sanitizer()

    analysis_source = source / "analysis"
    analysis_output = output / "analysis"
    for path in sorted(analysis_source.glob("*.json")):
        _export_json(path, _compressed(analysis_output / path.name), sanitizer)
    list(_copy_figures(analysis_source, analysis_output))

    for category in ("timeline", "gpu_samples", "transfer_trace"):
        for path in sorted((source / category).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source / category)
            safe_name = _NODE_PATTERN.sub("node", relative.name)
            destination = output / "raw_performance" / category / relative.with_name(safe_name)
            if path.suffix == ".json":
                _export_json(path, _compressed(destination), sanitizer)
            elif path.suffix == ".jsonl":
                _export_jsonl(path, _compressed(destination), sanitizer)

    _export_rollout_latency(
        source / "rollout_result" / "train", output / "raw_performance" / "rollout_latency", sanitizer
    )
    for name in ("direct_report.json", "g5_full24_report.json", "clock_sync_audit.json"):
        path = source / name
        if path.exists():
            _export_json(path, _compressed(output / "run_metadata" / name), sanitizer)
    _write_json(output / "manifest.json", _manifest(output))


if __name__ == "__main__":
    main()
