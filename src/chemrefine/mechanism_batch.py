"""Expand XYZ directories into isolated mechanism runs with a shared resource budget."""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from math import prod
from pathlib import Path
from typing import cast

from chemrefine import io, pipeline
from chemrefine.config import Config, MechanismConfig
from chemrefine.errors import ConfigError, JobFailureError
from chemrefine.step import RunPlan, StepMode


@dataclass(frozen=True)
class Variant:
    """One deterministic directory combination and its concrete mechanism config."""

    name: str
    config: Config
    sources: tuple[Path, ...]


def enabled(config: Config) -> bool:
    """Whether a mechanism requests independent reactant alternatives from a directory."""
    return config.mechanism is not None and any(
        source.input_dir is not None
        for stage in config.mechanism.stages
        for source in stage.reactant.sources
    )


def expand(config: Config) -> list[Variant]:
    """Enumerate sorted XYZ files, refusing empty directories or excessive combinations."""
    settings = config.mechanism
    if settings is None:
        raise ConfigError("directory variants require a mechanism")
    slots = []
    for i, stage in enumerate(settings.stages):
        for j, source in enumerate(stage.reactant.sources):
            if source.input_dir is not None:
                files = [
                    path
                    for path, directory in io.tree_entries(source.input_dir)
                    if not directory and path.suffix.lower() == ".xyz"
                ]
                if not files:
                    raise ConfigError(f"{source.input_dir}: no XYZ reactant files")
                stems = [path.stem for path in files]
                if len(set(stems)) != len(stems) or any(
                    re.fullmatch(r"[A-Za-z0-9_-]+", stem) is None for stem in stems
                ):
                    raise ConfigError(
                        "XYZ variant filenames must have unique safe stems: letters, digits, _ or -"
                    )
                slots.append((i, j, files))
    if not slots:
        raise ConfigError("no reactant input_dir was provided")
    count = prod(len(files) for _, _, files in slots)
    if count > settings.max_variants:
        raise ConfigError(
            f"directory expansion produces {count} variants; limit is {settings.max_variants}"
        )
    workers = min(settings.parallel_runs, config.max_cores, count)
    if workers > 1 and config.max_gpus is None:
        raise ConfigError("parallel directory runs require max_gpus explicitly (0 for CPU runs)")
    result = []
    for paths in product(*(files for _, _, files in slots)):
        name = (
            paths[0].stem
            if len(slots) == 1
            else "__".join(f"s{i + 1}_{path.stem}" for i, path in enumerate(paths))
        )
        stages = list(settings.stages)
        for (i, j, _), path in zip(slots, paths, strict=True):
            stage = stages[i]
            sources = list(stage.reactant.sources)
            sources[j] = sources[j].model_copy(update={"input": path, "input_dir": None})
            stages[i] = stage.model_copy(
                update={"reactant": stage.reactant.model_copy(update={"sources": sources})}
            )
        override = settings.variant_overrides.get(name)
        exceptions = override.model_dump(exclude_none=True) if override is not None else {}
        concrete = config.model_copy(
            update={
                "mechanism": settings.model_copy(
                    update={"stages": stages, "variant_overrides": {}, **exceptions}
                ),
                "output_dir": config.output_dir / "variants" / name,
                "max_cores": config.max_cores // workers,
                "max_gpus": None if config.max_gpus is None else config.max_gpus // workers,
            }
        )
        result.append(Variant(name, concrete, tuple(paths)))
    if len({variant.name for variant in result}) != len(result):
        raise ConfigError("directory combinations produce colliding names; rename the XYZ files")
    unknown = set(settings.variant_overrides) - {variant.name for variant in result}
    if unknown:
        raise ConfigError(f"variant_overrides names unknown variants: {', '.join(sorted(unknown))}")
    return result


def _execute(variant: Variant, mode: StepMode) -> dict[str, str | int]:
    """Use a separate driver process so local job registries and model runtimes are isolated."""
    directory = variant.config.output_dir
    config_path = directory / "variant.yaml"
    io.write_tree_file(config_path, variant.config.model_dump_json(indent=2), atomic=True)
    command = [
        sys.executable,
        "-m",
        "chemrefine",
        "resume" if mode == StepMode.RESUME else "run",
        str(config_path),
    ]
    with (
        io.tree_write(directory / "driver.log"),
        (directory / "driver.log").open("w", encoding="utf-8") as log,
    ):
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)  # noqa: S603
    return {
        "name": variant.name,
        "exit_code": process.returncode,
        "status": "refinement_completed" if process.returncode == 0 else "failed",
    }


def _publish(config: Config, variants: list[Variant], records: list[dict[str, str | int]]) -> None:
    """Keep failures visible beside successful variants and link every native run tree."""
    io.write_tree_file(
        config.output_dir / "batch.json",
        json.dumps(
            {
                "status": "running" if any(r["exit_code"] == -1 for r in records) else "finished",
                "verified_transition_states": False,
                "variants": [
                    {
                        **record,
                        "directory": f"variants/{variant.name}",
                        "sources": [
                            {
                                "path": str(path),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            }
                            for path in variant.sources
                        ],
                    }
                    for variant, record in zip(variants, records, strict=True)
                ],
            },
            indent=2,
        ),
        atomic=True,
    )
    rows = []
    for record in records:
        name = str(record["name"])
        links = (
            f'<a href="variants/{name}/variant.yaml">Input</a> · '
            f'<a href="variants/{name}/driver.log">Log</a>'
        )
        if record["exit_code"] == 0:
            links += (
                f' · <a href="variants/{name}/final/index.html">'
                "Geometries, outputs and frequencies</a>"
            )
        rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{record['status']}</td><td>{links}</td></tr>"
        )
    io.write_tree_file(
        config.output_dir / "index.html",
        '<!doctype html><meta charset="utf-8"><title>Mechanism variants</title>'
        "<h1>Mechanism variants</h1><p>Refinement completion does not establish TS identity; "
        "review frequency and IRC evidence for each variant.</p><table>"
        "<tr><th>Variant</th><th>Status</th><th>Files</th></tr>" + "".join(rows) + "</table>",
        atomic=True,
    )


def run(config: Config, plan: RunPlan) -> None:
    """Run every alternative despite individual failures, within the total CPU/GPU limits."""
    from chemrefine.mechanism import plan as plan_mechanism

    variants = expand(config)
    # Validate every candidate before any worker submits an expensive job.
    for variant in variants:
        plan_mechanism(variant.config)
    settings = cast(MechanismConfig, config.mechanism)
    workers = min(settings.parallel_runs, config.max_cores, len(variants))
    with pipeline.run_lock(config.output_dir):
        records: list[dict[str, str | int]] = [
            {"name": v.name, "exit_code": -1, "status": "queued_or_running"} for v in variants
        ]
        _publish(config, variants, records)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_execute, variant, plan.default): i
                for i, variant in enumerate(variants)
            }
            for future in as_completed(futures):
                index = futures[future]
                records[index] = future.result()
                _publish(config, variants, records)
        if any(record["exit_code"] != 0 for record in records):
            raise JobFailureError(
                f"one or more mechanism variants failed; see {config.output_dir / 'index.html'}"
            )
