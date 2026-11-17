"""Elementary-reaction orchestration using the existing calculation pipelines.

Atom correspondence is supplied by the user: all hydrogens must be explicit and
atom order must agree. A guess is never a claim of a verified saddle point.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.mep import NEB

from chemrefine import io, pipeline, slurm
from chemrefine.config import Config, StepConfig, TSGuesserConfig
from chemrefine.engines import preflight_backends, preflight_steps
from chemrefine.errors import ConfigError, JobFailureError
from chemrefine.state import PipelineState
from chemrefine.step import RunPlan, StepMode, StepOutcome, preflight_batches


def branch_config(config: Config, name: str, steps: list[StepConfig], seed: Path) -> Config:
    """Inherit machine settings while isolating caches, reports and work directories."""
    # Empty preparation branches only bootstrap; they never enter pipeline.run.
    return config.model_copy(
        update={
            "reaction": None,
            "mechanism": None,
            "steps": steps,
            "input": seed,
            "output_dir": config.output_dir / name,
        }
    )


def branches(config: Config) -> dict[str, Config]:
    """The four calculation branches, also used by validation and dry runs."""
    reaction = config.reaction
    if reaction is None:
        raise ConfigError("this config has no reaction")
    return {
        "reactant": branch_config(
            config, "reactant", reaction.reactant.steps, reaction.reactant.input
        ),
        "product": branch_config(config, "product", reaction.product.steps, reaction.product.input),
        "scoring": branch_config(
            config,
            "scoring",
            reaction.ts_guesser.scoring_steps,
            config.output_dir / "guesser" / "candidates.xyz",
        ),
        "ts": branch_config(
            config, "ts", reaction.ts_steps, config.output_dir / "guesser" / "ts_guess.xyz"
        ),
    }


def _xyz(frames: list[Atoms]) -> str:
    """Serialize plain XYZ without retaining calculators or previous energies."""
    return "".join(io.plain_xyz_text(atoms) for atoms in frames)


def check_pair(reactant: Atoms, product: Atoms) -> None:
    """Refuse incompatible endpoints before any expensive calculation."""
    if not len(reactant) or reactant.get_chemical_symbols() != product.get_chemical_symbols():
        raise ConfigError("reaction endpoints must have the same atoms in the same mapped order")
    for atoms in (reactant, product):
        if atoms.pbc.any():
            raise ConfigError(
                "reaction guessing currently requires nonperiodic molecular endpoints"
            )
        if not np.isfinite(atoms.positions).all():
            raise ConfigError("reaction geometries must have finite coordinates")


def _one(state: PipelineState, name: str) -> Atoms:
    """Require a single unambiguous geometry for an endpoint."""
    if len(state.structures) != 1:
        raise ConfigError(
            f"reaction {name} must contain exactly one structure; use a sample filter "
            "to select one prepared conformer"
        )
    return state.structures[0].atoms


def interpolate(reactant: Atoms, product: Atoms, guesser: TSGuesserConfig) -> list[Atoms]:
    """Generate a path; its interior images still need energies to locate a TS guess."""
    check_pair(reactant, product)
    end = product.copy()
    # Remove arbitrary endpoint rotation/translation without changing atom mapping.
    from ase.build import minimize_rotation_and_translation

    minimize_rotation_and_translation(reactant, end)
    images = [reactant.copy() for _ in range(guesser.n_images - 1)] + [end]
    if guesser.method == "idpp":
        NEB(images, method="improvedtangent").interpolate(method="idpp", apply_constraint=False)
    elif guesser.method == "geodesic":
        try:
            from geodesic_interpolate.geodesic import Geodesic
            from geodesic_interpolate.interpolation import redistribute
        except ImportError as e:
            raise ConfigError("geodesic guessing needs pip install 'chemrefine[reaction]'") from e
        symbols = reactant.get_chemical_symbols()
        raw = redistribute(
            symbols, np.array([reactant.positions, end.positions]), guesser.n_images, tol=0.01
        )
        smoother = Geodesic(symbols, raw, 1.7, threshold=3, friction=0.01)
        if len(reactant) > 35:
            smoother.sweep(tol=0.002, max_iter=15, micro_iter=20)
        else:
            smoother.smooth(tol=0.002, max_iter=15)
        for atoms, positions in zip(images, smoother.path, strict=True):
            atoms.positions = positions
    else:
        raise ConfigError(f"{guesser.method} is not an interpolation method")
    for atoms in images:
        check_pair(reactant, atoms)
    return images


def _external(
    reactant: Atoms, product: Atoms, guesser: TSGuesserConfig, config: Config, directory: Path
) -> list[Atoms]:
    """Run a user-installed model wrapper without a shell or stale output reuse."""
    with (
        io.tree_write(directory),
        tempfile.TemporaryDirectory(prefix="model-", dir=directory) as temporary,
    ):
        work = Path(temporary).resolve()
        r, p, output = (work / name for name in ("reactant.xyz", "product.xyz", "candidates.xyz"))
        io.write_tree_file(r, _xyz([reactant]))
        io.write_tree_file(p, _xyz([product]))
        fields = {
            "reactant": str(r),
            "product": str(p),
            "output": str(output),
            "charge": str(config.charge),
            "multiplicity": str(config.multiplicity),
        }
        try:
            command = [arg.format_map(fields) for arg in guesser.command or []]
        except (KeyError, ValueError) as e:
            raise ConfigError(f"invalid TS guesser command placeholder: {e}") from e
        try:
            # Stream model logs to disk; a large model must not fill the driver's RAM.
            with (directory / "model.log").open("w", encoding="utf-8") as log:
                subprocess.run(  # noqa: S603 — explicit user-configured argv, no shell
                    command,
                    cwd=work,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=guesser.timeout_seconds,
                    check=True,
                )
        except (OSError, subprocess.SubprocessError) as e:
            raise JobFailureError(
                f"{guesser.method} failed: {e}; see {directory / 'model.log'}"
            ) from e
        try:
            candidates = io.read_xyz_frames(output)
        except (OSError, ValueError, IndexError) as e:
            raise JobFailureError(
                f"{guesser.method} did not write valid candidates to {{output}}: {e}"
            ) from e
        if not candidates:
            raise JobFailureError(f"{guesser.method} returned no TS candidates")
        for atoms in candidates:
            check_pair(reactant, atoms)
        # Linker scripts may attach model/checkpoint provenance. Preserve it before
        # the isolated command work directory is removed.
        metadata = output.with_suffix(output.suffix + ".json")
        if io.file_there(metadata):
            io.write_tree_file(directory / "model.json", metadata.read_bytes(), atomic=True)
        return candidates


def generate_candidates(config: Config, reactant: Atoms, product: Atoms, *, resume: bool) -> Path:
    """Cache guesses by endpoint coordinates and all guesser settings, verifying bytes."""
    reaction = config.reaction
    if reaction is None:
        raise ConfigError("this config has no reaction")
    guesser = reaction.ts_guesser
    directory = config.output_dir / "guesser"
    path, manifest = directory / "candidates.xyz", directory / "guess.json"
    key = hashlib.sha256(
        (
            _xyz([reactant, product])
            + guesser.model_dump_json()
            + f"{config.charge}/{config.multiplicity}/v1"
        ).encode()
    ).hexdigest()
    if resume:
        try:
            record = json.loads(manifest.read_text())
            if (
                record["key"] == key
                and record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            ):
                return path
        except (OSError, ValueError, KeyError, TypeError):
            pass
    # Invalidate before generation so failure cannot bless an older candidate file.
    io.remove_tree_file(manifest)
    io.remove_tree_file(directory / "model.json")
    io.write_tree_file(directory / "reactant.xyz", _xyz([reactant]))
    io.write_tree_file(directory / "product.xyz", _xyz([product]))
    if guesser.method in ("idpp", "geodesic"):
        images = interpolate(reactant, product, guesser)
        io.write_tree_file(directory / "path.xyz", _xyz(images))
        candidates = images[1:-1]
    else:
        candidates = _external(reactant, product, guesser, config, directory)
    payload = _xyz(candidates)
    io.write_tree_file(path, payload, atomic=True)
    io.write_tree_file(
        manifest,
        json.dumps(
            {
                "key": key,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "method": guesser.method,
            }
        ),
        atomic=True,
    )
    return path


def run(
    config: Config,
    plan: RunPlan | None = None,
    *,
    endpoint_check: Callable[[str, Atoms], None] | None = None,
) -> list[StepOutcome]:
    """Prepare both endpoints, generate guesses, optionally score, then run TS steps."""
    plan = plan or RunPlan()
    if (
        plan.overrides
        or plan.stop_after is not None
        or plan.default not in (StepMode.EXECUTE, StepMode.RESUME)
    ):
        raise ConfigError("reaction workflows support whole-workflow run and resume")
    stages = branches(config)
    outcomes: list[StepOutcome] = []
    with pipeline.run_lock(config.output_dir):
        # Fence every branch before any branch mutates, including a previous TS run.
        for branch in stages.values():
            pipeline.require_no_live_jobs(branch)
        io.remove_tree_file(config.output_dir / "reaction.json")
        steps = [step for branch in stages.values() for step in branch.steps]
        preflight_backends(steps)
        preflight_steps(steps, charge=config.charge, multiplicity=config.multiplicity)
        for branch in stages.values():
            preflight_batches(branch, branch.steps, local=slurm.dispatch_locally(config.dispatch))
        raw_r = _one(pipeline.bootstrap(stages["reactant"]), "reactant input")
        raw_p = _one(pipeline.bootstrap(stages["product"]), "product input")
        check_pair(raw_r, raw_p)
        endpoints = []
        for name in ("reactant", "product"):
            branch = stages[name]
            result = pipeline.run(branch, plan) if branch.steps else []
            outcomes.extend(result)
            state = result[-1].state if result else pipeline.bootstrap(branch)
            atoms = _one(state, name)
            if endpoint_check is not None:
                endpoint_check(name, atoms)
            endpoints.append(atoms)
        check_pair(*endpoints)
        path = generate_candidates(config, *endpoints, resume=plan.default == StepMode.RESUME)
        if stages["scoring"].steps:
            result = pipeline.run(stages["scoring"], plan)
            outcomes.extend(result)
            candidates = result[-1].state.structures
            if len(candidates) != len(io.read_xyz_frames(path)) or any(
                s.energy_hartree is None or not np.isfinite(s.energy_hartree) for s in candidates
            ):
                raise JobFailureError("every TS path image must have a finite single-point energy")
            selected = [
                max(
                    candidates,
                    key=lambda s: s.energy_hartree if s.energy_hartree is not None else -np.inf,
                ).atoms
            ]
        else:
            selected = io.read_xyz_frames(path)
        io.write_tree_file(
            config.output_dir / "guesser" / "ts_guess.xyz", _xyz(selected), atomic=True
        )
        # A report from an interrupted or failed older refinement must not imply success.
        io.remove_tree_file(config.output_dir / "reaction.json")
        result = pipeline.run(stages["ts"], plan)
        outcomes.extend(result)
        if not result or not result[-1].state.structures:
            raise JobFailureError("TS refinement produced no surviving candidates")
        io.write_tree_file(
            config.output_dir / "reaction.json",
            json.dumps(
                {
                    "status": "refinement_completed",
                    "verified_transition_state": False,
                    "method": (config.reaction.ts_guesser.method if config.reaction else "unknown"),
                    "ts_candidates": len(result[-1].state.structures),
                    "branches": {name: str(branch.output_dir) for name, branch in stages.items()},
                    "note": (
                        "Confirm one imaginary frequency and IRC connectivity "
                        "before assigning a mechanism."
                    ),
                },
                indent=2,
            ),
            atomic=True,
        )
    return outcomes
