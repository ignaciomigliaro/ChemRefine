#!/usr/bin/env python
"""Link mapped endpoint XYZ files to React-OT or FragmentFlow in its own environment.

This script deliberately does not import ChemRefine: the upstream environments use
Python 3.10 and incompatible packages named ``reactot``. Run it with that model's
Python interpreter. See docs/tutorials/elementary-reactions.md for installation.

APIs inspected at upstream revisions:
React-OT: 6dfccd0112cf6b504da80ad2516ff8e5ab8bade1
FragmentFlow: bf7a09fd61cec67828182d8442cd3cb351387878
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from ase.io import read

UPSTREAM_REVISIONS = {
    "react-ot": "6dfccd0112cf6b504da80ad2516ff8e5ab8bade1",
    "fragmentflow": "bf7a09fd61cec67828182d8442cd3cb351387878",
}


def parser() -> argparse.ArgumentParser:
    """The command contract shared with ChemRefine's external guesser."""
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", choices=UPSTREAM_REVISIONS, required=True)
    cli.add_argument(
        "--repo", type=Path, help="Upstream checkout; prepended to Python's import path"
    )
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--reactant", type=Path, required=True)
    cli.add_argument("--product", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--charge", type=int, default=0)
    cli.add_argument("--multiplicity", type=int, default=1)
    cli.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    cli.add_argument("--nfe", type=int, default=100)
    cli.add_argument("--solver", choices=["ddpm", "ode", "ei"], default="ode")
    cli.add_argument("--method", default="midpoint", help="torchdiffeq ODE integration method")
    cli.add_argument("--order", type=int, choices=[1, 2, 3], default=1)
    cli.add_argument("--diz", choices=["linear", "quad"], default="linear")
    cli.add_argument("--normalize", action="store_true", help="Upstream EI solver normalization")
    cli.add_argument("--atol", type=float, default=0.01)
    cli.add_argument("--rtol", type=float, default=0.01)
    cli.add_argument("--seed", type=int, default=42)
    core = cli.add_mutually_exclusive_group()
    core.add_argument(
        "--core-indices", help="FragmentFlow core: comma-separated zero-based XYZ indices"
    )
    core.add_argument("--full-system", action="store_true", help="Explicitly predict all atoms")
    cli.add_argument(
        "--epsilon", type=int, default=0, help="Upstream automatic core shell expansion"
    )
    cli.add_argument(
        "--idpp-images", type=int, default=11, help="FragmentFlow reconstruction images"
    )
    return cli


def read_endpoints(args: argparse.Namespace) -> tuple[Atoms, Atoms]:
    """Validate and align endpoints without permuting their supplied atom mapping."""
    if args.charge != 0 or args.multiplicity != 1:
        raise ValueError(
            "These linkers support neutral singlet reactions only: the upstream models "
            "do not condition inference on molecular charge or spin."
        )
    if args.nfe < 1 or args.epsilon < 0 or args.idpp_images < 3 or args.seed < 0:
        raise ValueError("Require nfe >= 1, epsilon >= 0, idpp-images >= 3, and seed >= 0")
    if not all(np.isfinite(t) and t > 0 for t in (args.atol, args.rtol)):
        raise ValueError("ODE tolerances must be finite and positive")
    if args.model != "fragmentflow" and (args.core_indices is not None or args.full_system):
        raise ValueError("Core selection options apply only to FragmentFlow")
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")
    protected = (args.reactant, args.product, args.checkpoint)
    if args.output.resolve() in {p.resolve() for p in protected}:
        raise ValueError("Output must differ from both endpoint files and the checkpoint")
    endpoints = []
    for path in (args.reactant, args.product):
        frames = cast(list[Atoms], read(str(path), index=":", format="xyz"))
        if len(frames) != 1 or not len(frames[0]):
            raise ValueError(f"Expected exactly one nonempty XYZ geometry: {path}")
        atoms = frames[0]
        if not np.isfinite(atoms.positions).all():
            raise ValueError(f"Nonfinite coordinates in {path}")
        if args.model == "react-ot" and set(atoms.numbers) - {1, 6, 7, 8, 9}:
            raise ValueError("The inspected model atom vocabulary supports H, C, N, O, F only")
        endpoints.append(atoms)
    reactant, product = endpoints
    if not np.array_equal(reactant.numbers, product.numbers):
        raise ValueError("Reactant and product must have identical atoms in the same mapped order")
    # Geometrical alignment only: never use a matcher that reassigns same-element atoms.
    reactant.positions -= reactant.positions.mean(axis=0)
    minimize_rotation_and_translation(reactant, product)
    product.positions -= product.positions.mean(axis=0)
    return reactant, product


def upstream_module(args: argparse.Namespace, module: str) -> Any:
    """Resolve a model's real module, with a useful error for a missing/wrong environment."""
    if args.repo is not None:
        repo = args.repo.resolve()
        relative = (
            "reactot/trainer/pl_trainer.py"
            if args.model == "react-ot"
            else "reactot/utils/inference_tools.py"
        )
        if not (repo / relative).is_file():
            raise ValueError(
                f"{repo} is not the expected {args.model} checkout ({relative} missing)"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ValueError(
            f"Cannot import {module}: {exc}. Run this script with {args.model}'s own "
            "environment and --repo pointing to its checkout. The two models must "
            "not share an environment because both install a package named reactot."
        ) from exc


def load_model(args: argparse.Namespace) -> Any:
    """Load the upstream checkpoint and expose its EnSB sampler, without evaluation metrics."""
    if args.model == "react-ot":
        module = upstream_module(args, "reactot.trainer.pl_trainer")
        model = (
            module.SBModule.load_from_checkpoint(
                checkpoint_path=str(args.checkpoint.resolve()),
                map_location=args.device,
            )
            .eval()
            .to(args.device)
        )
        ddpm = model.ddpm
    else:
        module = upstream_module(args, "reactot.utils.inference_tools")
        # The upstream loader builds the actual architecture and prefers EMA weights.
        # No dataset is read: the linker constructs a mapped single-reaction tensor batch.
        import torch

        # Upstream saves argparse.Namespace in plain checkpoints; allow that exact
        # metadata class under modern PyTorch's weights-only loader.
        with torch.serialization.safe_globals([argparse.Namespace, SimpleNamespace]):
            wrapper = module.load_pytorch_checkpoint(
                str(args.checkpoint.resolve()),
                dataset_path="",
                device=args.device,
                solver=args.solver,
                method=args.method,
                atol=args.atol,
                rtol=args.rtol,
            )
        ddpm = wrapper.ddpm
    if ddpm.mapping != "R+P->TS" or ddpm.mapping_initial != "RP" or ddpm.idx != 1:
        raise ValueError("Checkpoint must predict TS from R+P with midpoint initialization (RP)")
    if args.nfe >= ddpm.T:
        raise ValueError(f"nfe must be less than the checkpoint's {ddpm.T} diffusion timesteps")
    # Set the complete option contract: FragmentFlow's loader omits EI order/diz,
    # while React-OT's ODE implementation also reads normalize.
    ddpm.opt = SimpleNamespace(
        solver=args.solver,
        method=args.method,
        order=args.order,
        diz=args.diz,
        normalize=args.normalize,
        atol=args.atol,
        rtol=args.rtol,
    )
    ddpm.eval()
    return ddpm


def representations(reactant: Atoms, product: Atoms, device: str) -> list[dict[str, Any]]:
    """Build the upstream [R, TS slot, P] tensor layout; the TS slot is a midpoint placeholder."""
    import torch

    mapping = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
    numbers = reactant.numbers
    if set(numbers) - set(mapping):
        raise ValueError(
            "Predicted atoms/core must use H, C, N, O, F; keep other elements in substituents"
        )
    hot = np.eye(5, dtype=np.int64)[[mapping[int(z)] for z in numbers]]
    positions = [
        reactant.positions,
        (reactant.positions + product.positions) / 2,
        product.positions,
    ]
    return [
        {
            "size": torch.tensor([len(numbers)], dtype=torch.int64, device=device),
            "pos": torch.tensor(x - x.mean(axis=0), dtype=torch.float32, device=device),
            "one_hot": torch.tensor(hot, dtype=torch.int64, device=device),
            # Upstream's 'charge' feature is the atomic number, not molecular charge.
            "charge": torch.tensor(numbers[:, None], dtype=torch.int64, device=device),
            "mask": torch.zeros(len(numbers), dtype=torch.int64, device=device),
        }
        for x in positions
    ]


def sample_ts(ddpm: Any, reactant: Atoms, product: Atoms, args: argparse.Namespace) -> np.ndarray:
    """Run the actual upstream sampler without requiring a known TS or computing a fake RMSD."""
    import torch

    batch = representations(reactant, product, args.device)
    conditions = torch.zeros((1, 1), device=args.device)
    with torch.no_grad():
        _, initial, *_ = ddpm.sample_batch(
            batch, conditions, return_timesteps=False, training=False
        )
        trajectory, _ = ddpm.sample(initial, batch, conditions, nfe=args.nfe, ot_ode=True)
    positions = np.asarray(trajectory[:, 0, :].detach().cpu().numpy(), dtype=float)
    if positions.shape != (len(reactant), 3) or not np.isfinite(positions).all():
        raise ValueError("Model returned invalid TS coordinates")
    return positions


def core_indices(args: argparse.Namespace, reactant: Atoms, product: Atoms) -> list[int]:
    """Select an explicit core, all atoms, or the authors' automatic fragmentation routine."""
    if args.core_indices is not None:
        try:
            indices = [int(i.strip()) for i in args.core_indices.split(",")]
        except ValueError as exc:
            raise ValueError("core-indices must be comma-separated integers") from exc
        if len(set(indices)) != len(indices) or any(i < 0 or i >= len(reactant) for i in indices):
            raise ValueError("Core indices must be unique and within the zero-based XYZ atom range")
        return sorted(indices)
    if args.full_system:
        return list(range(len(reactant)))
    # Reuse upstream core detection, including its Murcko/epsilon and hydrogen rules.
    # This optional path requires localmapper, which the upstream base env omits.
    try:
        module = upstream_module(args, "dataset_utils.fragment_dataset.fragment_t1x")
        mapper_module = upstream_module(args, "localmapper")
    except ValueError as exc:
        raise ValueError(
            f"{exc} Automatic core selection requires upstream's optional localmapper "
            "dependencies. Supply --core-indices to bypass atom mapping, or explicitly "
            "use --full-system."
        ) from exc
    result = module.process_single_reaction(
        reactant.numbers.tolist(),
        reactant.positions,
        product.numbers.tolist(),
        product.positions,
        reactant.numbers.tolist(),
        (reactant.positions + product.positions) / 2,
        mapper_module.localmapper(device="cpu"),
        [args.epsilon],
    )
    if result["error"]:
        raise ValueError(f"Automatic FragmentFlow core selection failed: {result['error']}")
    selected = result["epsilon_results"].get(args.epsilon)
    if selected is None:
        print(
            "Upstream fragmentation selected no smaller core; predicting the full system.",
            file=sys.stderr,
        )
        return list(range(len(reactant)))
    indices = sorted(selected["backbone_indices"])
    if not indices or any(i < 0 or i >= len(reactant) for i in indices):
        raise ValueError("Upstream fragmentation returned an invalid core")
    return indices


def fragmentflow_ts(
    ddpm: Any,
    reactant: Atoms,
    product: Atoms,
    args: argparse.Namespace,
) -> tuple[Atoms, list[int]]:
    """Predict a reactive core and call upstream constrained-IDPP substituent reconstruction."""
    indices = core_indices(args, reactant, product)
    positions = sample_ts(ddpm, reactant[indices], product[indices], args)
    if len(indices) == len(reactant):
        return Atoms(numbers=reactant.numbers, positions=positions), indices
    module = upstream_module(args, "post_processing.attach_substituents.complete_from_xyz_cores")
    numbers, positions, *_ = module.complete_fragmented_core_from_xyz(
        positions,
        reactant.numbers,
        reactant.positions,
        product.numbers,
        product.positions,
        indices,
        num_idpp_images=args.idpp_images,
    )
    if not np.array_equal(numbers, reactant.numbers):
        raise ValueError("FragmentFlow reconstruction changed the atom mapping")
    positions = np.asarray(positions)
    if positions.shape != (len(reactant), 3) or not np.isfinite(positions).all():
        raise ValueError("FragmentFlow reconstruction returned invalid coordinates")
    return Atoms(numbers=numbers, positions=positions), indices


def atomic_text(path: Path, text: str) -> None:
    """Publish a complete output without exposing a partially written XYZ to another reader."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> None:
    """Load a model once, predict the TS and publish a mapped XYZ plus provenance."""
    reactant, product = read_endpoints(args)
    import torch

    if args.device != "cpu" and not args.device.startswith("cuda"):
        raise ValueError("Supported devices are cpu and cuda[:index]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this model environment")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ddpm = load_model(args)
    if args.model == "fragmentflow":
        atoms, indices = fragmentflow_ts(ddpm, reactant, product, args)
    else:
        atoms = Atoms(numbers=reactant.numbers, positions=sample_ts(ddpm, reactant, product, args))
        indices = list(range(len(atoms)))
    rows = [str(len(atoms)), f"{args.model} TS guess; not a verified saddle point"]
    rows.extend(
        f"{s} {x:.16g} {y:.16g} {z:.16g}"
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.positions, strict=True)
    )
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    checkpoint_digest = digest.hexdigest()
    metadata = {
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "api_reference_revision": UPSTREAM_REVISIONS[args.model],
        "core_indices": indices,
        "nfe": args.nfe,
        "solver": args.solver,
        "method": args.method,
        "order": args.order,
        "diz": args.diz,
        "normalize": args.normalize,
        "atol": args.atol,
        "rtol": args.rtol,
        "epsilon": args.epsilon,
        "idpp_images": args.idpp_images,
        "repo": str(args.repo.resolve()) if args.repo else None,
        "seed": args.seed,
        "device": args.device,
        "charge": args.charge,
        "multiplicity": args.multiplicity,
        "verified_transition_state": False,
    }
    atomic_text(
        args.output.with_suffix(args.output.suffix + ".json"), json.dumps(metadata, indent=2)
    )
    atomic_text(args.output, "\n".join(rows) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Report integration/dependency errors with a nonzero status for the workflow driver."""
    args = parser().parse_args(argv)
    try:
        run(args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"TS linker failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
