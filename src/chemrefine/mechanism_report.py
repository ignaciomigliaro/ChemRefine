"""Portable final geometries and their matching calculation and frequency evidence."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ase import Atoms

from chemrefine import cache, io
from chemrefine.state import Structure


@dataclass(frozen=True)
class Calculation:
    """A selected stationary point and its manifest-identified native output, if available."""

    structure: Structure
    output: Path | None


def calculation(step_dir: Path, structure: Structure) -> Calculation:
    """Find the canonical job output, including an ensemble child's explicit parent job."""
    manifest = cache.load_manifest(step_dir)
    jobs = {sid: output for _, output, sid in manifest.files} if manifest else {}
    output = jobs.get(structure.id, jobs.get(structure.parent_id or ""))
    return Calculation(structure, output)


@dataclass(frozen=True)
class FinalStage:
    """Optimized endpoints and every surviving final TS candidate, in stage order."""

    name: str
    reaction: str
    reactant: Atoms
    product: Atoms
    transition_states: tuple[Structure, ...]
    atom_maps: list[int]
    charge: int
    multiplicity: int
    ts_source: str
    source_dir: str | None = None
    calculations: dict[str, Calculation] = field(default_factory=dict)
    force: bool = False
    force_endpoint_contacts: bool = False


def _clear_calculations(directory: Path) -> None:
    """Invalidate managed evidence copies while retaining notes and original job files."""
    for role, is_dir in io.tree_entries(directory):
        if is_dir and re.fullmatch(r"reactant|product|ts_\d{3,}", role.name):
            for path, nested in io.tree_entries(role):
                if not nested and (
                    path.name in {"geometry.xyz", "frequencies.json"}
                    or re.fullmatch(r"calculation(?:\.[A-Za-z0-9_-]+)?", path.name)
                ):
                    io.remove_tree_file(path)


def begin(output_dir: Path) -> None:
    """Invalidate an earlier collection before any calculations mutate its source tree."""
    directory = output_dir / "final"
    for path, is_dir in io.tree_entries(directory):
        if is_dir and re.fullmatch(r"\d{3,}_[A-Za-z0-9_-]+", path.name):
            _clear_calculations(path)
        if path.name in {
            "index.html",
            "manifest.json",
            "mechanism.xyz",
            "README.txt",
        } or re.fullmatch(r"\d{3,}_[A-Za-z0-9_-]+_(reactant|product|ts_\d{3,})\.xyz", path.name):
            io.remove_tree_file(path)
    io.write_tree_file(
        directory / "manifest.json",
        json.dumps({"status": "incomplete", "geometries": []}, indent=2),
        atomic=True,
    )
    io.write_tree_file(
        directory / "index.html",
        "<!doctype html><meta charset='utf-8'><title>Mechanism incomplete</title>"
        "<h1>Mechanism incomplete</h1><p>The run is in progress or stopped. "
        "Final geometries are published only after every stage completes.</p>",
        atomic=True,
    )


def publish(output_dir: Path, stages: list[FinalStage]) -> None:
    """Export labeled XYZ copies and a self-contained HTML index without running calculations."""
    directory = output_dir / "final"
    records = []
    frames = []
    cards = []
    for index, stage in enumerate(stages, 1):
        links = []
        points: list[tuple[str, Atoms, Structure | None]] = [("reactant", stage.reactant, None)]
        points.extend(
            (f"ts_{i:03d}", structure.atoms, structure)
            for i, structure in enumerate(stage.transition_states, 1)
        )
        points.append(("product", stage.product, None))
        for role, atoms, structure in points:
            filename = f"{index:03d}_{stage.name}_{role}.xyz"
            label = f"{index}. {stage.name} / {role}"
            xyz = io.plain_xyz_text(atoms, comment=label)
            io.write_tree_file(directory / filename, xyz, atomic=True)
            bundle = Path(f"{index:03d}_{stage.name}") / role
            io.write_tree_file(directory / bundle / "geometry.xyz", xyz, atomic=True)
            evidence = stage.calculations.get(role)
            selected = evidence.structure if evidence is not None else structure
            native = None
            if (
                evidence is not None
                and evidence.output is not None
                and not io.absent(evidence.output)
            ):
                native = str(bundle / f"calculation{evidence.output.suffix}")
                io.write_tree_file(directory / native, evidence.output.read_bytes(), atomic=True)
            frequencies = {
                "frequencies_cm_inverse": selected.frequencies if selected is not None else None,
                "imaginary_frequencies_cm_inverse": selected.imaginary_freqs
                if selected is not None
                else None,
                "energy_hartree": selected.energy_hartree if selected is not None else None,
                "calculation_output": Path(native).name if native is not None else None,
            }
            frequency_file = str(bundle / "frequencies.json")
            io.write_tree_file(
                directory / frequency_file, json.dumps(frequencies, indent=2), atomic=True
            )
            frames.append(xyz)
            record = {
                "stage": stage.name,
                "role": role,
                "force_metal_coordination": stage.force,
                "force_endpoint_contacts": stage.force_endpoint_contacts,
                "file": filename,
                "geometry": str(bundle / "geometry.xyz"),
                "output": native,
                "frequency_file": frequency_file,
                "frame": len(frames) - 1,
                "atom_maps": stage.atom_maps,
                "charge": stage.charge,
                "multiplicity": stage.multiplicity,
                "source": stage.ts_source
                if structure is not None
                else f"{stage.source_dir or stage.name}/endpoints/{role}.xyz",
                "structure_id": structure.id if structure is not None else None,
                "energy_hartree": selected.energy_hartree if selected is not None else None,
                "imaginary_frequencies_cm_inverse": selected.imaginary_freqs
                if selected is not None
                else None,
            }
            records.append(record)
            links.append(
                f'<a href="{html.escape(filename, quote=True)}" download>'
                f"{html.escape(role.replace('_', ' ').upper())} · {len(atoms)} atoms</a>"
            )
            links.append(f'<a href="{html.escape(frequency_file, quote=True)}">Frequencies</a>')
            if native is not None:
                links.append(f'<a href="{html.escape(native, quote=True)}">Calculation output</a>')
        cards.append(
            f"<section><h2>{index}. {html.escape(stage.name)}</h2>"
            f"<p class='reaction'>{html.escape(stage.reaction)}</p>"
            + (
                "<p>force_endpoint_contacts=true: endpoint contact changes, including "
                "ordinary contacts, were permitted. This is exploratory; review "
                "coordination_overrides.json and independently validate endpoint/TS identity.</p>"
                if stage.force_endpoint_contacts
                else ""
            )
            + (
                "<p>force=true: metal-contact changes were permitted. Review the stage's "
                "coordination_overrides.json; nominal product graphs are retained and "
                "do not certify the optimized coordination.</p>"
                if stage.force
                else ""
            )
            + f"<nav>{''.join(links)}</nav></section>"
        )
    io.write_tree_file(directory / "mechanism.xyz", "".join(frames), atomic=True)
    io.write_tree_file(
        directory / "README.txt",
        "Open index.html for the final mechanism geometry index.\n"
        "Each numbered stage has optimized reactant, final TS candidate(s), and product XYZs.\n"
        "Each stage/role directory also contains geometry.xyz, frequencies.json and the matching\n"
        "native calculation output when available. Null frequencies mean no frequency data;\n"
        "an empty imaginary-frequency table means none were found. Missing outputs are null\n"
        "in manifest.json; no other candidate's output is substituted.\n"
        "mechanism.xyz contains those frames in the same order, with labeled comment lines.\n"
        "Stages may contain different numbers of atoms; these frames are stationary points, "
        "not a continuous trajectory.\n"
        "manifest.json records frame order, persistent atom identities, charge, spin, "
        "and source paths relative to output_dir. TS IDs identify frames in source ensembles.\n"
        "The TS geometries are final survivors of the refinement branch, not guesser images.\n"
        "Completion does not verify a transition state: review frequencies and IRC connectivity.\n"
        "This folder is managed and regenerated on run/resume. Other calculation files remain "
        "in their original locations.\n",
        atomic=True,
    )
    io.write_tree_file(
        directory / "index.html",
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Final reaction mechanism · ChemRefine</title><style>"
        "body{font:16px system-ui,sans-serif;max-width:1000px;margin:48px auto;"
        "padding:0 24px;color:#17243b;background:#f4f7fb}"
        "section{background:white;padding:22px;margin:22px 0;border:1px solid #dbe3ef;"
        "border-radius:14px}h1{font-size:32px}h2{font-size:21px}"
        ".reaction{font-family:monospace;overflow-wrap:anywhere}"
        "nav{display:flex;gap:12px;flex-wrap:wrap}a{color:#174fa0}"
        "section a{padding:12px;border:1px solid #b7cbed;border-radius:8px;text-decoration:none}"
        "aside{padding:16px;background:#fff3d6;border-radius:8px}</style>"
        "<h1>Final reaction mechanism</h1><p>Optimized endpoints and final TS candidates, "
        "collected in reaction order.</p><nav><a href='mechanism.xyz' download>"
        "All geometries (XYZ)</a><a href='manifest.json'>Geometry manifest</a>"
        "<a href='README.txt'>About these files</a></nav>"
        "<aside>Refinement completed. TS identity still requires frequency and IRC checks. "
        "Stages can contain different atom counts; the combined XYZ is not a trajectory.</aside>"
        + "".join(cards)
        + "</html>",
        atomic=True,
    )
    io.write_tree_file(
        directory / "manifest.json",
        json.dumps(
            {
                "status": "refinement_completed",
                "verified_transition_state": False,
                "geometries": records,
            },
            indent=2,
        ),
        atomic=True,
    )
