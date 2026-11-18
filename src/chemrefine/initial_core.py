"""Assemble intact reactant fragments around a reduced coordination template."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from rdkit import Chem
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from chemrefine import product_core
from chemrefine.config import ProductCore
from chemrefine.errors import ConfigError


def plan(
    graph: Any, settings: ProductCore, *, max_candidates: int
) -> list[tuple[Any, dict[str, Any]]]:
    """Resolve coordination-only completions without modifying source chemistry."""
    core, atoms = product_core.read_structure(settings.input, settings.connectivity, reduced=True)
    matches = product_core.match_core(
        graph,
        core,
        anchor=settings.anchor - 1,
        hints=settings.match,
        dissociate=[],
        max_candidates=max_candidates,
        coordination_only=True,
    )
    groups = Chem.GetMolFrags(graph)
    fingerprint = hashlib.sha256(
        (settings.input.read_text() + settings.model_dump_json(exclude={"input"})).encode()
    ).hexdigest()
    results = []
    for match in matches:
        if any(not set(group).intersection(match.indices) for group in groups):
            raise ConfigError(
                "initial_core must anchor every reactant fragment; "
                "include spectator placement atoms"
            )
        results.append(
            (
                match.graph,
                {
                    "input": str(settings.input),
                    "template_sha256": fingerprint,
                    "core_indices": list(match.indices),
                    "core_atom_maps": [
                        graph.GetAtomWithIdx(i).GetAtomMapNum() for i in match.indices
                    ],
                    "core_coordinates": atoms.positions.tolist(),
                    "source_fragments": [list(group) for group in groups],
                    "coordination_edits": match.edits,
                },
            )
        )
    return results


def assemble(
    atoms: Atoms,
    source_graph: Any,
    graph: Any,
    mapping: dict[str, Any],
    settings: ProductCore,
) -> tuple[Atoms, list[Atoms], list[float]]:
    """Fit and sample rigid source fragments, minimizing clashes with the fixed template.

    Sources retain every internal distance. Template positions are fitting targets;
    at most 0.15 angstrom deviation accommodates independently prepared source bonds.
    The score is geometric, never an electronic energy.
    """
    product_core.validate_connectivity(source_graph, atoms, label="initial-core sources")
    indices = np.array(mapping["core_indices"], dtype=int)
    targets = np.array(mapping["core_coordinates"])
    groups = [np.array(group, dtype=int) for group in mapping["source_fragments"]]
    reference = atoms.positions.copy()
    centers, anchors = [], []
    for group in groups:
        rows = np.flatnonzero(np.isin(indices, group))
        fixed = indices[rows]
        rotation, translation = product_core.fit_rigid(reference[fixed], targets[rows])
        reference[group] = reference[group] @ rotation + translation
        centers.append(reference[fixed].mean(axis=0))
        anchors.append(fixed)
    membership = {int(i): n for n, group in enumerate(groups) for i in group}
    edges = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in graph.GetBonds()}
    pairs = np.array(
        [
            (a, b)
            for a, b in combinations(range(len(atoms)), 2)
            if membership[a] != membership[b] and (a, b) not in edges
        ],
        dtype=int,
    ).reshape(-1, 2)
    limits = 1.35 * covalent_radii[atoms.numbers[pairs]].sum(axis=1)
    rng = np.random.default_rng(42)
    candidates, scores = [], []
    for trial in range(settings.placements):
        start = reference.copy()
        for group, center, fixed in zip(groups, centers, anchors, strict=True):
            if trial:
                guide = reference[fixed] - center
                if np.linalg.matrix_rank(guide, tol=1e-7) == 0:
                    rotation = Rotation.random(random_state=rng)
                elif np.linalg.matrix_rank(guide, tol=1e-7) == 1:
                    axis = guide[np.argmax(np.linalg.norm(guide, axis=1))]
                    rotation = Rotation.from_rotvec(
                        axis / np.linalg.norm(axis) * rng.uniform(-np.pi, np.pi)
                    )
                else:
                    continue
                start[group] = rotation.apply(start[group] - center) + center

        def positions(values: Any, start: Any = start) -> Any:
            """Apply one rigid rotation and translation per source fragment."""
            result = start.copy()
            for group, center, transform in zip(
                groups, centers, values.reshape(-1, 6), strict=True
            ):
                result[group] = (
                    Rotation.from_rotvec(transform[:3]).apply(start[group] - center)
                    + center
                    + transform[3:]
                )
            return result

        def residual(values: Any, positions: Any = positions) -> Any:
            """Fit template anchors and repel unintended inter-fragment contacts."""
            result = positions(values)
            distances = np.linalg.norm(result[pairs[:, 0]] - result[pairs[:, 1]], axis=1)
            return np.concatenate(
                [
                    20 * (result[indices] - targets).ravel(),
                    10 * np.maximum(limits - distances, 0),
                    0.002 * values,
                ]
            )

        fit = least_squares(residual, np.zeros(6 * len(groups)), max_nfev=150)
        candidate = atoms.copy()
        candidate.info.clear()
        candidate.positions = positions(fit.x)
        if np.max(np.linalg.norm(candidate.positions[indices] - targets, axis=1)) > 0.15:
            continue
        try:
            product_core.validate_connectivity(graph, candidate, label="assembled initial core")
        except ConfigError:
            continue
        candidates.append(candidate)
        scores.append(float(np.sum(residual(fit.x) ** 2)))
    if not candidates:
        raise ConfigError(
            "initial-core assembly could not fit the template "
            "without clashes or changed connectivity"
        )
    return candidates[int(np.argmin(scores))], candidates, scores
