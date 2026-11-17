"""Atom-conserving completion of reduced product geometries.

XYZ coordination graphs deliberately carry no inferred oxidation states. Ordinary bond
orders are proposed from distances subject to valence limits; metal contacts stay separate.
Chemical matching never uses the nearest XYZ row as an atom identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from rdkit import Chem, rdBase
from rdkit.Chem import rdDetermineBonds
from scipy.optimize import Bounds, LinearConstraint, least_squares, milp
from scipy.sparse import lil_matrix

from chemrefine import io
from chemrefine.config import MolecularConnectivity, ProductCore
from chemrefine.errors import ConfigError

# Main-group nonmetals, metalloids and noble gases are not metal anchors.
_NONMETALS = frozenset(
    {
        1,
        2,
        5,
        6,
        7,
        8,
        9,
        10,
        14,
        15,
        16,
        17,
        18,
        32,
        33,
        34,
        35,
        36,
        51,
        52,
        53,
        54,
        85,
        86,
        117,
        118,
    }
)
MATCH_SEARCH_BUDGET = 100_000
GEOMETRY_MATCH_LIMIT = 10_000
# Reference lengths (angstrom) for common single, double and triple organic bonds.
# These guide correspondence, not electronic energies or oxidation-state assignment.
_BOND_LENGTHS = {
    (6, 6): (1.54, 1.34, 1.20),
    (6, 7): (1.47, 1.29, 1.16),
    (6, 8): (1.43, 1.23),
    (7, 7): (1.45, 1.25, 1.10),
    (7, 8): (1.40, 1.21),
    (8, 8): (1.48, 1.21),
}
_MAX_VALENCE = {6: 4, 7: 3, 8: 2}


@dataclass(frozen=True)
class CoreMatch:
    """One chemically distinct product graph and its core-to-reactant row binding."""

    graph: Any
    indices: tuple[int, ...]
    edits: int


def is_metal(number: int) -> bool:
    """Whether an element can act as the inherited metal anchor."""
    return number not in _NONMETALS and number > 0


def distance_edges(atoms: Atoms) -> set[tuple[int, int]]:
    """Propose adjacency using a covalent-radius cutoff, without assigning bond orders."""
    distances = atoms.get_all_distances()
    radii = covalent_radii[atoms.numbers]
    return {
        (i, j)
        for i, j in combinations(range(len(atoms)), 2)
        if distances[i, j] < 1.25 * (radii[i] + radii[j])
    }


def _add_bond(graph: Any, first: int, second: int, kind: Any) -> None:
    """Represent a metal coordination edge in the donor-to-metal direction."""
    if (
        kind == Chem.BondType.DATIVE
        and is_metal(graph.GetAtomWithIdx(first).GetAtomicNum())
        and not is_metal(graph.GetAtomWithIdx(second).GetAtomicNum())
    ):
        first, second = second, first
    graph.AddBond(first, second, kind)


def _perceived_bonds(atoms: Atoms) -> list[tuple[int, int, int]]:
    """Infer ordinary bond orders jointly, without treating metal contacts as valence."""
    edges = sorted(distance_edges(atoms))
    orders = [
        0 if is_metal(int(atoms.numbers[a])) or is_metal(int(atoms.numbers[b])) else 1
        for a, b in edges
    ]
    valences = np.zeros(len(atoms))
    for (a, b), order in zip(edges, orders, strict=True):
        valences[[a, b]] += order
    choices = []
    for index, ((a, b), base) in enumerate(zip(edges, orders, strict=True)):
        first, second = sorted((int(atoms.numbers[a]), int(atoms.numbers[b])))
        lengths = _BOND_LENGTHS.get((first, second))
        if base == 0 or lengths is None:
            continue
        distance = atoms.get_distance(a, b)
        for order, length in enumerate(lengths[1:], 2):
            gain = (distance - lengths[0]) ** 2 - (distance - length) ** 2
            if gain > 1e-8:
                choices.append((index, a, b, order, gain))
    if choices:
        limits = np.concatenate(
            [
                np.ones(len(edges)),
                [
                    max(0, _MAX_VALENCE.get(int(z), int(v)) - v)
                    for z, v in zip(atoms.numbers, valences, strict=True)
                ],
            ]
        )
        matrix = np.zeros((len(limits), len(choices)))
        for column, (index, a, b, order, _) in enumerate(choices):
            matrix[index, column] = 1
            matrix[len(edges) + np.array([a, b]), column] = order - 1
        fit = milp(
            c=-np.array([gain for _, _, _, _, gain in choices]),
            integrality=np.ones(len(choices)),
            bounds=Bounds(0, 1),
            constraints=LinearConstraint(matrix, 0, limits),
            options={"time_limit": 5},
        )
        if not fit.success:
            raise ConfigError("automatic organic bond perception failed; check the input geometry")
        for chosen, (index, _, _, order, _) in zip(fit.x, choices, strict=True):
            if chosen > 0.5:
                orders[index] = order
    edge_orders = dict(zip(edges, orders, strict=True))
    for index, (a, b) in enumerate(edges):
        if orders[index] != 0:
            continue
        metal, carbon = (a, b) if is_metal(int(atoms.numbers[a])) else (b, a)
        if atoms.numbers[carbon] != 6:
            continue
        # A pi contact to both ends of a multiple bond does not consume carbon
        # valence. A lone metal-carbon attachment is sigma bonded and does.
        pi_contact = any(
            order > 1
            and carbon in edge
            and tuple(sorted((metal, edge[1] if edge[0] == carbon else edge[0]))) in edge_orders
            for edge, order in edge_orders.items()
        )
        if not pi_contact:
            orders[index] = 1
    return [(a + 1, b + 1, order) for (a, b), order in zip(edges, orders, strict=True)]


def read_structure(
    path: Path, connectivity: MolecularConnectivity, *, charge: int = 0, reduced: bool = False
) -> tuple[Any, Atoms]:
    """Read a finite single-frame XYZ and establish explicit or inferred connectivity."""
    try:
        frames = io.read_xyz_frames(path)
    except (OSError, ValueError, IndexError) as exc:
        raise ConfigError(f"{path}: cannot read XYZ: {exc}") from exc
    if len(frames) != 1 or not len(frames[0]):
        raise ConfigError(f"{path}: require exactly one nonempty XYZ frame")
    atoms = frames[0]
    if atoms.pbc.any() or not np.isfinite(atoms.positions).all():
        raise ConfigError(f"{path}: require finite nonperiodic molecular coordinates")
    metal = any(is_metal(int(z)) for z in atoms.numbers)
    if connectivity.bonds is None and not metal and not reduced:
        graph = Chem.MolFromXYZBlock(io.plain_xyz_text(atoms))
        try:
            rdDetermineBonds.DetermineBonds(graph, charge=charge, embedChiral=True)
        except (ValueError, RuntimeError) as exc:
            raise ConfigError(
                f"{path}: ambiguous XYZ bonding; check geometry and charge: {exc}"
            ) from exc
    else:
        graph = Chem.RWMol()
        for z in atoms.numbers:
            atom = Chem.Atom(int(z))
            atom.SetNoImplicit(True)
            graph.AddAtom(atom)
        if connectivity.bonds is None:
            radii = covalent_radii[atoms.numbers]
            for i, j in combinations(range(len(atoms)), 2):
                ratio = atoms.get_distance(i, j) / (radii[i] + radii[j])
                if ratio < 0.55 or 1.225 < ratio < 1.275:
                    raise ConfigError(
                        f"{path}: ambiguous distance at rows {i + 1},{j + 1}; "
                        "adjust the structure to distinguish a contact from a separated pair"
                    )
            bonds = _perceived_bonds(atoms)
        else:
            bonds = [(a, b, int(order)) for a, b, order in connectivity.bonds]
        for first, second, order in bonds:
            if not (1 <= first <= len(atoms) and 1 <= second <= len(atoms)) or first == second:
                raise ConfigError(f"{path}: connectivity rows must be distinct and in range")
            if graph.GetBondBetweenAtoms(first - 1, second - 1) is not None:
                raise ConfigError(f"{path}: duplicate connectivity bond")
            # Dative bonds preserve the distinction between coordination and covalent bonding.
            kind = {
                0: Chem.BondType.DATIVE,
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
            }[order]
            _add_bond(graph, first - 1, second - 1, kind)
        graph = graph.GetMol()
        graph.SetBoolProp("connectivity_only", connectivity.bonds is None)
    for row, value in connectivity.formal_charges.items():
        if not 1 <= row <= len(atoms):
            raise ConfigError(f"{path}: formal charge row outside XYZ")
        graph.GetAtomWithIdx(row - 1).SetFormalCharge(value)
        graph.GetAtomWithIdx(row - 1).SetBoolProp("formal_charge_explicit", True)
    graph.UpdatePropertyCache(strict=False)
    atoms.info["input_path"] = str(path)
    atoms.info["input_atom_indices"] = list(range(len(atoms)))
    return graph, atoms


def graph_key(graph: Any) -> str:
    """Canonical chemical signature, ignoring persistent atom identifiers."""
    return str(Chem.MolToSmiles(graph, canonical=True, allHsExplicit=True))


def adjacency_graph(graph: Any) -> Any:
    """An element/adjacency query without unestablished metal valences or bond orders."""
    result = Chem.RWMol()
    for atom in graph.GetAtoms():
        result.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    for bond in graph.GetBonds():
        result.AddBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return result.GetMol()


def _unlabelled(graph: Any) -> Any:
    """Copy a graph without its persistent identifiers for symmetry comparisons."""
    graph = Chem.Mol(graph)
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(0)
    return graph


def _edges(graph: Any) -> dict[tuple[int, int], Any]:
    """Index graph bonds in an orientation-independent form."""
    return {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): b.GetBondType()
        for b in graph.GetBonds()
    }


def _valid_valences(graph: Any) -> bool:
    """Reject impossible ordinary-atom valences without imposing metal valence rules."""
    organic = Chem.RWMol(graph)
    for index in range(graph.GetNumAtoms()):
        if is_metal(graph.GetAtomWithIdx(index).GetAtomicNum()):
            atom = organic.GetAtomWithIdx(index)
            atom.SetAtomicNum(0)
            atom.SetFormalCharge(0)
            for neighbor in graph.GetAtomWithIdx(index).GetNeighbors():
                # A coordination contact does not consume a lone-pair donor's
                # ordinary valence. Covalent carbon/hydride attachments still do.
                other = neighbor.GetIdx()
                if neighbor.GetAtomicNum() in {7, 8, 9, 15, 16, 17, 33, 34, 35, 53}:
                    organic.RemoveBond(index, other)
                    organic.AddBond(other, index, Chem.BondType.DATIVE)
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(organic.GetMol())
    except (ValueError, RuntimeError):
        return False
    return True


def _released(graph: Any, anchor: int, selectors: list[str]) -> set[tuple[int, int]]:
    """Resolve explicit whole-ligand selectors without deleting any ligand atoms."""
    cut = Chem.RWMol(graph)
    neighbors = [a.GetIdx() for a in graph.GetAtomWithIdx(anchor).GetNeighbors()]
    for index in neighbors:
        cut.RemoveBond(anchor, index)
    groups = Chem.GetMolFrags(cut.GetMol(), sanitizeFrags=False)
    removed: set[tuple[int, int]] = set()
    for selector in selectors:
        query = Chem.MolFromSmiles(selector)
        query = Chem.AddHs(query) if query is not None else Chem.MolFromSmarts(selector)
        if query is None:
            raise ConfigError(f"invalid dissociate SMARTS: {selector!r}")
        matches = {frozenset(match) for match in graph.GetSubstructMatches(query)}
        choices = [
            set(group) for group in groups if anchor not in group and frozenset(group) in matches
        ]
        if len(choices) != 1:
            raise ConfigError(f"dissociate {selector!r} must match exactly one whole ligand")
        removed.update(tuple(sorted((anchor, i))) for i in neighbors if i in choices[0])
    return removed


def match_core(
    reactant: Any,
    core: Any,
    *,
    anchor: int | None,
    hints: dict[int, int],
    dissociate: list[str],
    max_candidates: int,
    anchor_ids: set[int] | None = None,
) -> list[CoreMatch]:
    """Find element-conserving bindings, quotienting exact graph symmetries.

    Hints bind one-based core rows to persistent reactant identities. A bounded
    search refuses unresolved combinatorics instead of silently truncating chemistry.
    """
    count = core.GetNumAtoms()
    if count > reactant.GetNumAtoms() or (anchor is not None and not 0 <= anchor < count):
        raise ConfigError("core inventory or anchor is outside the supplied structures")
    if anchor is not None and not is_metal(core.GetAtomWithIdx(anchor).GetAtomicNum()):
        raise ConfigError("product core anchor must designate a metal placeholder")
    identities = {a.GetAtomMapNum(): a.GetIdx() for a in reactant.GetAtoms()}
    if any(not 1 <= row <= count or identity not in identities for row, identity in hints.items()):
        raise ConfigError("core match hints must bind core rows to existing atom identities")
    options = []
    for i, atom in enumerate(core.GetAtoms()):
        possible = [
            a.GetIdx()
            for a in reactant.GetAtoms()
            if (
                is_metal(a.GetAtomicNum())
                if i == anchor
                else a.GetAtomicNum() == atom.GetAtomicNum()
            )
        ]
        if i == anchor and anchor_ids is not None:
            possible = [
                j for j in possible if reactant.GetAtomWithIdx(j).GetAtomMapNum() in anchor_ids
            ]
        if i + 1 in hints:
            possible = [j for j in possible if j == identities[hints[i + 1]]]
        if not possible:
            raise ConfigError(f"core row {i + 1} has no compatible carried or new atom")
        options.append(possible)
    order = sorted(
        range(count), key=lambda i: (core.GetAtomWithIdx(i).GetAtomicNum() == 1, len(options[i]))
    )
    original = _edges(reactant)
    target = _edges(core)
    assignment: dict[int, int] = {}
    best = np.inf
    results: dict[str, CoreMatch] = {}
    visits = 0

    def search(depth: int) -> None:
        """Enumerate chemical completions; minimize edits only for complete XYZ pairs."""
        nonlocal best, visits
        visits += 1
        if visits > MATCH_SEARCH_BUDGET:
            raise ConfigError(
                "core matching search exceeded budget; narrow the core or supply match hints"
            )
        if depth == count:
            indices = tuple(assignment[i] for i in range(count))
            changed = dict(original)
            for a, b in combinations(indices, 2):
                changed.pop(tuple(sorted((a, b))), None)
            for (a, b), kind in target.items():
                changed[tuple(sorted((indices[a], indices[b])))] = kind
            if dissociate:
                if anchor is None:
                    raise ConfigError("dissociate requires a metal anchor")
                for edge in _released(reactant, indices[anchor], dissociate):
                    if edge in {tuple(sorted((indices[a], indices[b]))) for a, b in target}:
                        raise ConfigError("dissociated ligand is still bonded in the product core")
                    changed.pop(edge, None)
            edits = sum(
                original.get(edge) != changed.get(edge) for edge in original.keys() | changed.keys()
            )
            if anchor is not None and edits == 0:
                return
            if anchor is None and edits > best:
                return
            product = Chem.RWMol(reactant)
            for i, index in enumerate(indices):
                atom = core.GetAtomWithIdx(i)
                if anchor is None or atom.HasProp("formal_charge_explicit"):
                    product.GetAtomWithIdx(index).SetFormalCharge(atom.GetFormalCharge())
                if anchor is None:
                    product.GetAtomWithIdx(index).SetChiralTag(atom.GetChiralTag())
            for a, b in original:
                product.RemoveBond(a, b)
            for (a, b), kind in changed.items():
                _add_bond(product, a, b, kind)
            product = product.GetMol()
            product.UpdatePropertyCache(strict=False)
            if not _valid_valences(product):
                return
            # Product connectivity alone is insufficient: mark the core membership so
            # substitution sites with identical final graphs remain distinguishable.
            signature = _unlabelled(product)
            for i in indices:
                signature.GetAtomWithIdx(i).SetAtomMapNum(1)
            key = graph_key(signature)
            if anchor is None and edits < best:
                results.clear()
                best = edits
            # Equivalent products can differ only by unnecessary H scrambling.
            # Keep the correspondence that preserves the most existing bonds.
            existing = results.get(key)
            if existing is None or edits < existing.edits:
                results[key] = CoreMatch(product, indices, edits)
            return
        row = order[depth]
        seen = set()
        possible = options[row]
        # Complete endpoints normally preserve H parents. Prefer those bindings
        # before considering transfers, avoiding factorial permutations of ligand Hs.
        if anchor is None and core.GetAtomWithIdx(row).GetAtomicNum() == 1:
            parents = [
                assignment[a.GetIdx()]
                for a in core.GetAtomWithIdx(row).GetNeighbors()
                if a.GetIdx() in assignment
            ]
            preferred = [
                i
                for i in possible
                if i not in assignment.values()
                and any(reactant.GetBondBetweenAtoms(i, parent) is not None for parent in parents)
            ]
            if preferred:
                possible = preferred
        for index in possible:
            if index in assignment.values():
                continue
            assignment[row] = index
            marked = _unlabelled(reactant)
            for core_row, source_row in assignment.items():
                marked.GetAtomWithIdx(source_row).SetAtomMapNum(core_row + 1)
            key = graph_key(marked)
            if key not in seen:
                seen.add(key)
                search(depth + 1)
            del assignment[row]

    search(0)
    if not results:
        raise ConfigError("no atom-conserving product-core correspondence")
    if len(results) > max_candidates:
        raise ConfigError(
            f"core produced {len(results)} chemical candidates; limit is {max_candidates}; "
            "narrow matching or increase max_candidates"
        )
    return list(results.values())


def validate_connectivity(
    graph: Any,
    atoms: Atoms,
    *,
    label: str,
    allow_metal_changes: bool = False,
    allow_contact_changes: bool = False,
) -> None:
    """Check finite, element-ordered coordinates and every expected coordination edge."""
    if (
        list(atoms.numbers) != [a.GetAtomicNum() for a in graph.GetAtoms()]
        or atoms.pbc.any()
        or not np.isfinite(atoms.positions).all()
    ):
        raise ConfigError(f"{label}: geometry lost its atom identities or finite coordinates")
    changed = set(_edges(graph)) ^ distance_edges(atoms)
    if allow_metal_changes:
        changed = {
            (a, b)
            for a, b in changed
            if not (is_metal(int(atoms.numbers[a])) or is_metal(int(atoms.numbers[b])))
        }
    if changed and not allow_contact_changes:
        raise ConfigError(f"{label}: optimized connectivity or metal coordination changed")
    for a, b in combinations(range(len(atoms)), 2):
        if atoms.get_distance(a, b) < 0.55 * sum(covalent_radii[atoms.numbers[[a, b]]]):
            raise ConfigError(f"{label}: severe atomic overlap")


def _fragment(graph: Any, indices: list[int], colors: list[int]) -> Any:
    """Build a colored adjacency query for heavy atoms or hydrogen attachment groups."""
    result = Chem.RWMol()
    local = {index: i for i, index in enumerate(indices)}
    for index, color in zip(indices, colors, strict=True):
        atom = Chem.Atom(graph.GetAtomWithIdx(index).GetAtomicNum())
        atom.SetIsotope(color)
        result.AddAtom(atom)
    for a, b in _edges(graph):
        if a in local and b in local:
            result.AddBond(local[a], local[b], Chem.BondType.SINGLE)
    return result.GetMol()


def bind_geometry(graph: Any, atoms: Atoms, *, label: str) -> Atoms:
    """Bind heavy-atom adjacency first, then H groups, avoiding factorial H symmetries."""
    query = adjacency_graph(graph)
    observed = Chem.RWMol()
    for number in atoms.numbers:
        observed.AddAtom(Chem.Atom(int(number)))
    for a, b in distance_edges(atoms):
        observed.AddBond(a, b, Chem.BondType.SINGLE)
    if (
        query.GetNumAtoms() != observed.GetNumAtoms()
        or query.GetNumBonds() != observed.GetNumBonds()
    ):
        raise ConfigError(
            f"{label}: XYZ atom inventory or connectivity differs from declared graph"
        )
    observed = observed.GetMol()
    heavy = [
        [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1] for mol in (query, observed)
    ]
    h_counts = [
        [
            1 + sum(a.GetAtomicNum() == 1 for a in mol.GetAtomWithIdx(i).GetNeighbors())
            for i in indices
        ]
        for mol, indices in zip((query, observed), heavy, strict=True)
    ]
    qheavy = _fragment(query, heavy[0], h_counts[0])
    oheavy = _fragment(observed, heavy[1], h_counts[1])
    matches = oheavy.GetSubstructMatches(
        qheavy, uniquify=False, maxMatches=GEOMETRY_MATCH_LIMIT + 1
    )
    if len(matches) > GEOMETRY_MATCH_LIMIT:
        raise ConfigError(f"{label}: XYZ correspondence is missing or exceeds the symmetry limit")
    hydrogens = [
        [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1] for mol in (query, observed)
    ]
    for match in matches:
        binding = dict(zip(heavy[0], [heavy[1][i] for i in match], strict=True))
        groups = [
            [
                tuple(
                    sorted(
                        binding[a.GetIdx()] if side == 0 else a.GetIdx()
                        for a in mol.GetAtomWithIdx(i).GetNeighbors()
                        if a.GetAtomicNum() != 1
                    )
                )
                for i in indices
            ]
            for side, (mol, indices) in enumerate(zip((query, observed), hydrogens, strict=True))
        ]
        colors = {key: i + 1 for i, key in enumerate(sorted(set(groups[0]) | set(groups[1])))}
        hydrogen_graphs = [
            _fragment(mol, indices, [colors[key] for key in keys])
            for mol, indices, keys in zip((query, observed), hydrogens, groups, strict=True)
        ]
        hmatch = hydrogen_graphs[1].GetSubstructMatch(hydrogen_graphs[0])
        if len(hmatch) != len(hydrogens[0]):
            continue
        binding.update(zip(hydrogens[0], [hydrogens[1][i] for i in hmatch], strict=True))
        indices = [binding[i] for i in range(len(atoms))]
        result = atoms[indices]
        validate_connectivity(graph, result, label=label)
        result.info["input_path"] = label
        result.info["input_atom_indices"] = indices
        return result
    raise ConfigError(f"{label}: XYZ correspondence is missing")


def _fit(source: Any, target: Any) -> tuple[Any, Any]:
    """Return a proper rigid least-squares rotation and translation for row vectors."""
    first, second = np.mean(source, axis=0), np.mean(target, axis=0)
    u, _, vt = np.linalg.svd((source - first).T @ (target - second))
    sign = np.linalg.det(u @ vt)
    rotation = u @ np.diag([1, 1, sign]) @ vt
    return rotation, second - first @ rotation


def assemble(
    reactant: Atoms,
    graph: Any,
    product: Any,
    core_atoms: Atoms,
    indices: tuple[int, ...],
    settings: ProductCore,
    *,
    allow_metal_changes: bool = False,
    allow_contact_changes: bool = False,
) -> tuple[Atoms, list[Atoms], list[float]]:
    """Rigidly transfer omitted groups, sample orientations, and relax steric overlap.

    The core remains fixed during geometric relaxation. All distances within each
    inherited connected group are restrained to preserve chelation and conformation.
    This geometric objective is not an electronic energy or a force-field model.
    """
    validate_connectivity(
        graph,
        reactant,
        label="assembly reactant",
        allow_metal_changes=allow_metal_changes,
        allow_contact_changes=allow_contact_changes,
    )
    fixed = np.array(indices)
    free = np.array([i for i in range(len(reactant)) if i not in indices], dtype=int)
    rotation, translation = _fit(reactant.positions[fixed], core_atoms.positions)
    initial = reactant.positions @ rotation + translation
    initial[fixed] = core_atoms.positions
    cut = Chem.RWMol(product)
    for index in sorted(indices, reverse=True):
        cut.RemoveAtom(index)
    groups = [
        [int(free[i]) for i in group]
        for group in Chem.GetMolFrags(cut.GetMol(), sanitizeFrags=False)
    ]
    edges = _edges(product)
    restraints: dict[tuple[int, int], float] = {}
    for group in groups:
        for a, b in combinations(group, 2):
            restraints[a, b] = reactant.get_distance(a, b)
        boundary = [
            (a, b)
            for a, b in edges
            if (a in group) != (b in group) and (a in indices or b in indices)
        ]
        attached = sorted({a if a in indices else b for a, b in boundary})
        if attached:
            # One attachment alone fixes translation, not orientation. Use
            # conserved core neighbours as direction guides: an omitted H/R
            # on an alkyne must rotate with its unchanged C#C bond.
            guides = set(attached)
            for anchor in attached:
                for neighbor in graph.GetAtomWithIdx(anchor).GetNeighbors():
                    index = neighbor.GetIdx()
                    if index in indices and product.GetBondBetweenAtoms(anchor, index) is not None:
                        guides.add(index)
            guide_rows = sorted(guides)
            r, _ = _fit(reactant.positions[guide_rows], initial[guide_rows])
            # Guide bond lengths may change. Fit translation to the actual
            # attachment sites, keeping the inherited boundary bonds intact.
            t = initial[attached].mean(axis=0) - reactant.positions[attached].mean(axis=0) @ r
            initial[group] = reactant.positions[group] @ r + t
            for a, b in boundary:
                restraints[a, b] = reactant.get_distance(a, b)
        else:
            # Retain dissociated species explicitly, away from the reactive core.
            initial[group] += (
                core_atoms.positions.mean(axis=0)
                + np.array([5.0, 0.0, 0.0])
                - initial[group].mean(axis=0)
            )
    pairs = [
        (a, b)
        for a, b in combinations(range(len(reactant)), 2)
        if (a, b) not in edges and (a in free or b in free)
    ]
    radii = covalent_radii[reactant.numbers]
    # Clear the same adjacency threshold used by endpoint validation, with a
    # margin. Merely removing very short contacts can leave unintended new bonds.
    limits = np.array([1.35 * (radii[a] + radii[b]) for a, b in pairs])
    restraint_pairs = list(restraints)
    lengths = np.array(list(restraints.values()))
    restrained = np.array(restraint_pairs, dtype=int).reshape(-1, 2)
    nonbonded = np.array(pairs, dtype=int).reshape(-1, 2)
    free_columns = {int(index): 3 * column for column, index in enumerate(free)}
    count = len(restraint_pairs) + len(pairs)
    sparsity = lil_matrix((count + 3 * len(free), 3 * len(free)), dtype=int)
    for row, (a, b) in enumerate([*restraint_pairs, *pairs]):
        for atom in (a, b):
            if atom in free_columns:
                column = free_columns[atom]
                sparsity[row, column : column + 3] = 1
    for column in range(3 * len(free)):
        sparsity[count + column, column] = 1
    candidates, scores = [], []
    for trial in range(settings.placements if len(free) else 1):
        start = initial.copy()
        # Rotate each inherited group around its attachment axis, retaining its
        # internal geometry. Multiple attachments are handled together by relaxation.
        for ordinal, group in enumerate(groups):
            attached = sorted(
                {b if a in group else a for a, b in edges if (a in group) != (b in group)}
            )
            if len(attached) == 1:
                center = initial[attached[0]]
                axis = initial[group].mean(axis=0) - center
                norm = np.linalg.norm(axis)
                if norm > 1e-8:
                    axis /= norm
                    if trial:
                        z = 1 - 2 * (trial + 0.5) / (settings.placements + 1)
                        phi = (trial + ordinal) * 2.399963229728653
                        direction = np.array(
                            [np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z]
                        )
                        r, t = _fit(
                            np.array([center, center + axis]),
                            np.array([center, center + direction]),
                        )
                        start[group] = start[group] @ r + t
                    angle = trial * (ordinal + 1) * 2.399963229728653
                    vectors = start[group] - center
                    start[group] = (
                        vectors * np.cos(angle)
                        + np.cross(axis, vectors) * np.sin(angle)
                        + np.outer(vectors @ axis, axis) * (1 - np.cos(angle))
                    ) + center

        def residual(values: Any, start: Any = start) -> Any:
            """Penalize inherited distortion and short nonbonded contacts with fixed core."""
            positions = start.copy()
            positions[free] = values.reshape(-1, 3)
            bonded = np.linalg.norm(
                positions[restrained[:, 0]] - positions[restrained[:, 1]], axis=1
            )
            distances = np.linalg.norm(
                positions[nonbonded[:, 0]] - positions[nonbonded[:, 1]], axis=1
            )
            return np.concatenate(
                [
                    5 * (bonded - lengths),
                    10 * np.maximum(limits - distances, 0),
                    0.02 * (positions[free] - initial[free]).ravel(),
                ]
            )

        candidate = reactant.copy()
        candidate.info.clear()
        candidate.positions = start
        if len(free):
            result = least_squares(
                residual, start[free].ravel(), max_nfev=150, jac_sparsity=sparsity.tocsr()
            )
            candidate.positions[free] = result.x.reshape(-1, 3)
        if any(
            abs(candidate.get_distance(a, b) - length) > 0.15
            for (a, b), length in restraints.items()
        ):
            continue
        try:
            validate_connectivity(
                product,
                candidate,
                label="assembled product",
                allow_metal_changes=allow_metal_changes,
                allow_contact_changes=allow_contact_changes,
            )
        except ConfigError:
            continue
        score = float(np.sum(residual(candidate.positions[free].ravel()) ** 2))
        candidates.append(candidate)
        scores.append(score)
    if not candidates:
        raise ConfigError(
            "product-core assembly could not resolve clashes or preserve connectivity"
        )
    return candidates[int(np.argmin(scores))], candidates, scores
