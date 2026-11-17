"""Automatic reaction mapping and persistent atom-identity allocation.

A reaction mapper proposes heavy-atom correspondence. It never owns persistent
identifiers: those come from carried molecules or fresh reagent allocations.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter, defaultdict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
from rdkit import Chem

from chemrefine.errors import ConfigError


class IdAllocator:
    """Allocate identities once per mechanism; never recycle an earlier reagent's atoms."""

    def __init__(self) -> None:
        self.used: set[int] = set()
        self.next_id = 1

    def fresh(self) -> int:
        """Return an unused positive identifier."""
        while self.next_id in self.used:
            self.next_id += 1
        result = self.next_id
        if result > 2_147_483_647:
            raise ConfigError("mechanism atom identifiers exceed the supported integer range")
        self.used.add(result)
        self.next_id += 1
        return result

    def introduce(self, graph: Any, *, retain_unused_labels: bool) -> tuple[Any, dict[int, int]]:
        """Give a new reagent fresh identities, with a local alias table for legacy labels."""
        graph = Chem.Mol(graph)
        aliases = {}
        # Reserve unused supplied labels before allocating replacements, so an
        # old label cannot accidentally consume a later atom's requested label.
        retained = {
            a.GetAtomMapNum()
            for a in graph.GetAtoms()
            if retain_unused_labels and a.GetAtomMapNum() and a.GetAtomMapNum() not in self.used
        }
        self.used.update(retained)
        for atom in graph.GetAtoms():
            old = atom.GetAtomMapNum()
            new = old if old in retained else self.fresh()
            atom.SetAtomMapNum(new)
            if old:
                aliases[old] = new
        return graph, aliases


def completely_labelled(smiles: str) -> bool:
    """Whether a declaration provides a complete legacy map on its explicit atoms."""
    options: Any = Chem.SmilesParserParams()
    options.removeHs = False
    graph = Chem.MolFromSmiles(smiles, options)
    return bool(
        graph is not None
        and graph.GetNumAtoms()
        and all(a.GetAtomMapNum() > 0 for a in graph.GetAtoms())
    )


def unlabelled_smiles(graph: Any) -> str:
    """Strip map labels from a graph without changing connectivity or stereochemistry."""
    graph = Chem.Mol(graph)
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(0)
    return str(Chem.MolToSmiles(graph, canonical=True, isomericSmiles=True))


@lru_cache(maxsize=1)
def _model() -> Any:
    """Load the optional packaged mapper once, without network downloads or a shell."""
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"pkg_resources is deprecated as an API.*", category=UserWarning
            )
            warnings.filterwarnings(
                "ignore", category=DeprecationWarning, module=r"pkg_resources(\..*)?"
            )
            from rxnmapper import RXNMapper

            return RXNMapper()
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        raise ConfigError(
            "automatic atom mapping needs a working RXNMapper installation; "
            "install 'chemrefine[mapping]' in the driver environment "
            f"(loader reported: {exc})"
        ) from exc


@lru_cache(maxsize=128)
def _predict(reactant: str, product: str) -> tuple[tuple[int, ...], float]:
    """Request original-SMILES-order mapping indices and validate the model response."""
    try:
        result = _model().get_attention_guided_atom_maps(
            [f"{reactant}>>{product}"],
            canonicalize_rxns=False,
            detailed_output=True,
        )[0]
        raw = result["pxr_mapping_vector"]
        if any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in raw
        ):
            raise ValueError("mapper indices are not integers")
        indices = tuple(int(i) for i in raw)
        confidence = float(result["confidence"])
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("invalid mapping confidence")
    except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise ConfigError(f"automatic reaction mapper failed: {exc}") from exc
    return indices, confidence


def _skeleton(graph: Any) -> tuple[str, list[int]]:
    """Serialize heavy atoms and retain their full-graph indices in SMILES traversal order."""
    copy = Chem.Mol(graph)
    for atom in copy.GetAtoms():
        atom.SetIntProp("_full_index", atom.GetIdx())
        atom.SetAtomMapNum(0)
    # RDKit's hydrogen remover preserves H counts and tetrahedral parity. Merely
    # deleting vertices would turn saturated atoms into radicals for the model.
    params: Any = Chem.RemoveHsParameters()
    params.removeMapped = True
    params.removeDegreeZero = True
    params.removeOnlyHNeighbors = True
    params.removeHydrides = True
    params.removeDefiningBondStereo = True
    params.updateExplicitCount = True
    heavy = Chem.RemoveHs(copy, params)
    if not heavy.GetNumAtoms():
        return "", []
    smiles = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    order = json.loads(heavy.GetProp("_smilesAtomOutputOrder"))
    return str(smiles), [heavy.GetAtomWithIdx(i).GetIntProp("_full_index") for i in order]


def _hydrogens(reactant: Any, product: Any) -> list[dict[str, Any]]:
    """Preserve attached Hs, then assign transfers only when donor/acceptor routing is unique."""
    available = {a.GetIdx() for a in reactant.GetAtoms() if a.GetAtomicNum() == 1}
    pending = {a.GetIdx() for a in product.GetAtoms() if a.GetAtomicNum() == 1}

    # Preserve unchanged isolated H-only species before considering transfers.
    rfragments = list(
        zip(Chem.GetMolFrags(reactant), Chem.GetMolFrags(reactant, asMols=True), strict=True)
    )
    pfragments = zip(Chem.GetMolFrags(product), Chem.GetMolFrags(product, asMols=True), strict=True)
    for pfrag_indices, pfrag in pfragments:
        if not all(product.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in pfrag_indices):
            continue
        for rfrag_indices, rfrag in rfragments:
            if not set(rfrag_indices) <= available or unlabelled_smiles(rfrag) != unlabelled_smiles(
                pfrag
            ):
                continue
            match = rfrag.GetSubstructMatch(pfrag, useChirality=True)
            if len(match) == len(pfrag_indices):
                for pi, ri in zip(pfrag_indices, match, strict=True):
                    ridx = rfrag_indices[ri]
                    product.GetAtomWithIdx(pi).SetAtomMapNum(
                        reactant.GetAtomWithIdx(ridx).GetAtomMapNum()
                    )
                    available.remove(ridx)
                    pending.remove(pi)
                break

    def parent(atom: Any) -> tuple[int, ...]:
        return tuple(
            sorted(a.GetAtomMapNum() for a in atom.GetNeighbors() if a.GetAtomicNum() != 1)
        )

    donors: dict[tuple[int, ...], list[int]] = defaultdict(list)
    acceptors: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in sorted(available, key=lambda i: reactant.GetAtomWithIdx(i).GetAtomMapNum()):
        donors[parent(reactant.GetAtomWithIdx(index))].append(index)
    for index in sorted(pending):
        acceptors[parent(product.GetAtomWithIdx(index))].append(index)
    for key in donors.keys() & acceptors.keys():
        count = min(len(donors[key]), len(acceptors[key]))
        for ri, pi in zip(donors[key][:count], acceptors[key][:count], strict=True):
            product.GetAtomWithIdx(pi).SetAtomMapNum(reactant.GetAtomWithIdx(ri).GetAtomMapNum())
        del donors[key][:count]
        del acceptors[key][:count]
    donor_groups = {k: v for k, v in donors.items() if v}
    acceptor_groups = {k: v for k, v in acceptors.items() if v}
    if len(donor_groups) > 1 and len(acceptor_groups) > 1:
        raise ConfigError(
            "hydrogen transfer has multiple inequivalent donor-to-acceptor assignments; "
            "split the transformation into elementary steps or supply a mapped override "
            "for this reaction. No TS search was submitted."
        )
    remaining_r = [
        (parent, index) for parent in sorted(donor_groups) for index in donor_groups[parent]
    ]
    remaining_p = [
        (parent, index) for parent in sorted(acceptor_groups) for index in acceptor_groups[parent]
    ]
    if len(remaining_r) != len(remaining_p):
        raise ConfigError("hydrogen inventory is unbalanced after heavy-atom mapping")
    transfers = []
    for (rparent, ri), (pparent, pi) in zip(remaining_r, remaining_p, strict=True):
        identifier = reactant.GetAtomWithIdx(ri).GetAtomMapNum()
        product.GetAtomWithIdx(pi).SetAtomMapNum(identifier)
        transfers.append(
            {"atom_map": identifier, "from_parents": list(rparent), "to_parents": list(pparent)}
        )
    return transfers


def map_product(
    reactant: Any, product: Any, *, min_confidence: float = 0.0
) -> tuple[Any, dict[str, Any]]:
    """Map a balanced plain product graph onto the reactants' persistent identities."""
    if Counter(a.GetAtomicNum() for a in reactant.GetAtoms()) != Counter(
        a.GetAtomicNum() for a in product.GetAtoms()
    ):
        raise ConfigError(
            "reactant/product atom inventories differ; include all byproducts and hydrogens"
        )
    if Chem.GetFormalCharge(reactant) != Chem.GetFormalCharge(product):
        raise ConfigError("reactant/product formal charges differ")
    product = Chem.Mol(product)
    for atom in product.GetAtoms():
        atom.SetAtomMapNum(0)
    rtext, rorder = _skeleton(reactant)
    ptext, porder = _skeleton(product)
    if rorder:
        mapping, confidence = _predict(rtext, ptext)
        if len(mapping) != len(porder) or set(mapping) != set(range(len(rorder))):
            raise ConfigError(
                "RXNMapper returned a missing, duplicated, or non-bijective atom mapping"
            )
        if confidence < min_confidence:
            raise ConfigError(f"mapping confidence {confidence:.6g} is below {min_confidence:.6g}")
        for pi, ri in zip(porder, mapping, strict=True):
            source = reactant.GetAtomWithIdx(rorder[ri])
            target = product.GetAtomWithIdx(pi)
            if source.GetAtomicNum() != target.GetAtomicNum():
                raise ConfigError("RXNMapper correspondence changes an atom's element")
            target.SetAtomMapNum(source.GetAtomMapNum())
    else:
        confidence = None
    transfers = _hydrogens(reactant, product)
    identifiers = [a.GetAtomMapNum() for a in product.GetAtoms()]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ConfigError(
            "automatic mapping did not assign exactly one identity to every product atom"
        )
    try:
        backend_version = version("rxnmapper") if rorder else None
    except PackageNotFoundError:
        backend_version = "unknown"
    return product, {
        "method": "rxnmapper" if rorder else "hydrogen_graph",
        "backend_version": backend_version,
        "confidence": confidence,
        "hydrogen_transfers": transfers,
        "hydrogen_policy": "preserve_parent_then_unambiguous_transfer",
    }
