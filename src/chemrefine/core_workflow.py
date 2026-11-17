"""Plan graph-backed XYZ endpoints and reduced-core mechanism candidates.

Planning uses declared graphs only. Runtime geometry assembly remains deferred
until the selected parent product has actually been optimized.
"""

from __future__ import annotations

import hashlib
import json
from itertools import product as combinations
from pathlib import Path
from typing import Any, cast

from rdkit import Chem

from chemrefine import atom_mapping, mechanism, product_core, reaction
from chemrefine.config import Config, MechanismConfig, MechanismStage
from chemrefine.errors import ConfigError


def validate_stage(config: Config, reactant: Any, product: Any) -> None:
    """Check atom accounting, known formal charges, and consistent electronic settings."""
    first = {a.GetAtomMapNum(): a.GetAtomicNum() for a in reactant.GetAtoms()}
    second = {a.GetAtomMapNum(): a.GetAtomicNum() for a in product.GetAtoms()}
    if first != second:
        raise ConfigError(
            "reactant/product mapped atom inventories differ; include every byproduct"
        )
    # A distance-only coordination graph does not establish formal oxidation states.
    for graph in (reactant, product):
        if (
            not graph.HasProp("connectivity_only") or not graph.GetBoolProp("connectivity_only")
        ) and Chem.GetFormalCharge(graph) != config.charge:
            raise ConfigError("endpoint formal charge differs from stage charge")
    electrons = sum(first.values()) - config.charge
    unpaired = config.multiplicity - 1
    if unpaired > electrons or (electrons - unpaired) % 2:
        raise ConfigError("stage multiplicity is incompatible with the electron count")
    for branch in reaction.branches(config).values():
        for step in branch.steps:
            if (step.charge is not None and step.charge != config.charge) or (
                step.multiplicity is not None and step.multiplicity != config.multiplicity
            ):
                raise ConfigError("step charge/spin overrides must match the stage")


def _products(
    config: Config,
    stage: MechanismStage,
    reactant: Any,
    aliases: dict[int, int],
    ambiguous_aliases: bool,
    carried_ids: set[int],
) -> list[tuple[Any, dict[str, Any]]]:
    """Resolve a complete endpoint or all distinct reduced-core graph completions."""
    settings = cast(MechanismConfig, config.mechanism)
    endpoint = stage.product
    if endpoint.smiles is not None:
        if atom_mapping.completely_labelled(endpoint.smiles):
            if ambiguous_aliases:
                raise ConfigError("provided map labels conflict; use ordinary product SMILES")
            graph = mechanism.parse_smiles(endpoint.smiles, mapped=True)
            for atom in graph.GetAtoms():
                atom.SetAtomMapNum(aliases.get(atom.GetAtomMapNum(), atom.GetAtomMapNum()))
            return [(mechanism.ordered(graph), {"method": "provided", "confidence": None})]
        graph, info = atom_mapping.map_product(
            reactant,
            mechanism.parse_smiles(endpoint.smiles, mapped=False),
            min_confidence=settings.min_mapping_confidence,
        )
        return [(mechanism.ordered(graph), info)]
    core = endpoint.core
    path = core.input if core is not None else endpoint.input
    path = cast(Path, path)
    graph, atoms = product_core.read_structure(
        path,
        core.connectivity if core is not None else endpoint.connectivity,
        charge=config.charge if stage.charge is None else stage.charge,
        reduced=core is not None,
    )
    if core is None and graph.GetNumAtoms() != reactant.GetNumAtoms():
        raise ConfigError("complete XYZ product must contain every reactant atom")
    matches = product_core.match_core(
        reactant,
        graph,
        anchor=core.anchor - 1 if core is not None else None,
        hints=core.match if core is not None else {},
        dissociate=core.dissociate if core is not None else [],
        max_candidates=settings.max_candidates,
        # A starting catalyst is supplied as a fresh source. Once intermediates
        # are carried, newly added metal reagents cannot replace their anchor.
        anchor_ids=carried_ids or set(mechanism.maps(reactant)),
    )
    fingerprint = hashlib.sha256(
        (
            path.read_text()
            + json.dumps(
                core.model_dump(mode="json", exclude={"input"})
                if core is not None
                else endpoint.connectivity.model_dump(),
                sort_keys=True,
            )
        ).encode()
    ).hexdigest()
    results = []
    for match in matches:
        # Renumber the actual full target graph to preserve its stereochemistry,
        # bond directions, and charges, rather than copying reactant annotations.
        product = match.graph
        if core is None:
            product = Chem.RenumberAtoms(
                graph, [match.indices.index(i) for i in range(reactant.GetNumAtoms())]
            )
            for atom, identity in zip(product.GetAtoms(), mechanism.maps(reactant), strict=True):
                atom.SetAtomMapNum(identity)
            if graph.HasProp("connectivity_only"):
                product.SetBoolProp("connectivity_only", graph.GetBoolProp("connectivity_only"))
        info = {
            "method": "product_core" if core is not None else "xyz_graph",
            "input": str(path),
            "template_sha256": fingerprint,
            "core_indices": list(match.indices),
            "core_atom_maps": [mechanism.maps(reactant)[i] for i in match.indices],
            "bond_edits": match.edits,
            "core_coordinates": atoms.positions.tolist(),
        }
        results.append((product, info))
    return results


def plan(config: Config) -> list[mechanism.StagePlan]:
    """Expand chemically distinct matches into explicitly named, linked candidate branches."""
    settings = config.mechanism
    if settings is None:
        raise ConfigError("this config has no mechanism")
    allocator = atom_mapping.IdAllocator()
    previous: dict[str, list[mechanism.StagePlan]] = {}
    result = []
    for stage in settings.stages:
        parents = list(dict.fromkeys(s.from_stage for s in stage.reactant.sources if s.from_stage))
        choices = list(combinations(*(previous[name] for name in parents)))
        planned = []
        for choice in choices:
            bindings = dict(zip(parents, choice, strict=True))
            lineage: dict[str, str] = {}
            compatible = True
            for parent in choice:
                for name, candidate in parent.mapping["lineage"].items():
                    if name in lineage and lineage[name] != candidate:
                        compatible = False
                    lineage[name] = candidate
            if not compatible:
                continue
            sources, declarations, allocations = [], [], []
            aliases: dict[int, int] = {}
            ambiguous_aliases = False
            for source in stage.reactant.sources:
                declaration = source
                if source.from_stage is not None:
                    parent = bindings[source.from_stage]
                    graph, _ = mechanism.select_component(
                        parent.product,
                        source.select_smiles,
                        occurrence=source.select_index,
                        selection=source.select,
                    )
                    if (
                        (source.select_smiles is not None or source.select is not None)
                        and graph.HasProp("connectivity_only")
                        and graph.GetBoolProp("connectivity_only")
                        and (stage.charge is None or stage.multiplicity is None)
                    ):
                        raise ConfigError(
                            "selecting from a distance-only XYZ mixture requires "
                            "explicit stage charge and multiplicity"
                        )
                    local_aliases = {i: i for i in mechanism.maps(graph)}
                    declaration = source.model_copy(update={"from_stage": parent.stage.name})
                else:
                    labelled = source.smiles is not None and atom_mapping.completely_labelled(
                        source.smiles
                    )
                    if source.smiles is not None:
                        graph = mechanism.parse_smiles(source.smiles, mapped=labelled)
                    else:
                        graph, _ = product_core.read_structure(
                            cast(Path, source.input), source.connectivity, charge=source.charge
                        )
                    if not labelled:
                        for atom in graph.GetAtoms():
                            atom.SetAtomMapNum(0)
                    graph, local_aliases = allocator.introduce(graph, retain_unused_labels=labelled)
                    graph = mechanism.ordered(graph)
                for local, persistent in local_aliases.items():
                    if local in aliases and aliases[local] != persistent:
                        ambiguous_aliases = True
                    aliases[local] = persistent
                declarations.append(declaration)
                sources.append(graph)
                allocations.append(
                    {
                        "from": declaration.from_stage,
                        "smiles": source.smiles,
                        "input": str(source.input) if source.input else None,
                        "atom_maps": mechanism.maps(graph),
                    }
                )
            reactant = sources[0]
            connectivity_only = reactant.HasProp("connectivity_only") and reactant.GetBoolProp(
                "connectivity_only"
            )
            for graph in sources[1:]:
                connectivity_only |= graph.HasProp("connectivity_only") and graph.GetBoolProp(
                    "connectivity_only"
                )
                reactant = Chem.CombineMols(reactant, graph)
            reactant = mechanism.ordered(reactant)
            if connectivity_only:
                reactant.SetBoolProp("connectivity_only", True)
            if len(set(mechanism.maps(reactant))) != reactant.GetNumAtoms():
                raise ConfigError("reactant sources repeat the same atoms")
            parent_charges = {p.config.charge for p in choice}
            parent_spins = {p.config.multiplicity for p in choice}
            expected_charge = sum(Chem.GetFormalCharge(g) for g in sources)
            if (
                parent_charges
                and stage.charge is None
                and (
                    parent_charges != {config.charge}
                    or any(
                        s.charge or (s.from_stage is None and Chem.GetFormalCharge(g))
                        for s, g in zip(stage.reactant.sources, sources, strict=True)
                    )
                    or (not connectivity_only and expected_charge != config.charge)
                )
            ):
                raise ConfigError("changed carried/reagent charge requires explicit stage charge")
            if (
                parent_spins
                and stage.multiplicity is None
                and parent_spins != {config.multiplicity}
            ):
                raise ConfigError("changed carried spin requires explicit stage multiplicity")
            carried_ids = {
                identity
                for source, graph in zip(stage.reactant.sources, sources, strict=True)
                if source.from_stage is not None
                for identity in mechanism.maps(graph)
            }
            for product, mapping in _products(
                config, stage, reactant, aliases, ambiguous_aliases, carried_ids
            ):
                concrete = stage.model_copy(
                    update={"reactant": stage.reactant.model_copy(update={"sources": declarations})}
                )
                planned.append(
                    (
                        concrete,
                        reactant,
                        product,
                        tuple(sources),
                        {
                            **mapping,
                            "sources": allocations,
                            "logical_stage": stage.name,
                            "lineage": lineage.copy(),
                        },
                    )
                )
        if len(planned) > settings.max_candidates:
            raise ConfigError(
                f"stage {stage.name} exceeds max_candidates={settings.max_candidates}"
            )
        branches = []
        for i, (concrete, reactant, product, source_tuple, mapping) in enumerate(planned):
            if len(planned) > 1:
                concrete = concrete.model_copy(
                    update={"name": f"{stage.name}__candidate_{i + 1:03d}"}
                )
            mapping["lineage"][stage.name] = concrete.name
            identity_data = dict(mapping)
            if settings.force:
                identity_data["force_metal_coordination"] = True
            if settings.force_endpoint_contacts:
                identity_data["force_endpoint_contacts"] = True
            identity_data.pop("input", None)
            identity_data["sources"] = [
                {key: value for key, value in source.items() if key != "input"}
                for source in mapping["sources"]
            ]
            identity = hashlib.sha256(
                (
                    json.dumps(identity_data, sort_keys=True)
                    + product_core.graph_key(reactant)
                    + product_core.graph_key(product)
                ).encode()
            ).hexdigest()[:16]
            cfg = mechanism.stage_config(
                config.model_copy(
                    update={"output_dir": config.output_dir / "_resolved" / identity}
                ),
                concrete,
            )
            validate_stage(cfg, reactant, product)
            item = mechanism.StagePlan(concrete, cfg, reactant, product, source_tuple, mapping)
            branches.append(item)
            result.append(item)
        previous[stage.name] = branches
    if len({item.stage.name for item in result}) != len(result):
        raise ConfigError("generated candidate names collide with a declared stage name")
    return result
