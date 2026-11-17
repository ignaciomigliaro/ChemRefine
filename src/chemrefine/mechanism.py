"""Sequential elementary reactions with graph-validated runtime geometry handoffs.

Graphs and persistent map identifiers are authoritative; coordinates are attached to
those identifiers, never matched by nearest atom or by XYZ element order alone.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from ase import Atoms
from ase.build import minimize_rotation_and_translation
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdDistGeom, rdForceFieldHelpers

from chemrefine import atom_mapping, io, mechanism_report, pipeline, product_core, reaction, slurm
from chemrefine.config import Config, MechanismStage, ReactionConfig
from chemrefine.engines import preflight_backends, preflight_steps
from chemrefine.errors import ConfigError
from chemrefine.step import RunPlan, StepMode, StepOutcome, preflight_batches

# Reserve a deterministic identifier namespace for implicit hydrogens. An implicit
# H keeps its identity only while attached to the same mapped parent. Transferred
# hydrogens must be explicitly mapped on both sides of every reaction.
_HYDROGEN_BASE = 1_000_000_000
_MAX_USER_MAP = 999_999


@dataclass(frozen=True)
class StagePlan:
    """A completely validated stage graph and the source fragments that assemble it."""

    stage: MechanismStage
    config: Config
    reactant: Any
    product: Any
    sources: tuple[Any, ...]
    mapping: dict[str, Any] = field(default_factory=dict)


def maps(mol: Any) -> list[int]:
    """Map identifiers in a molecule's coordinate order."""
    return [atom.GetAtomMapNum() for atom in mol.GetAtoms()]


def ordered(mol: Any) -> Any:
    """Canonical coordinate order by persistent identifier, independent of SMILES spelling."""
    identifiers = maps(mol)
    result = Chem.RenumberAtoms(mol, sorted(range(mol.GetNumAtoms()), key=lambda i: identifiers[i]))
    if mol.HasProp("connectivity_only"):
        result.SetBoolProp("connectivity_only", mol.GetBoolProp("connectivity_only"))
    return result


def parse_smiles(text: str, *, mapped: bool) -> Any:
    """Expand a graph, assigning stable H identifiers only for nontransferred implicit Hs."""
    options: Any = Chem.SmilesParserParams()
    options.removeHs = False
    mol = Chem.MolFromSmiles(text, options)
    if mol is None or not mol.GetNumAtoms():
        raise ConfigError(f"invalid or empty mechanism SMILES: {text!r}")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ConfigError("isotopically labelled mechanism SMILES are not supported yet")
    explicit = maps(mol)
    if mapped and (
        any(i <= 0 or i > _MAX_USER_MAP for i in explicit) or len(set(explicit)) != len(explicit)
    ):
        raise ConfigError(
            "reaction SMILES require unique atom maps 1..999999 on every explicit atom"
        )
    if not mapped and len([i for i in explicit if i]) != len({i for i in explicit if i}):
        raise ConfigError("selector SMILES has duplicate atom-map identifiers")
    original_count = mol.GetNumAtoms()
    mol = Chem.AddHs(mol)
    if mapped:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum():
                continue
            parent = atom.GetNeighbors()[0]
            ordinal = sum(
                a.GetAtomicNum() == 1 and original_count <= a.GetIdx() < atom.GetIdx()
                for a in parent.GetNeighbors()
            )
            atom.SetAtomMapNum(_HYDROGEN_BASE + parent.GetAtomMapNum() * 100 + ordinal)
        mol = ordered(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def select_component(
    product: Any,
    smiles: str | None = None,
    *,
    occurrence: int | None = None,
    selection: Literal["metal_complex"] | None = None,
) -> tuple[Any, list[int]]:
    """Select the sole product automatically, or match an explicitly selected whole molecule."""
    components: list[Any] = []
    index_groups: list[tuple[int, ...]] = []
    components = list(
        Chem.GetMolFrags(
            product, asMols=True, fragsMolAtomMapping=index_groups, sanitizeFrags=False
        )
    )
    if product.HasProp("connectivity_only"):
        for component in components:
            component.SetBoolProp("connectivity_only", product.GetBoolProp("connectivity_only"))
    if selection == "metal_complex":
        metal_components = [
            (component, indices)
            for component, indices in zip(components, index_groups, strict=True)
            if any(product_core.is_metal(atom.GetAtomicNum()) for atom in component.GetAtoms())
        ]
        if len(metal_components) != 1:
            raise ConfigError(
                f"metal_complex selection matches {len(metal_components)} molecules; need one"
            )
        components = [metal_components[0][0]]
        index_groups = [tuple(metal_components[0][1])]
    if smiles is None and len(components) != 1:
        raise ConfigError(
            f"previous stage has {len(components)} product molecules; "
            "provide select_smiles to choose which complete molecule to carry forward"
        )
    if smiles is None:
        if occurrence not in (None, 0):
            raise ConfigError(f"select_index {occurrence} is outside 1 matching molecules")
        return components[0], list(index_groups[0])
    query = parse_smiles(smiles, mapped=False)
    selected = []
    for component, indices in zip(components, index_groups, strict=True):
        if (
            component.GetNumAtoms() != query.GetNumAtoms()
            or component.GetNumBonds() != query.GetNumBonds()
        ):
            continue
        match_graph, match_query = component, query
        if component.HasProp("connectivity_only") and component.GetBoolProp("connectivity_only"):
            match_graph = product_core.adjacency_graph(component)
            match_query = product_core.adjacency_graph(query)
        matches = match_graph.GetSubstructMatches(
            match_query, useChirality=True, uniquify=False, maxMatches=10001
        )
        if len(matches) > 10000:
            raise ConfigError(
                "selector has too many symmetric matches; supply a more specific mapped SMILES"
            )
        for match in matches:
            if _signature(match_query) != _signature(Chem.RenumberAtoms(match_graph, list(match))):
                continue
            if all(
                not atom.GetAtomMapNum()
                or atom.GetAtomMapNum()
                == component.GetAtomWithIdx(match[atom.GetIdx()]).GetAtomMapNum()
                for atom in query.GetAtoms()
            ):
                selected.append((component, list(indices)))
                break
    if occurrence is not None:
        if occurrence < 0 or occurrence >= len(selected):
            raise ConfigError(
                f"select_index {occurrence} is outside {len(selected)} matching molecules"
            )
        return selected[occurrence]
    if len(selected) != 1:
        raise ConfigError(
            f"select_smiles {smiles!r} matches {len(selected)} complete product molecules; "
            "require exactly one (use select_index or optional atom maps to resolve duplicates)"
        )
    return selected[0]


def stage_config(config: Config, stage: MechanismStage) -> Config:
    """Reuse the existing elementary-reaction runner, caches and machine settings."""
    directory = config.output_dir / stage.name
    return config.model_copy(
        update={
            "mechanism": None,
            "input": None,
            "steps": [],
            "output_dir": directory,
            "charge": config.charge if stage.charge is None else stage.charge,
            "multiplicity": config.multiplicity
            if stage.multiplicity is None
            else stage.multiplicity,
            "reaction": ReactionConfig.model_validate(
                {
                    "reactant": {
                        "input": directory / "_inputs" / "reactant.xyz",
                        "steps": stage.reactant.steps,
                    },
                    "product": {
                        "input": directory / "_inputs" / "product.xyz",
                        "steps": stage.product.steps,
                    },
                    "ts_guesser": stage.ts_guesser,
                    "ts_steps": stage.ts_steps,
                }
            ),
        }
    )


def plan(config: Config) -> list[StagePlan]:
    """Resolve all graph references and atom/charge balances before any calculations run."""
    if config.mechanism is None:
        raise ConfigError("this config has no mechanism")
    from chemrefine import mechanism_batch

    if mechanism_batch.enabled(config):
        return [
            stage for variant in mechanism_batch.expand(config) for stage in plan(variant.config)
        ]
    if any(
        stage.product.core is not None
        or stage.product.smiles is None
        or any(
            source.smiles is None and source.from_stage is None for source in stage.reactant.sources
        )
        for stage in config.mechanism.stages
    ):
        from chemrefine.core_workflow import plan as core_plan

        return core_plan(config)
    previous: dict[str, Any] = {}
    allocator = atom_mapping.IdAllocator()
    result = []
    for stage in config.mechanism.stages:
        product_smiles = cast(str, stage.product.smiles)
        sources = []
        aliases: dict[int, int] = {}
        ambiguous_aliases = False
        allocations = []
        for source in stage.reactant.sources:
            if source.from_stage is not None:
                graph, _ = select_component(
                    previous[source.from_stage],
                    source.select_smiles,
                    occurrence=source.select_index,
                    selection=source.select,
                )
                source_aliases = {identifier: identifier for identifier in maps(graph)}
            else:
                labelled = atom_mapping.completely_labelled(source.smiles or "")
                graph = parse_smiles(source.smiles or "", mapped=labelled)
                if not labelled:
                    for atom in graph.GetAtoms():
                        atom.SetAtomMapNum(0)
                graph, source_aliases = allocator.introduce(graph, retain_unused_labels=labelled)
                graph = ordered(graph)
            for local, persistent in source_aliases.items():
                if local in aliases and aliases[local] != persistent:
                    ambiguous_aliases = True
                aliases[local] = persistent
            allocations.append(
                {"from": source.from_stage, "smiles": source.smiles, "atom_maps": maps(graph)}
            )
            sources.append(graph)
        reactant = sources[0]
        for graph in sources[1:]:
            reactant = Chem.CombineMols(reactant, graph)
        if len(set(maps(reactant))) != reactant.GetNumAtoms():
            raise ConfigError(f"stage {stage.name}: reactant sources repeat the same atoms")
        reactant = ordered(reactant)
        if atom_mapping.completely_labelled(product_smiles):
            if ambiguous_aliases:
                raise ConfigError(
                    "provided map labels conflict within the stage; use plain product SMILES "
                    "for automatic mapping"
                )
            product = parse_smiles(product_smiles, mapped=True)
            for atom in product.GetAtoms():
                atom.SetAtomMapNum(aliases.get(atom.GetAtomMapNum(), atom.GetAtomMapNum()))
            product = ordered(product)
            mapping_info: dict[str, Any] = {"method": "provided", "confidence": None}
        else:
            product, mapping_info = atom_mapping.map_product(
                reactant,
                parse_smiles(product_smiles, mapped=False),
                min_confidence=config.mechanism.min_mapping_confidence,
            )
            product = ordered(product)
        mapping_info.update(
            {
                "sources": allocations,
                "mapped_reaction": (
                    Chem.MolToSmiles(reactant, allHsExplicit=True)
                    + ">>"
                    + Chem.MolToSmiles(product, allHsExplicit=True)
                ),
            }
        )
        r_atoms = {a.GetAtomMapNum(): a.GetAtomicNum() for a in reactant.GetAtoms()}
        p_atoms = {a.GetAtomMapNum(): a.GetAtomicNum() for a in product.GetAtoms()}
        if r_atoms != p_atoms:
            raise ConfigError(
                f"stage {stage.name}: reactant/product mapped atom inventories differ; "
                "include byproducts and explicitly map transferred hydrogens"
            )
        cfg = stage_config(config, stage)
        if (
            Chem.GetFormalCharge(reactant) != cfg.charge
            or Chem.GetFormalCharge(product) != cfg.charge
        ):
            raise ConfigError(
                f"stage {stage.name}: both endpoint formal charges must equal charge={cfg.charge}"
            )
        electrons = sum(r_atoms.values()) - cfg.charge
        unpaired = cfg.multiplicity - 1
        if unpaired > electrons or (electrons - unpaired) % 2:
            raise ConfigError(
                f"stage {stage.name}: multiplicity is incompatible with the electron count"
            )
        for branch in reaction.branches(cfg).values():
            for step in branch.steps:
                if (step.charge is not None and step.charge != cfg.charge) or (
                    step.multiplicity is not None and step.multiplicity != cfg.multiplicity
                ):
                    raise ConfigError(
                        f"stage {stage.name}: step charge/spin overrides must match the stage"
                    )
        previous[stage.name] = product
        result.append(StagePlan(stage, cfg, reactant, product, tuple(sources), mapping_info))
    return result


def _signature(mol: Any) -> tuple[Any, ...]:
    """A bond-order/charge graph signature at fixed atom identities, excluding optional stereo."""
    atoms = tuple(
        (a.GetAtomicNum(), a.GetFormalCharge(), a.GetNumRadicalElectrons()) for a in mol.GetAtoms()
    )
    bonds = tuple(
        sorted(
            (
                min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
                str(b.GetBondType()),
            )
            for b in mol.GetBonds()
        )
    )
    return atoms, bonds


def validate_geometry(
    graph: Any,
    atoms: Atoms,
    *,
    label: str,
    allow_metal_changes: bool = False,
    allow_contact_changes: bool = False,
) -> None:
    """Reject optimization products that no longer represent their declared mapped graph."""
    if any(product_core.is_metal(a.GetAtomicNum()) for a in graph.GetAtoms()):
        product_core.validate_connectivity(
            graph,
            atoms,
            label=label,
            allow_metal_changes=allow_metal_changes,
            allow_contact_changes=allow_contact_changes,
        )
        return
    numbers = [a.GetAtomicNum() for a in graph.GetAtoms()]
    if list(atoms.numbers) != numbers or atoms.pbc.any() or not np.isfinite(atoms.positions).all():
        raise ConfigError(
            f"{label}: geometry lost its mapped atom order or finite molecular coordinates"
        )
    inferred = Chem.MolFromXYZBlock(io.plain_xyz_text(atoms))
    try:
        rdDetermineBonds.DetermineBonds(
            inferred,
            charge=Chem.GetFormalCharge(graph),
            allowChargedFragments=not any(a.GetNumRadicalElectrons() for a in graph.GetAtoms()),
            embedChiral=True,
        )
    except (ValueError, RuntimeError) as exc:
        raise ConfigError(f"{label}: cannot validate connectivity from coordinates: {exc}") from exc
    signature = _signature(inferred)
    if signature != _signature(graph):
        compatible = False
        for resonance in Chem.ResonanceMolSupplier(graph, maxStructs=128):
            Chem.SanitizeMol(resonance)
            if signature == _signature(resonance):
                compatible = True
                break
        if not compatible:
            raise ConfigError(
                f"{label}: optimized connectivity/bond orders/charge differ from declared SMILES"
            )
    Chem.AssignStereochemistry(inferred, cleanIt=True, force=True)
    for expected, observed in zip(graph.GetAtoms(), inferred.GetAtoms(), strict=True):
        if expected.HasProp("_CIPCode") and (
            not observed.HasProp("_CIPCode")
            or expected.GetProp("_CIPCode") != observed.GetProp("_CIPCode")
        ):
            raise ConfigError(
                f"{label}: optimized stereochemistry differs at atom map {expected.GetAtomMapNum()}"
            )
    for bond in graph.GetBonds():
        if bond.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY):
            observed = inferred.GetBondBetweenAtoms(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            if observed.GetStereo() != bond.GetStereo():
                raise ConfigError(
                    f"{label}: optimized double-bond stereochemistry differs from SMILES"
                )


def _place(parts: list[Atoms], gap: float) -> Atoms:
    """Preserve each fragment's internal geometry and place additional fragments without overlap."""
    combined: Atoms = parts[0].copy()
    if len(parts) > 1:
        combined.info.clear()
    for part in parts[1:]:
        added = part.copy()
        shift = combined.positions.max(axis=0)[0] - added.positions.min(axis=0)[0] + gap
        added.positions[:, 0] += shift
        combined += added
    return combined


def embed(graph: Any, config: Config) -> Atoms:
    """Sample conformers per connected component and select the lowest force-field energy."""
    settings = config.mechanism
    if settings is None:
        raise ConfigError("geometry generation requires mechanism settings")
    groups: list[tuple[int, ...]] = []
    fragments = Chem.GetMolFrags(graph, asMols=True, fragsMolAtomMapping=groups)
    parts = []
    for fragment in fragments:
        params: Any = rdDistGeom.ETKDGv3()
        params.randomSeed = settings.seed
        ids = list(
            rdDistGeom.EmbedMultipleConfs(fragment, numConfs=settings.conformers, params=params)
        )
        if not ids:
            raise ConfigError("SMILES conformer generation failed; supply an endpoint input XYZ")
        if rdForceFieldHelpers.MMFFHasAllMoleculeParams(fragment):
            energies = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(fragment, maxIters=1000)
        elif rdForceFieldHelpers.UFFHasAllMoleculeParams(fragment):
            energies = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(fragment, maxIters=1000)
        else:
            raise ConfigError("no force-field parameters for SMILES embedding; supply an input XYZ")
        choices = [
            (energy, cid)
            for cid, (status, energy) in zip(ids, energies, strict=True)
            if status == 0 and np.isfinite(energy)
        ]
        if not choices:
            raise ConfigError("no generated conformer converged; supply an endpoint input XYZ")
        conformer = fragment.GetConformer(min(choices)[1])
        parts.append(
            Atoms(
                numbers=[a.GetAtomicNum() for a in fragment.GetAtoms()],
                positions=np.array(conformer.GetPositions()),
            )
        )
    placed = _place(parts, settings.fragment_gap_angstrom)
    # GetMolFrags groups atoms by component; restore the graph's map-index order.
    indices = [i for group in groups for i in group]
    return placed[np.argsort(indices)]


def _read_geometry(path: Path, graph: Any) -> Atoms:
    """Match an unordered user XYZ to the declared graph and record its row correspondence."""
    try:
        frames = io.read_xyz_frames(path)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read mechanism geometry {path}: {exc}") from exc
    if len(frames) != 1:
        raise ConfigError(f"{path}: expected exactly one mapped endpoint frame")
    atoms = frames[0]
    if atoms.pbc.any() or not np.isfinite(atoms.positions).all():
        raise ConfigError(f"{path}: expected finite nonperiodic molecular coordinates")
    if any(product_core.is_metal(a.GetAtomicNum()) for a in graph.GetAtoms()):
        return product_core.bind_geometry(graph, atoms, label=str(path))
    try:
        validate_geometry(graph, atoms, label=str(path))
    except ConfigError:
        pass
    else:
        atoms.info["input_atom_indices"] = list(range(len(atoms)))
        atoms.info["input_path"] = str(path)
        return atoms
    inferred = Chem.MolFromXYZBlock(io.plain_xyz_text(atoms))
    try:
        rdDetermineBonds.DetermineBonds(
            inferred,
            charge=Chem.GetFormalCharge(graph),
            allowChargedFragments=not any(a.GetNumRadicalElectrons() for a in graph.GetAtoms()),
            embedChiral=True,
        )
    except (ValueError, RuntimeError) as exc:
        raise ConfigError(f"{path}: cannot infer input connectivity: {exc}") from exc
    if inferred.GetNumAtoms() != graph.GetNumAtoms():
        raise ConfigError(f"{path}: XYZ atom inventory differs from the declared SMILES")
    matches = inferred.GetSubstructMatches(
        graph, useChirality=True, uniquify=False, maxMatches=10001
    )
    if not matches:
        for resonance in Chem.ResonanceMolSupplier(graph, maxStructs=128):
            Chem.SanitizeMol(resonance)
            matches = inferred.GetSubstructMatches(
                resonance, useChirality=True, uniquify=False, maxMatches=10001
            )
            if matches:
                break
    if len(matches) > 10000:
        raise ConfigError(f"{path}: input correspondence exceeds the symmetry search limit")
    for match in matches:
        candidate = atoms[list(match)]
        try:
            validate_geometry(graph, candidate, label=str(path))
        except ConfigError:
            continue
        candidate.info["input_atom_indices"] = list(match)
        candidate.info["input_path"] = str(path)
        return candidate
    raise ConfigError(f"{path}: XYZ graph or stereochemistry does not match the declared SMILES")


def _persist(
    directory: Path,
    name: str,
    graph: Any,
    atoms: Atoms,
    *,
    multiplicity: int,
    charge: int = 0,
) -> None:
    """Save coordinates together with an explicit atom-map/index binding and graph fingerprint."""
    xyz = io.plain_xyz_text(atoms)
    io.write_tree_file(directory / f"{name}.xyz", xyz, atomic=True)
    io.write_tree_file(
        directory / f"{name}.json",
        json.dumps(
            {
                "smiles": Chem.MolToSmiles(graph, canonical=False, allHsExplicit=True),
                "atom_maps": maps(graph),
                "atom_numbers": list(map(int, atoms.numbers)),
                "xyz_sha256": hashlib.sha256(xyz.encode()).hexdigest(),
                "charge": charge,
                "graph_formal_charge": None
                if graph.HasProp("connectivity_only") and graph.GetBoolProp("connectivity_only")
                else Chem.GetFormalCharge(graph),
                "bond_orders_established": not (
                    graph.HasProp("connectivity_only") and graph.GetBoolProp("connectivity_only")
                ),
                "multiplicity": multiplicity,
                "input_path": atoms.info.get("input_path"),
                "input_atom_indices": atoms.info.get("input_atom_indices"),
            },
            indent=2,
        ),
        atomic=True,
    )


def run(config: Config, run_plan: RunPlan | None = None) -> list[StepOutcome]:
    """Resolve optimized product selections at runtime and drive each elementary reaction."""
    run_plan = run_plan or RunPlan()
    if (
        run_plan.overrides
        or run_plan.stop_after is not None
        or run_plan.default
        not in (
            StepMode.EXECUTE,
            StepMode.RESUME,
        )
    ):
        raise ConfigError("mechanisms support whole-workflow run and resume")
    settings = config.mechanism
    if settings is None:
        raise ConfigError("this config has no mechanism")
    from chemrefine import mechanism_batch

    if mechanism_batch.enabled(config):
        mechanism_batch.run(config, run_plan)
        return []
    stages = plan(config)
    outcomes = []
    final_stages = []
    products: dict[str, tuple[Any, Atoms]] = {}
    with pipeline.run_lock(config.output_dir), contextlib.ExitStack() as locks:
        for item in stages:
            locks.enter_context(pipeline.run_lock(item.config.output_dir))
        # Fence/preflight the entire sequence before changing any stage's inputs.
        for item in stages:
            for branch in reaction.branches(item.config).values():
                pipeline.require_no_live_jobs(branch)
                preflight_backends(branch.steps)
                preflight_steps(
                    branch.steps, charge=item.config.charge, multiplicity=item.config.multiplicity
                )
                preflight_batches(
                    branch, branch.steps, local=slurm.dispatch_locally(item.config.dispatch)
                )
        io.remove_tree_file(config.output_dir / "mechanism.json")
        mechanism_report.begin(config.output_dir)
        for item in stages:
            stage, cfg = item.stage, item.config
            directory = cfg.output_dir
            io.write_tree_file(
                directory / "mapping.json", json.dumps(item.mapping, indent=2), atomic=True
            )
            source_atoms: list[Atoms] = []
            atoms: Atoms | None
            source_maps = []
            selections = []
            source_geometries = []
            for source, graph in zip(stage.reactant.sources, item.sources, strict=True):
                if source.from_stage is not None:
                    previous_graph, previous_atoms = products[source.from_stage]
                    selected_graph, indices = select_component(
                        previous_graph,
                        source.select_smiles,
                        occurrence=source.select_index,
                        selection=source.select,
                    )
                    atoms = previous_atoms[indices]
                    atoms.info.clear()
                    # The declared graph and runtime selection must select identical identities.
                    if maps(selected_graph) != maps(graph):
                        raise ConfigError("runtime product selection changed atom identities")
                    validate_geometry(
                        graph,
                        atoms,
                        label=f"{stage.name} carried intermediate",
                        allow_metal_changes=settings.force,
                        allow_contact_changes=settings.force_endpoint_contacts,
                    )
                    selections.append(
                        {
                            "from": source.from_stage,
                            "select_smiles": source.select_smiles,
                            "select": source.select,
                            "select_index": source.select_index,
                            "atom_maps": maps(graph),
                            "source_indices": indices,
                        }
                    )
                else:
                    atoms = (
                        None
                        if stage.reactant.input is not None
                        else _read_geometry(source.input, graph)
                        if source.input
                        else embed(graph, config)
                    )
                if atoms is not None:
                    source_atoms.append(atoms)
                    if source.input is not None:
                        source_geometries.append(
                            {
                                "input": str(source.input),
                                "atom_maps": maps(graph),
                                "input_atom_indices": atoms.info.get("input_atom_indices"),
                            }
                        )
                source_maps.extend(maps(graph))
            if stage.reactant.input is not None:
                reactant_atoms = _read_geometry(stage.reactant.input, item.reactant)
            else:
                assembled = _place(source_atoms, settings.fragment_gap_angstrom)
                reactant_atoms = assembled[np.argsort(source_maps)]
            if stage.product.core is not None:
                core = stage.product.core
                core_atoms = Atoms(
                    numbers=[
                        item.reactant.GetAtomWithIdx(i).GetAtomicNum()
                        for i in item.mapping["core_indices"]
                    ],
                    positions=item.mapping["core_coordinates"],
                )
                product_atoms, placements, scores = product_core.assemble(
                    reactant_atoms,
                    item.reactant,
                    item.product,
                    core_atoms,
                    tuple(item.mapping["core_indices"]),
                    core,
                    allow_metal_changes=settings.force,
                    allow_contact_changes=settings.force_endpoint_contacts,
                )
                io.write_tree_file(
                    directory / "assembly_candidates.xyz",
                    "".join(io.plain_xyz_text(a) for a in placements),
                    atomic=True,
                )
                io.write_tree_file(
                    directory / "assembly.json",
                    json.dumps(
                        {
                            **item.mapping,
                            "placement_scores": scores,
                            "selected_placement": int(np.argmin(scores)),
                            "score_kind": "geometric penalty, not electronic energy",
                            "inherited_atom_maps": [
                                i
                                for i in maps(item.reactant)
                                if i not in item.mapping["core_atom_maps"]
                            ],
                        },
                        indent=2,
                    ),
                    atomic=True,
                )
            elif item.mapping["method"] == "xyz_graph":
                product_atoms = reactant_atoms.copy()
                product_atoms.info.clear()
                product_atoms.info["input_path"] = str(stage.product.input)
                product_atoms.info["input_atom_indices"] = np.argsort(
                    item.mapping["core_indices"]
                ).tolist()
                product_atoms.positions[item.mapping["core_indices"]] = item.mapping[
                    "core_coordinates"
                ]
            else:
                product_atoms = (
                    _read_geometry(stage.product.input, item.product)
                    if stage.product.input
                    else embed(item.product, config)
                )
            minimize_rotation_and_translation(reactant_atoms, product_atoms)
            validate_geometry(
                item.reactant,
                reactant_atoms,
                label=f"{stage.name} reactant seed",
                allow_metal_changes=settings.force,
                allow_contact_changes=settings.force_endpoint_contacts,
            )
            validate_geometry(
                item.product,
                product_atoms,
                label=f"{stage.name} product seed",
                allow_metal_changes=settings.force,
                allow_contact_changes=settings.force_endpoint_contacts,
            )
            _persist(
                directory / "_inputs",
                "reactant",
                item.reactant,
                reactant_atoms,
                multiplicity=cfg.multiplicity,
                charge=cfg.charge,
            )
            _persist(
                directory / "_inputs",
                "product",
                item.product,
                product_atoms,
                multiplicity=cfg.multiplicity,
                charge=cfg.charge,
            )
            io.write_tree_file(
                directory / "handoff.json",
                json.dumps(
                    {
                        "selections": selections,
                        "source_geometries": source_geometries,
                        "reactant_input_override": (
                            str(stage.reactant.input) if stage.reactant.input else None
                        ),
                        "retained_atom_maps": maps(item.reactant),
                        "excluded_atom_maps_by_source_stage": {
                            previous: sorted(set(maps(products[previous][0])) - set(source_maps))
                            for previous in {
                                s.from_stage for s in stage.reactant.sources if s.from_stage
                            }
                        },
                    },
                    indent=2,
                ),
                atomic=True,
            )
            prepared: dict[str, Atoms] = {}
            coordination_overrides: dict[str, Any] = {}

            def endpoint_ready(
                name: str,
                atoms: Atoms,
                *,
                current: StagePlan = item,
                sink: dict[str, Atoms] = prepared,
                overrides: dict[str, Any] = coordination_overrides,
            ) -> None:
                """Bind validated optimized coordinates before TS guessing and the next stage."""
                graph = current.reactant if name == "reactant" else current.product
                validate_geometry(
                    graph,
                    atoms,
                    label=f"{current.stage.name} optimized {name}",
                    allow_metal_changes=settings.force,
                    allow_contact_changes=settings.force_endpoint_contacts,
                )
                expected = {
                    tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                    for b in graph.GetBonds()
                }
                observed = product_core.distance_edges(atoms)
                if (settings.force or settings.force_endpoint_contacts) and expected != observed:
                    overrides[name] = {
                        "removed_contacts_rows": [
                            [a + 1, b + 1] for a, b in sorted(expected - observed)
                        ],
                        "added_contacts_rows": [
                            [a + 1, b + 1] for a, b in sorted(observed - expected)
                        ],
                        "atom_maps": maps(graph),
                        "verified_intended_endpoint": False,
                    }
                    logging.getLogger(__name__).warning(
                        "%s %s: contact override accepted changed contacts; "
                        "endpoint identity requires independent verification",
                        current.stage.name,
                        name,
                    )
                io.write_tree_file(
                    current.config.output_dir / "coordination_overrides.json",
                    json.dumps(
                        {
                            "force": settings.force,
                            "force_endpoint_contacts": settings.force_endpoint_contacts,
                            "endpoints": overrides,
                        },
                        indent=2,
                    ),
                    atomic=True,
                )
                _persist(
                    current.config.output_dir / "endpoints",
                    name,
                    graph,
                    atoms,
                    multiplicity=current.config.multiplicity,
                    charge=current.config.charge,
                )
                sink[name] = atoms.copy()

            stage_outcomes = reaction.run(cfg, run_plan, endpoint_check=endpoint_ready)
            outcomes.extend(stage_outcomes)
            calculations = {}
            offset = 0
            for role, steps in [
                ("reactant", stage.reactant.steps),
                ("product", stage.product.steps),
            ]:
                offset += len(steps)
                if steps:
                    calculations[role] = mechanism_report.calculation(
                        directory / role / steps[-1].dir_name(),
                        stage_outcomes[offset - 1].state.structures[0],
                    )
            for number, structure in enumerate(stage_outcomes[-1].state.structures, 1):
                calculations[f"ts_{number:03d}"] = mechanism_report.calculation(
                    directory / "ts" / stage.ts_steps[-1].dir_name(), structure
                )
            final_stages.append(
                mechanism_report.FinalStage(
                    name=stage.name,
                    reaction=" + ".join(
                        source.select_smiles or f"product of {source.from_stage}"
                        if source.from_stage
                        else str(source.smiles or source.input)
                        for source in stage.reactant.sources
                    )
                    + " → "
                    + (
                        stage.product.smiles
                        or str(
                            stage.product.core.input
                            if stage.product.core is not None
                            else stage.product.input
                        )
                    ),
                    reactant=prepared["reactant"],
                    product=prepared["product"],
                    transition_states=stage_outcomes[-1].state.structures,
                    atom_maps=maps(item.reactant),
                    charge=cfg.charge,
                    multiplicity=cfg.multiplicity,
                    source_dir=str(directory.relative_to(config.output_dir)),
                    calculations=calculations,
                    force=settings.force,
                    force_endpoint_contacts=settings.force_endpoint_contacts,
                    ts_source=str(
                        directory.relative_to(config.output_dir)
                        / "ts"
                        / stage.ts_steps[-1].dir_name()
                        / f"step{stage.ts_steps[-1].step}_survivors.xyz"
                    ),
                )
            )
            products[stage.name] = (item.product, prepared["product"])
        mechanism_report.publish(config.output_dir, final_stages)
        io.write_tree_file(
            config.output_dir / "mechanism.json",
            json.dumps(
                {
                    "status": "refinement_completed",
                    "verified_transition_state": False,
                    "force_metal_coordination": settings.force,
                    "force_endpoint_contacts": settings.force_endpoint_contacts,
                    "stages": [item.stage.name for item in stages],
                    "final_geometries": "final/index.html",
                    "note": (
                        "Each TS still requires frequency and IRC verification; "
                        "no mechanism energy is inferred."
                    ),
                },
                indent=2,
            ),
            atomic=True,
        )
    return outcomes
