"""Initial coordination cores assemble separate, intact reagents before a mechanism."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from ase import Atoms
from pydantic import ValidationError
from rdkit import Chem
from test_mechanism import STEP
from test_product_core import write_xyz

from chemrefine import initial_core, io, mechanism, mechanism_batch, product_core, recovery
from chemrefine.config import (
    Config,
    MechanismReactant,
    MolecularConnectivity,
    ProductCore,
    resolve_relative_paths,
)
from chemrefine.errors import ConfigError

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/a3_initial_core"


def configuration(tmp_path):
    """Use the shipped separate-source batch with the deterministic test engine."""
    raw = yaml.safe_load((EXAMPLE / "input.yaml").read_text())
    raw["output_dir"] = tmp_path / "outputs"
    raw["executables"] = {}
    for stage in raw["mechanism"]["stages"]:
        for role in ["reactant", "product"]:
            stage[role]["steps"] = [STEP]
        stage["ts_steps"] = [STEP]
        stage["ts_guesser"]["scoring_steps"] = [STEP]
    return resolve_relative_paths(Config.model_validate(raw), base=EXAMPLE)


@pytest.mark.parametrize("name,carbon_count", [("propyne", 3), ("1_butyne", 4), ("1_pentyne", 5)])
def test_separate_sources_complete_all_stages_and_resume(tmp_path, monkeypatch, name, carbon_count):
    """Batch expansion, rigid assembly, proton transfer and handoff agree on atom identity."""
    from fake_engine import FakeEngine

    cfg = next(v.config for v in mechanism_batch.expand(configuration(tmp_path)) if v.name == name)
    plans = mechanism.plan(cfg)
    first = plans[0]
    assert first.stage.reactant.initial_core.input == EXAMPLE / "inputs/pi_core.xyz"
    assert first.mapping["initial_core"]["coordination_edits"] == 2
    assert sum(a.GetAtomicNum() == 6 for a in first.reactant.GetAtoms()) == carbon_count
    assert Chem.GetFormalCharge(first.reactant) == -1
    assert first.config.multiplicity == 2
    assert [p.stage.name for p in plans] == ["h_abstraction", "sigma_rearrangement", "cc_addition"]
    results = mechanism.run(cfg)
    assert results[-1].state.structures
    assembled = io.read_xyz_frames(first.config.output_dir / "_inputs/reactant.xyz")[0]
    mapping = json.loads((first.config.output_dir / "initial_assembly.json").read_text())
    assert len(mapping["placement_scores"]) > 1
    for source, group in zip(
        first.stage.reactant.sources, mapping["source_fragments"], strict=True
    ):
        original = mechanism._read_geometry(
            source.input, first.sources[first.stage.reactant.sources.index(source)]
        )
        np.testing.assert_allclose(
            assembled[group].get_all_distances(), original.get_all_distances(), atol=1e-6
        )
    for plan in plans:
        for role, graph in [("reactant", plan.reactant), ("product", plan.product)]:
            atoms = io.read_xyz_frames(plan.config.output_dir / "endpoints" / f"{role}.xyz")[0]
            mechanism.validate_geometry(graph, atoms, label=role)
    # Exactly the terminal alkyne H changes its parent, becoming water.
    changes = []
    for atom in first.reactant.GetAtoms():
        if atom.GetAtomicNum() == 1:
            index = atom.GetIdx()
            before = {a.GetIdx() for a in atom.GetNeighbors()}
            after = {a.GetIdx() for a in first.product.GetAtomWithIdx(index).GetNeighbors()}
            if before != after:
                changes.append((before, after))
    assert len(changes) == 1
    assert first.reactant.GetAtomWithIdx(next(iter(changes[0][0]))).GetAtomicNum() == 6
    assert first.product.GetAtomWithIdx(next(iter(changes[0][1]))).GetAtomicNum() == 8
    monkeypatch.setattr(
        FakeEngine, "submit", lambda *_: pytest.fail("resubmitted cached calculation")
    )
    mechanism.run(cfg, recovery.RunPlan(default=recovery.StepMode.RESUME))


@pytest.mark.parametrize(
    "extra,error",
    [
        ({"input": "full.xyz"}, "complete input"),
        ({"initial_core": {"input": "core.xyz", "dissociate": ["Cl"]}}, "dissociate"),
    ],
)
def test_initial_core_rejects_conflicting_inputs(extra, error):
    """A complete geometry override and dissociation cannot compete with initial assembly."""
    raw = {"sources": [{"input": "catalyst.xyz"}], "initial_core": {"input": "core.xyz"}, **extra}
    with pytest.raises(ValidationError, match=error):
        MechanismReactant.model_validate(raw)


def test_initial_template_changes_invalidate_planned_stage_identity(tmp_path):
    """Template coordinates affect cache identity even when the coordination graph is unchanged."""
    cfg = mechanism_batch.expand(configuration(tmp_path))[0].config
    first = mechanism.plan(cfg)[0]
    core = first.stage.reactant.initial_core
    atoms = io.read_xyz_frames(core.input)[0]
    atoms.positions += [2, 3, -1]
    path = write_xyz(tmp_path, "translated_core.xyz", atoms)
    replacement = first.stage.model_copy(
        update={
            "reactant": first.stage.reactant.model_copy(
                update={"initial_core": core.model_copy(update={"input": path})}
            )
        }
    )
    changed = cfg.model_copy(
        update={
            "mechanism": cfg.mechanism.model_copy(
                update={"stages": [replacement, *cfg.mechanism.stages[1:]]}
            )
        }
    )
    assert mechanism.plan(changed)[0].config.output_dir != first.config.output_dir


def test_initial_core_requires_every_fragment_to_have_a_template_anchor(tmp_path):
    """Unanchored spectators require explicit placement rather than a guessed position."""
    graph = mechanism.parse_smiles("[Cu:1].[He:2]", mapped=True)
    path = write_xyz(tmp_path, "core.xyz", Atoms("Cu", positions=[[0, 0, 0]]))
    with pytest.raises(ConfigError, match="anchor every reactant fragment"):
        initial_core.plan(graph, ProductCore(input=path), max_candidates=16)


@pytest.mark.parametrize("bonds,charges", [([(1, 2, 1)], {}), ([], {1: 1})])
def test_initial_matching_cannot_change_organic_bonds_or_explicit_charge(tmp_path, bonds, charges):
    """Initial cores cannot perform chemistry before the declared first reaction."""
    graph = mechanism.parse_smiles("[Cu:1].[OH-:2]", mapped=True)
    path = write_xyz(tmp_path, "core.xyz", Atoms("CuO", positions=[[0, 0, 0], [2, 0, 0]]))
    connectivity = MolecularConnectivity(bonds=bonds, formal_charges=charges)
    with pytest.raises(ConfigError, match="no atom-conserving"):
        initial_core.plan(
            graph, ProductCore(input=path, connectivity=connectivity), max_candidates=16
        )


def test_rigid_assembly_rejects_incompatible_template_bond_lengths(tmp_path):
    """Independent source bonds cannot be stretched to satisfy an incompatible core."""
    atoms = Atoms("CuClCl", positions=[[0, 0, 0], [-2.3, 0, 0], [0, 2.3, 0]])
    source = write_xyz(tmp_path, "catalyst.xyz", atoms)
    graph, _ = product_core.read_structure(source, MolecularConnectivity())
    for index, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    core = atoms.copy()
    core.positions[1:] *= 0.7
    path = write_xyz(tmp_path, "short_core.xyz", core)
    settings = ProductCore(
        input=path, placements=2, connectivity=MolecularConnectivity(bonds=[(1, 2, 0), (1, 3, 0)])
    )
    target, mapping = initial_core.plan(graph, settings, max_candidates=16)[0]
    assert mapping["coordination_edits"] == 0
    with pytest.raises(ConfigError, match="could not fit the template"):
        initial_core.assemble(atoms, graph, target, mapping, settings)


def test_rigid_assembly_samples_noncollinear_source_without_deformation(tmp_path):
    """Three noncollinear anchors keep a catalyst's complete orientation determined."""
    atoms = Atoms("CuClCl", positions=[[0, 0, 0], [-2.3, 0, 0], [0, 2.3, 0]])
    path = write_xyz(tmp_path, "catalyst.xyz", atoms)
    graph, _ = product_core.read_structure(path, MolecularConnectivity())
    for index, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    settings = ProductCore(input=path, placements=2)
    target, mapping = initial_core.plan(graph, settings, max_candidates=16)[0]
    result, candidates, _ = initial_core.assemble(atoms, graph, target, mapping, settings)
    assert len(candidates) == 2
    np.testing.assert_allclose(result.positions, atoms.positions, atol=1e-7)


def test_rigid_assembly_rejects_unintended_contacts(tmp_path):
    """A template cannot force two nonbonded sources inside a coordination contact."""
    atoms = Atoms("CuHe", positions=[[0, 0, 0], [10, 0, 0]])
    source = write_xyz(tmp_path, "sources.xyz", atoms)
    graph, _ = product_core.read_structure(source, MolecularConnectivity(bonds=[]))
    for index, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(index)
    core = atoms.copy()
    core.positions[1] = [1.7, 0, 0]
    path = write_xyz(tmp_path, "contact.xyz", core)
    settings = ProductCore(input=path, placements=1, connectivity=MolecularConnectivity(bonds=[]))
    target, mapping = initial_core.plan(graph, settings, max_candidates=16)[0]
    with pytest.raises(ConfigError, match="could not fit the template"):
        initial_core.assemble(atoms, graph, target, mapping, settings)
