"""Automatic identifiers, real packaged RXNMapper inference, and safe XYZ correspondence."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem

from chemrefine import atom_mapping, io, mechanism, pipeline
from chemrefine.config import Config
from chemrefine.errors import ConfigError

STEP = {"step": 1, "engine": "fake", "operation": "sp"}


def automatic_config(tmp_path):
    """The user's plain-SMILES example, with separate instances of old and new water."""

    def stage(name, sources, product):
        return {
            "name": name,
            "reactant": {"sources": sources, "steps": [STEP]},
            "product": {"smiles": product, "steps": [STEP]},
            "ts_guesser": {"method": "idpp", "n_images": 3, "scoring_steps": [STEP]},
            "ts_steps": [STEP],
        }

    return Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "mechanism": {
                "conformers": 2,
                "stages": [
                    stage("first", [{"smiles": "OC(O)"}], "C=O.O"),
                    stage(
                        "second",
                        [{"from": "first", "select_smiles": "C=O"}, {"smiles": "O"}],
                        "OC(O)",
                    ),
                ],
            },
        }
    )


def test_real_rxnmapper_and_fresh_second_reagent_ids(tmp_path):
    """Use actual pretrained mapping weights on both reactions, not dummy atom-map output."""
    cfg = automatic_config(tmp_path)
    first, second = mechanism.plan(cfg)
    old_atoms = set(mechanism.maps(first.product))
    carried = set(mechanism.maps(second.sources[0]))
    new_atoms = set(mechanism.maps(second.sources[1]))
    assert carried <= old_atoms
    assert not old_atoms & new_atoms
    assert set(mechanism.maps(second.product)) == carried | new_atoms
    assert len(old_atoms) == 7 and len(carried) == 4 and len(new_atoms) == 3
    for item in (first, second):
        assert item.mapping["method"] == "rxnmapper"
        assert 0 < item.mapping["confidence"] <= 1
        assert item.mapping["backend_version"] != "unknown"
        assert len(item.mapping["hydrogen_transfers"]) == 1
        r = {a.GetAtomMapNum(): a.GetAtomicNum() for a in item.reactant.GetAtoms()}
        p = {a.GetAtomMapNum(): a.GetAtomicNum() for a in item.product.GetAtoms()}
        assert r == p
    for before, after in zip((first, second), mechanism.plan(cfg), strict=True):
        assert before.mapping == after.mapping
    assert not cfg.output_dir.exists()


def test_full_ordinary_smiles_workflow_handoff_and_resume(tmp_path, monkeypatch):
    cfg = automatic_config(tmp_path)
    from fake_engine import FakeEngine

    original = FakeEngine.parse

    def optimize_product(self, inputs, ctx):
        result = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "product":
            struct = result.structures[0]
            atoms = struct.atoms.copy()
            atoms.positions += [0.5, -0.2, 0.1]
            return replace(result, structures=(replace(struct, atoms=atoms),))
        return result

    monkeypatch.setattr(FakeEngine, "parse", optimize_product)
    pipeline.run(cfg)
    first, second = mechanism.plan(cfg)
    prior = io.read_xyz_frames(cfg.output_dir / "first" / "endpoints" / "product.xyz")[0]
    next_r = io.read_xyz_frames(cfg.output_dir / "second" / "_inputs" / "reactant.xyz")[0]
    prior_by_id = dict(zip(mechanism.maps(first.product), prior.positions, strict=True))
    next_by_id = dict(zip(mechanism.maps(second.reactant), next_r.positions, strict=True))
    for identifier in mechanism.maps(second.sources[0]):
        np.testing.assert_allclose(next_by_id[identifier], prior_by_id[identifier], atol=1e-6)
    mapping = json.loads((cfg.output_dir / "second" / "mapping.json").read_text())
    assert mapping["mapped_reaction"] == second.mapping["mapped_reaction"]
    assert mapping["confidence"] == second.mapping["confidence"]
    monkeypatch.setattr(
        FakeEngine, "submit", lambda *a: pytest.fail("resubmitted cached calculation")
    )
    pipeline.run(cfg)


def test_repeated_labelled_new_reagent_is_also_fresh(tmp_path):
    from test_mechanism import configuration

    raw = configuration(tmp_path).model_dump(by_alias=True)
    # The same labels used by an earlier water do not request reuse of that water.
    raw["mechanism"]["stages"][2]["reactant"]["sources"][1]["smiles"] = "[O:5]([H:6])[H:7]"
    first, second, third = mechanism.plan(Config.model_validate(raw))
    assert not set(mechanism.maps(third.sources[1])) & set(mechanism.maps(second.product))
    assert mechanism.maps(third.sources[1]) == [8, 9, 10]
    assert set(mechanism.maps(first.reactant)) == set(range(1, 8))


def test_xyz_rows_are_matched_automatically_and_recorded(tmp_path):
    cfg = automatic_config(tmp_path)
    graph = mechanism.parse_smiles("CCO", mapped=False)
    graph, _ = atom_mapping.IdAllocator().introduce(graph, retain_unused_labels=False)
    original = mechanism.embed(graph, cfg)
    order = [8, 2, 5, 0, 7, 3, 1, 4, 6]
    path = tmp_path / "reordered.xyz"
    io.write_single_xyz(original[order], path)
    matched = mechanism._read_geometry(path, graph)
    mechanism.validate_geometry(graph, matched, label="matched XYZ")
    np.testing.assert_allclose(matched.positions[:3], original.positions[:3], atol=1e-6)
    assert matched.info["input_atom_indices"][:3] == [order.index(i) for i in range(3)]
    assert matched.info["input_path"] == str(path)


def test_skeleton_preserves_valence_and_stereochemistry():
    for text in ["[C:1]([O:2][H:6])([O:5][H:7])([H:3])[H:4]", "[F:2][C@H:1]([Cl:3])[Br:4]"]:
        graph = mechanism.parse_smiles(text, mapped=True)
        smiles, order = atom_mapping._skeleton(graph)
        heavy = Chem.MolFromSmiles(smiles)
        assert all(a.GetNumRadicalElectrons() == 0 for a in heavy.GetAtoms())
        assert len(order) == sum(a.GetAtomicNum() != 1 for a in graph.GetAtoms())
        expected = Chem.RemoveHs(graph)
        for atom in expected.GetAtoms():
            atom.SetAtomMapNum(0)
        assert Chem.MolToSmiles(heavy) == Chem.MolToSmiles(expected)


def test_ambiguous_hydrogen_routing_stops_instead_of_guessing():
    reactant = mechanism.parse_smiles("[OH:1].[OH:2].[CH3:3].[CH3:4]", mapped=True)
    product = mechanism.parse_smiles("[O:1].[O:2].[CH4:3].[CH4:4]", mapped=False)
    with pytest.raises(ConfigError, match="inequivalent donor-to-acceptor"):
        atom_mapping._hydrogens(reactant, product)


def test_mapper_confidence_can_gate_calculation(tmp_path):
    raw = automatic_config(tmp_path).model_dump(by_alias=True)
    raw["mechanism"]["min_mapping_confidence"] = 1.0
    with pytest.raises(ConfigError, match="mapping confidence"):
        pipeline.run(Config.model_validate(raw))
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("bad", [(-1, 1), (0, 0), (0,), (True, 1), (0, 4)])
def test_bad_mapper_correspondence_cannot_reach_a_ts_search(monkeypatch, bad):
    graph = mechanism.parse_smiles("CO", mapped=False)
    graph, _ = atom_mapping.IdAllocator().introduce(graph, retain_unused_labels=False)
    product = mechanism.parse_smiles("CO", mapped=False)
    atom_mapping._predict.cache_clear()
    fake = SimpleNamespace(
        get_attention_guided_atom_maps=lambda *a, **kw: [
            {"pxr_mapping_vector": bad, "confidence": 0.9}
        ]
    )
    monkeypatch.setattr(atom_mapping, "_model", lambda: fake)
    try:
        with pytest.raises(ConfigError, match=r"mapping|mapper"):
            atom_mapping.map_product(graph, product)
    finally:
        atom_mapping._predict.cache_clear()


def test_element_changing_mapping_is_rejected(monkeypatch):
    graph = mechanism.parse_smiles("CO", mapped=False)
    graph, _ = atom_mapping.IdAllocator().introduce(graph, retain_unused_labels=False)
    monkeypatch.setattr(atom_mapping, "_predict", lambda *a: ((1, 0), 0.9))
    with pytest.raises(ConfigError, match="element"):
        atom_mapping.map_product(graph, mechanism.parse_smiles("CO", mapped=False))


def test_unbalanced_reaction_is_rejected_before_model_loading(monkeypatch):
    graph = mechanism.parse_smiles("CO", mapped=False)
    graph, _ = atom_mapping.IdAllocator().introduce(graph, retain_unused_labels=False)
    monkeypatch.setattr(
        atom_mapping, "_model", lambda: pytest.fail("loaded mapper for unbalanced reaction")
    )
    with pytest.raises(ConfigError, match="inventories"):
        atom_mapping.map_product(graph, mechanism.parse_smiles("C", mapped=False))


def test_select_identical_molecule_by_occurrence_without_manual_atom_ids():
    product = mechanism.parse_smiles("[OH2:1].[OH2:2]", mapped=True)
    first, _ = mechanism.select_component(product, "O", occurrence=0)
    second, _ = mechanism.select_component(product, "O", occurrence=1)
    assert not set(mechanism.maps(first)) & set(mechanism.maps(second))
    with pytest.raises(ConfigError, match="select_index"):
        mechanism.select_component(product, "O", occurrence=2)


def test_selected_molecule_cannot_be_declared_twice(tmp_path):
    raw = automatic_config(tmp_path).model_dump(by_alias=True)
    raw["mechanism"]["stages"][1]["reactant"]["sources"] = [
        {"from": "first", "select_smiles": "C=O"},
        {"from": "first", "select_smiles": "C=O"},
    ]
    with pytest.raises(ConfigError, match="repeat the same atoms"):
        mechanism.plan(Config.model_validate(raw))


def test_allocator_skips_reserved_labels_and_refuses_overflow():
    allocator = atom_mapping.IdAllocator()
    graph = mechanism.parse_smiles("[OH2:1]", mapped=True)
    allocated, aliases = allocator.introduce(graph, retain_unused_labels=True)
    assert aliases[1] == 1
    assert len(set(mechanism.maps(allocated))) == 3
    assert allocator.fresh() not in mechanism.maps(allocated)
    allocator.next_id = 2_147_483_648
    with pytest.raises(ConfigError, match="integer range"):
        allocator.fresh()


@pytest.mark.parametrize("error", [ImportError("missing"), RuntimeError("invalid weights")])
def test_mapper_load_failure_is_actionable(monkeypatch, error):
    import sys

    def fail():
        raise error

    monkeypatch.setitem(sys.modules, "rxnmapper", SimpleNamespace(RXNMapper=fail))
    atom_mapping._model.cache_clear()
    try:
        with pytest.raises(ConfigError, match=r"chemrefine\[mapping\]"):
            atom_mapping._model()
    finally:
        atom_mapping._model.cache_clear()


@pytest.mark.parametrize("confidence", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_model_confidence_is_rejected(monkeypatch, confidence):
    fake = SimpleNamespace(
        get_attention_guided_atom_maps=lambda *a, **kw: [
            {"pxr_mapping_vector": [0], "confidence": confidence}
        ]
    )
    monkeypatch.setattr(atom_mapping, "_model", lambda: fake)
    atom_mapping._predict.cache_clear()
    try:
        with pytest.raises(ConfigError, match="invalid mapping confidence"):
            atom_mapping._predict("C", "C")
    finally:
        atom_mapping._predict.cache_clear()


def test_charge_imbalance_does_not_load_mapper(monkeypatch):
    monkeypatch.setattr(atom_mapping, "_predict", lambda *a: pytest.fail("called mapper"))
    with pytest.raises(ConfigError, match="formal charges differ"):
        atom_mapping.map_product(
            mechanism.parse_smiles("[NH4+]", mapped=False),
            mechanism.parse_smiles("N.[H]", mapped=False),
        )


def test_hydrogen_only_reaction_preserves_molecular_identity():
    reactant = mechanism.parse_smiles("[H:1][H:2].[H:3][H:4]", mapped=True)
    product, info = atom_mapping.map_product(
        reactant, mechanism.parse_smiles("[H][H].[H][H]", mapped=False)
    )
    assert info["method"] == "hydrogen_graph"
    assert info["confidence"] is None and info["backend_version"] is None
    assert mechanism.maps(product) == [1, 2, 3, 4]
    assert info["hydrogen_transfers"] == []


def test_hydrogen_species_can_transfer_and_inventory_is_checked():
    reactant = mechanism.parse_smiles("[H:1][H:2]", mapped=True)
    product = mechanism.parse_smiles("[H].[H]", mapped=False)
    atom_mapping._hydrogens(reactant, product)
    assert mechanism.maps(product) == [1, 2]
    with pytest.raises(ConfigError, match="hydrogen inventory"):
        atom_mapping._hydrogens(reactant, mechanism.parse_smiles("[H]", mapped=False))


def test_hydrogen_graph_match_must_cover_the_whole_fragment(monkeypatch):
    reactant = mechanism.parse_smiles("[H:1][H:2]", mapped=True)
    product = mechanism.parse_smiles("[H][H]", mapped=False)
    monkeypatch.setattr(Chem.Mol, "GetSubstructMatch", lambda *a, **kw: ())
    atom_mapping._hydrogens(reactant, product)
    assert set(mechanism.maps(product)) == {1, 2}


def test_missing_backend_metadata_is_recorded_as_unknown(monkeypatch):
    def absent(_):
        raise atom_mapping.PackageNotFoundError("rxnmapper")

    monkeypatch.setattr(atom_mapping, "version", absent)
    monkeypatch.setattr(atom_mapping, "_predict", lambda *a: ((0,), 0.9))
    graph = mechanism.parse_smiles("[CH4:1]", mapped=True)
    _, info = atom_mapping.map_product(graph, mechanism.parse_smiles("C", mapped=False))
    assert info["backend_version"] == "unknown"


def test_unassigned_or_duplicated_final_ids_are_rejected(monkeypatch):
    monkeypatch.setattr(atom_mapping, "_predict", lambda *a: ((0,), 0.9))
    monkeypatch.setattr(atom_mapping, "_hydrogens", lambda *a: [])
    with pytest.raises(ConfigError, match="exactly one identity"):
        atom_mapping.map_product(
            mechanism.parse_smiles("[CH4:1]", mapped=True),
            mechanism.parse_smiles("C", mapped=False),
        )
