"""Deferred product selection preserves optimized coordinates and mapped chemical identity."""

import json
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from chemrefine import io, mechanism, pipeline, reaction, recovery
from chemrefine.config import Config, load_config
from chemrefine.errors import ConfigError, JobFailureError
from chemrefine.validate import validate_config_text

# Formaldehyde + water -> methanediol, then methanediol -> formaldehyde + water.
# Every H is explicit/mapped because these reactions transfer hydrogens.
R = "[C:1](=[O:2])([H:3])[H:4].[O:5]([H:6])[H:7]"
P = "[C:1]([O:2][H:6])([O:5][H:7])([H:3])[H:4]"
STEP = {"step": 1, "engine": "fake", "operation": "sp"}


def stage(name, sources, product):
    """An elementary-reaction configuration using the suite's calculation engine."""
    return {
        "name": name,
        "reactant": {"sources": sources, "steps": [STEP]},
        "product": {"smiles": product, "steps": [STEP]},
        "ts_guesser": {"method": "idpp", "n_images": 3, "scoring_steps": [STEP]},
        "ts_steps": [STEP],
    }


def configuration(tmp_path):
    """Three linked stages: form an intermediate, release water, then select only formaldehyde."""
    stages = [
        stage("hydrate", [{"smiles": R}], P),
        stage("dehydrate", [{"from": "hydrate", "select_smiles": "OC(O)"}], R),
        stage(
            "continue",
            [{"from": "dehydrate", "select_smiles": "C=O"}, {"smiles": "[O:8]([H:9])[H:10]"}],
            "[C:1]([O:2][H:9])([O:8][H:10])([H:3])[H:4]",
        ),
    ]
    return Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "mechanism": {"conformers": 2, "stages": stages},
        }
    )


@pytest.mark.parametrize("halide", ["Cl", "Br", "I"])
def test_metal_complex_selection_does_not_encode_catalyst_ligands(halide):
    graph = mechanism.parse_smiles(f"[Cu+:1]<-[{halide}-:2].[OH2:3]", mapped=True)
    selected, indices = mechanism.select_component(graph, selection="metal_complex")
    assert indices == [0, 1]
    assert [a.GetSymbol() for a in selected.GetAtoms()] == ["Cu", halide]
    with pytest.raises(ConfigError, match="select_index"):
        mechanism.select_component(graph, selection="metal_complex", occurrence=1)


@pytest.mark.parametrize("smiles,count", [("O", 0), ("[Cu].[Pd]", 2)])
def test_metal_complex_selection_rejects_missing_or_ambiguous_catalyst(smiles, count):
    with pytest.raises(ConfigError, match=f"metal_complex selection matches {count}"):
        mechanism.select_component(
            mechanism.parse_smiles(smiles, mapped=False), selection="metal_complex"
        )


@pytest.mark.parametrize(
    "source",
    [
        {"smiles": "O", "select": "metal_complex"},
        {"from": "first", "select": "metal_complex", "select_smiles": "O"},
        {"from": "first", "select": "unknown"},
    ],
)
def test_catalyst_selector_is_only_an_unambiguous_carried_source(source):
    from chemrefine.config import MechanismSource

    with pytest.raises(ValueError):
        MechanismSource.model_validate(source)


def test_graph_planning_selects_only_requested_product_and_adds_new_reagent(tmp_path):
    cfg = configuration(tmp_path)
    plans = mechanism.plan(cfg)
    assert [p.stage.name for p in plans] == ["hydrate", "dehydrate", "continue"]
    assert mechanism.maps(plans[2].sources[0]) == [1, 2, 3, 4]
    assert mechanism.maps(plans[2].sources[1]) == [8, 9, 10]
    assert set(mechanism.maps(plans[2].reactant)) == {1, 2, 3, 4, 8, 9, 10}
    assert set(mechanism.maps(plans[1].product)) == set(range(1, 8))
    assert not cfg.output_dir.exists()  # Planning never needs optimized XYZ files.


def test_run_uses_optimized_product_coordinates_and_resume_caches(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    from fake_engine import FakeEngine

    original = FakeEngine.parse
    seen = []

    def shifted_product(self, inputs, ctx):
        result = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "product":
            prepared = []
            for struct in result.structures:
                atoms = struct.atoms.copy()
                # A rigid shift distinguishes optimized output from its generated seed.
                atoms.positions += [0.25, -0.4, 0.7]
                prepared.append(replace(struct, atoms=atoms))
            return replace(result, structures=tuple(prepared))
        if ctx.step_dir.parent.name == "reactant":
            seen.append(ctx.prev_state.structures[0].atoms.copy())
        return result

    monkeypatch.setattr(FakeEngine, "parse", shifted_product)
    recovery.execute(cfg, recovery.Action.RUN)
    assert len(seen) == 3
    prior = io.read_xyz_frames(cfg.output_dir / "dehydrate" / "endpoints" / "product.xyz")[0]
    next_seed = io.read_xyz_frames(cfg.output_dir / "continue" / "_inputs" / "reactant.xyz")[0]
    np.testing.assert_allclose(next_seed.positions[:4], prior.positions[:4], atol=1e-7)
    intermediate = io.read_xyz_frames(cfg.output_dir / "hydrate" / "endpoints" / "product.xyz")[0]
    np.testing.assert_allclose(seen[1].positions, intermediate.positions, atol=1e-7)
    binding = json.loads((cfg.output_dir / "continue" / "handoff.json").read_text())
    assert binding["selections"][0]["atom_maps"] == [1, 2, 3, 4]
    assert binding["excluded_atom_maps_by_source_stage"] == {"dehydrate": [5, 6, 7]}
    metadata = json.loads((cfg.output_dir / "continue" / "endpoints" / "reactant.json").read_text())
    assert metadata["atom_maps"] == [1, 2, 3, 4, 8, 9, 10]
    assert (
        json.loads((cfg.output_dir / "mechanism.json").read_text())["verified_transition_state"]
        is False
    )
    # Resume regenerates deterministic seeds and resolves selections through
    # cached optimized outputs.
    monkeypatch.setattr(
        FakeEngine, "submit", lambda *args: pytest.fail("resubmitted unchanged step")
    )
    recovery.execute(cfg, recovery.Action.RESUME)


def test_ambiguous_product_selection_requires_maps(tmp_path):
    product = mechanism.parse_smiles("[OH2:1].[OH2:2]", mapped=True)
    with pytest.raises(ConfigError, match="matches 2"):
        mechanism.select_component(product, "O")
    selected, indices = mechanism.select_component(product, "[OH2:2]")
    assert 2 in mechanism.maps(selected) and 1 not in mechanism.maps(selected)
    assert len(indices) == 3
    with pytest.raises(ConfigError, match="matches 0"):
        mechanism.select_component(product, "CO")


def test_selection_never_cuts_a_substructure():
    product = mechanism.parse_smiles("[CH3:1][OH:2]", mapped=True)
    with pytest.raises(ConfigError, match="matches 0"):
        mechanism.select_component(product, "O")


def test_implicit_hydrogens_have_stable_parent_based_maps():
    first = mechanism.parse_smiles("[CH3:1][OH:2]", mapped=True)
    reordered = mechanism.parse_smiles("[OH:2][CH3:1]", mapped=True)
    assert mechanism.maps(first) == mechanism.maps(reordered)
    assert mechanism._signature(first) == mechanism._signature(reordered)


@pytest.mark.parametrize("failure", ["hydrogen_transfer", "unbalanced", "charge"])
def test_bad_chemistry_is_rejected_before_computation(tmp_path, failure):
    raw = configuration(tmp_path).model_dump(by_alias=True)
    if failure == "hydrogen_transfer":
        raw["mechanism"]["stages"][0]["reactant"]["sources"] = [{"smiles": "[CH2:1]=[O:2].[OH2:5]"}]
    elif failure == "unbalanced":
        raw["mechanism"]["stages"][1]["product"]["smiles"] = "[C:1](=[O:2])([H:3])[H:4]"
    else:
        raw["mechanism"]["stages"][0]["charge"] = 1
    cfg = Config.model_validate(raw)
    with pytest.raises(ConfigError):
        pipeline.run(cfg)
    assert not cfg.output_dir.exists()


def test_changed_connectivity_after_optimization_stops_before_guesser(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    from fake_engine import FakeEngine

    original = FakeEngine.parse

    def broken(self, inputs, ctx):
        result = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "product":
            s = result.structures[0]
            atoms = s.atoms.copy()
            atoms.positions[3] += [10, 0, 0]
            return replace(result, structures=(replace(s, atoms=atoms),))
        return result

    monkeypatch.setattr(FakeEngine, "parse", broken)
    monkeypatch.setattr(
        reaction, "generate_candidates", lambda *a, **kw: pytest.fail("guessed invalid endpoint")
    )
    with pytest.raises(ConfigError, match="connectivity"):
        pipeline.run(cfg)
    assert not (cfg.output_dir / "mechanism.json").exists()
    assert not (cfg.output_dir / "dehydrate" / "_inputs").exists()


def test_failed_ts_never_advances_the_mechanism(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    original = reaction.run

    def failed(*args, **kwargs):
        original(*args, **kwargs)
        raise JobFailureError("synthetic failed TS")

    monkeypatch.setattr(reaction, "run", failed)
    with pytest.raises(JobFailureError):
        pipeline.run(cfg)
    assert not (cfg.output_dir / "dehydrate" / "_inputs").exists()
    assert not (cfg.output_dir / "mechanism.json").exists()


@pytest.mark.parametrize("failure", ["forward_ref", "duplicate_stage", "mixed_modes", "bad_path"])
def test_invalid_schema(tmp_path, failure):
    raw = configuration(tmp_path).model_dump(by_alias=True)
    if failure == "forward_ref":
        raw["mechanism"]["stages"][1]["reactant"]["sources"][0]["from"] = "continue"
    elif failure == "duplicate_stage":
        raw["mechanism"]["stages"][1]["name"] = "hydrate"
    elif failure == "bad_path":
        raw["mechanism"]["stages"][0]["name"] = "../outside"
    else:
        raw["steps"] = [STEP]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_validation_checks_graphs_and_all_branch_engines(tmp_path):
    raw = configuration(tmp_path).model_dump(mode="json", by_alias=True)
    raw["mechanism"]["stages"][2]["ts_steps"][0]["engine"] = "unknown-engine"
    report = validate_config_text(json.dumps(raw), base_dir=tmp_path)
    assert not report.ok
    assert report.issues[0].loc[:3] == ("mechanism", "stages", 2)
    raw["mechanism"]["stages"][1]["product"]["smiles"] = "[OH2:5]"
    report = validate_config_text(json.dumps(raw), base_dir=tmp_path)
    assert not report.ok and report.issues[0].kind == "chemistry"


def test_nested_geometry_and_model_paths_resolve_at_config_file(tmp_path):
    raw = configuration(tmp_path).model_dump(mode="json", by_alias=True)
    stage0 = raw["mechanism"]["stages"][0]
    stage0["reactant"]["input"] = "r.xyz"
    stage0["reactant"]["sources"][0]["input"] = "source.xyz"
    stage0["product"]["input"] = "p.xyz"
    stage0["ts_steps"][0]["options"] = {"model_path": "weights.model"}
    path = tmp_path / "mechanism.yaml"
    path.write_text(json.dumps(raw))
    cfg = load_config(path)
    st = cfg.mechanism.stages[0]
    assert st.reactant.input == tmp_path / "r.xyz"
    assert st.reactant.sources[0].input == tmp_path / "source.xyz"
    assert st.product.input == tmp_path / "p.xyz"
    assert st.ts_steps[0].options["model_path"] == str(tmp_path / "weights.model")


def test_changed_upstream_optimization_invalidates_downstream_inputs(tmp_path, monkeypatch):
    """A carried geometry change must trigger new downstream calculations on resume."""
    cfg = configuration(tmp_path)
    pipeline.run(cfg)
    from fake_engine import FakeEngine

    parse_original, submit_original = FakeEngine.parse, FakeEngine.submit
    submitted = []

    def parse_shifted(self, inputs, ctx):
        result = parse_original(self, inputs, ctx)
        if ctx.step_dir.parents[1].name == "hydrate" and ctx.step_dir.parent.name == "product":
            atoms = result.structures[0].atoms.copy()
            atoms.positions += [1.5, 0, 0]
            return replace(result, structures=(replace(result.structures[0], atoms=atoms),))
        return result

    def submit_recorded(self, inputs, ctx):
        submitted.append((ctx.step_dir.parents[1].name, ctx.step_dir.parent.name))
        if ctx.step_dir.parent.name == "reactant":
            assert all(s.energy_hartree is None for s in ctx.prev_state.structures)
        return submit_original(self, inputs, ctx)

    raw = cfg.model_dump(by_alias=True)
    # A changed endpoint calculation gets a new cache namespace.
    raw["mechanism"]["stages"][0]["product"]["steps"][0]["name"] = "new_preparation"
    monkeypatch.setattr(FakeEngine, "parse", parse_shifted)
    monkeypatch.setattr(FakeEngine, "submit", submit_recorded)
    recovery.execute(Config.model_validate(raw), recovery.Action.RESUME)
    assert ("dehydrate", "reactant") in submitted
    assert ("hydrate", "reactant") not in submitted
    previous = io.read_xyz_frames(cfg.output_dir / "hydrate" / "endpoints" / "product.xyz")[0]
    next_seed = io.read_xyz_frames(cfg.output_dir / "dehydrate" / "_inputs" / "reactant.xyz")[0]
    np.testing.assert_allclose(next_seed.positions, previous.positions, atol=1e-7)


def test_validation_rejects_mirrored_required_stereochemistry(tmp_path):
    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[F:2][C@H:1]([Cl:3])[Br:4]", mapped=True)
    atoms = mechanism.embed(graph, cfg)
    mechanism.validate_geometry(graph, atoms, label="correct stereo")
    atoms.positions[:, 0] *= -1
    with pytest.raises(ConfigError, match="stereochemistry"):
        mechanism.validate_geometry(graph, atoms, label="mirrored")


def test_validation_rejects_wrong_double_bond_stereochemistry(tmp_path):
    cfg = configuration(tmp_path)
    expected = mechanism.parse_smiles("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]", mapped=True)
    wrong = mechanism.parse_smiles("[CH3:1]/[CH:2]=[CH:3]\\[CH3:4]", mapped=True)
    atoms = mechanism.embed(wrong, cfg)
    with pytest.raises(ConfigError, match="double-bond stereochemistry"):
        mechanism.validate_geometry(expected, atoms, label="wrong isomer")


def test_inequivalent_same_element_atom_permutation_is_rejected(tmp_path):
    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[CH3:1][CH2:2][OH:3]", mapped=True)
    atoms = mechanism.embed(graph, cfg)
    mechanism.validate_geometry(graph, atoms, label="ethanol")
    atoms.positions[[0, 1]] = atoms.positions[[1, 0]]
    with pytest.raises(ConfigError, match="connectivity"):
        mechanism.validate_geometry(graph, atoms, label="reordered carbons")


def test_optional_prepared_xyz_and_source_xyz_are_used(tmp_path):
    cfg = configuration(tmp_path)
    first = mechanism.plan(cfg)[0]
    r, p = mechanism.embed(first.reactant, cfg), mechanism.embed(first.product, cfg)
    r.positions += [4, 3, 2]
    r_path, p_path = tmp_path / "r.xyz", tmp_path / "p.xyz"
    io.write_single_xyz(r, r_path)
    io.write_single_xyz(p, p_path)
    raw = cfg.model_dump(by_alias=True)
    raw["mechanism"]["stages"] = raw["mechanism"]["stages"][:1]
    raw["mechanism"]["stages"][0]["reactant"]["sources"][0]["input"] = r_path
    raw["mechanism"]["stages"][0]["reactant"]["input"] = r_path
    raw["mechanism"]["stages"][0]["product"]["input"] = p_path
    pipeline.run(Config.model_validate(raw))
    saved = io.read_xyz_frames(cfg.output_dir / "hydrate" / "_inputs" / "reactant.xyz")[0]
    np.testing.assert_array_equal(saved.positions, io.read_xyz_frames(r_path)[0].positions)


@pytest.mark.parametrize("smiles", ["CO", "[CH3:1][OH:1]", "[13CH3:1][OH:2]", "not a SMILES"])
def test_mapping_or_isotope_information_cannot_be_silently_lost(smiles):
    with pytest.raises(ConfigError):
        mechanism.parse_smiles(smiles, mapped=True)


def test_stage_lock_and_live_job_fence_cover_the_entire_mechanism(tmp_path, monkeypatch):
    from chemrefine.errors import RunLockError

    cfg = configuration(tmp_path)
    seed = cfg.output_dir / "hydrate" / "_inputs" / "reactant.xyz"
    io.write_tree_file(seed, "prior seed must survive")
    original = pipeline.require_no_live_jobs

    def later_is_running(branch):
        if branch.output_dir.parent.name == "continue":
            raise RunLockError("previous later-stage job is still live")
        original(branch)

    monkeypatch.setattr(pipeline, "require_no_live_jobs", later_is_running)
    with pytest.raises(RunLockError):
        pipeline.run(cfg)
    assert seed.read_text() == "prior seed must survive"


def test_scoped_mechanism_recovery_is_explicitly_refused(tmp_path):
    cfg = configuration(tmp_path)
    with pytest.raises(ConfigError, match="whole-workflow"):
        recovery.execute(cfg, recovery.Action.REBUILD_CACHE)
    assert not cfg.output_dir.exists()


def test_electron_count_rejects_incompatible_stage_multiplicity(tmp_path):
    raw = configuration(tmp_path).model_dump(by_alias=True)
    raw["mechanism"]["stages"][0]["multiplicity"] = 2
    with pytest.raises(ConfigError, match="electron count"):
        mechanism.plan(Config.model_validate(raw))


def test_full_reactant_input_bypasses_unnecessary_source_embedding(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    item = mechanism.plan(cfg)[0]
    reactant, product = mechanism.embed(item.reactant, cfg), mechanism.embed(item.product, cfg)
    r_path, p_path = tmp_path / "r.xyz", tmp_path / "p.xyz"
    io.write_single_xyz(reactant, r_path)
    io.write_single_xyz(product, p_path)
    raw = cfg.model_dump(by_alias=True)
    raw["mechanism"]["stages"] = raw["mechanism"]["stages"][:1]
    raw["mechanism"]["stages"][0]["reactant"]["input"] = r_path
    raw["mechanism"]["stages"][0]["product"]["input"] = p_path
    monkeypatch.setattr(
        mechanism, "embed", lambda *args: pytest.fail("embedded an overridden endpoint")
    )
    pipeline.run(Config.model_validate(raw))
    report = json.loads((cfg.output_dir / "hydrate" / "handoff.json").read_text())
    assert report["reactant_input_override"] == str(r_path)


@pytest.mark.parametrize("failure", ["no_embedding", "no_force_field", "nonconverged"])
def test_geometry_generation_failure_stops_before_calculation(tmp_path, monkeypatch, failure):
    cfg = configuration(tmp_path)
    if failure == "no_embedding":
        monkeypatch.setattr(mechanism.rdDistGeom, "EmbedMultipleConfs", lambda *args, **kw: [])
    elif failure == "no_force_field":
        monkeypatch.setattr(
            mechanism.rdForceFieldHelpers, "MMFFHasAllMoleculeParams", lambda *a: False
        )
        monkeypatch.setattr(
            mechanism.rdForceFieldHelpers, "UFFHasAllMoleculeParams", lambda *a: False
        )
    else:
        monkeypatch.setattr(
            mechanism.rdForceFieldHelpers,
            "MMFFOptimizeMoleculeConfs",
            lambda *a, **kw: [(1, 0.0)] * cfg.mechanism.conformers,
        )
    with pytest.raises(ConfigError, match=r"supply an|supply a"):
        pipeline.run(cfg)
    assert not (cfg.output_dir / "hydrate" / "reactant").exists()
    assert not (cfg.output_dir / "mechanism.json").exists()


def test_missing_or_empty_supplied_geometry_has_actionable_error(tmp_path):
    cfg = configuration(tmp_path)
    graph = mechanism.plan(cfg)[0].reactant
    path = tmp_path / "missing.xyz"
    with pytest.raises(ConfigError, match="cannot read"):
        mechanism._read_geometry(path, graph)
    path.write_text("")
    with pytest.raises(ConfigError, match="exactly one"):
        mechanism._read_geometry(path, graph)


def test_graph_sidecar_records_spin_and_original_index_binding(tmp_path):
    cfg = configuration(tmp_path)
    pipeline.run(cfg)
    path = cfg.output_dir / "hydrate" / "endpoints" / "product.json"
    binding = json.loads(path.read_text())
    assert binding["multiplicity"] == 1 and binding["charge"] == 0
    assert binding["atom_maps"] == list(range(1, 8))
    assert binding["xyz_sha256"]


def test_duplicate_selector_labels_are_rejected():
    with pytest.raises(ConfigError, match="duplicate atom-map"):
        mechanism.parse_smiles("[CH3:1][OH:1]", mapped=False)


def test_helpers_require_mechanism_settings(tmp_path):
    from chemrefine.config import StepConfig

    cfg = Config(output_dir=tmp_path, steps=[StepConfig.model_validate(STEP)])
    for action in (mechanism.plan, mechanism.run):
        with pytest.raises(ConfigError, match="no mechanism"):
            action(cfg)
    with pytest.raises(ConfigError, match="geometry generation requires"):
        mechanism.embed(mechanism.parse_smiles("O", mapped=False), cfg)


def test_selector_symmetry_limit_and_radical_signature(monkeypatch):
    from rdkit import Chem

    graph = mechanism.parse_smiles("[OH2:1]", mapped=True)
    monkeypatch.setattr(Chem.Mol, "GetSubstructMatches", lambda *a, **kw: [(0, 1, 2)] * 10001)
    with pytest.raises(ConfigError, match="too many symmetric"):
        mechanism.select_component(graph, "O")
    monkeypatch.undo()
    # Same atom and bond counts are insufficient: formal charges must also agree.
    with pytest.raises(ConfigError, match="matches 0"):
        mechanism.select_component(mechanism.parse_smiles("[NH3+:1]", mapped=True), "N")


def test_conflicting_local_aliases_need_plain_product_smiles(tmp_path):
    raw = configuration(tmp_path).model_dump(by_alias=True)
    first = raw["mechanism"]["stages"][0]
    first["reactant"]["sources"] = [{"smiles": "[OH2:1]"}, {"smiles": "[OH2:1]"}]
    first["product"]["smiles"] = "[OH2:1].[OH2:2]"
    with pytest.raises(ConfigError, match="map labels conflict"):
        mechanism.plan(Config.model_validate(raw))


@pytest.mark.parametrize("override", [{"charge": 1}, {"multiplicity": 3}])
def test_branch_charge_and_spin_must_agree_with_stage(tmp_path, override):
    raw = configuration(tmp_path).model_dump(by_alias=True)
    raw["mechanism"]["stages"][0]["ts_steps"][0].update(override)
    with pytest.raises(ConfigError, match="overrides must match"):
        mechanism.plan(Config.model_validate(raw))


def test_geometry_bond_inference_failure_is_actionable(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    graph = mechanism.plan(cfg)[0].product
    atoms = mechanism.embed(graph, cfg)

    def fail(*a, **kw):
        raise ValueError("cannot assign valences")

    monkeypatch.setattr(mechanism.rdDetermineBonds, "DetermineBonds", fail)
    with pytest.raises(ConfigError, match="cannot validate connectivity"):
        mechanism.validate_geometry(graph, atoms, label="product")
    path = tmp_path / "input.xyz"
    io.write_single_xyz(atoms, path)
    with pytest.raises(ConfigError, match="cannot infer input connectivity"):
        mechanism._read_geometry(path, graph)


def test_resonance_equivalent_coordinate_assignment_is_accepted(tmp_path):
    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[O-:1][CH:2]=[O:3]", mapped=True)
    atoms = mechanism.embed(graph, cfg)
    atoms.positions[[0, 2]] = atoms.positions[[2, 0]]
    mechanism.validate_geometry(graph, atoms, label="resonance equivalent formate")


def test_matching_double_bond_stereochemistry_is_accepted(tmp_path):
    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[CH3:1]/[CH:2]=[CH:3]/[CH3:4]", mapped=True)
    mechanism.validate_geometry(graph, mechanism.embed(graph, cfg), label="trans butene")


def test_uff_fallback_generates_valid_coordinates(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[OH2:1]", mapped=True)
    monkeypatch.setattr(mechanism.rdForceFieldHelpers, "MMFFHasAllMoleculeParams", lambda *a: False)
    atoms = mechanism.embed(graph, cfg)
    mechanism.validate_geometry(graph, atoms, label="UFF water")


def test_input_geometry_rejects_nonmolecular_coordinates(tmp_path, monkeypatch):
    from ase import Atoms

    graph = mechanism.parse_smiles("O", mapped=False)
    monkeypatch.setattr(io, "read_xyz_frames", lambda _: [Atoms("OH2", pbc=True)])
    with pytest.raises(ConfigError, match="finite nonperiodic"):
        mechanism._read_geometry(tmp_path / "periodic.xyz", graph)


def test_input_geometry_wrong_inventory_and_graph_are_rejected(tmp_path):
    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[OH2:1]", mapped=True)
    wrong = mechanism.parse_smiles("[CH4:1]", mapped=True)
    path = tmp_path / "methane.xyz"
    io.write_single_xyz(mechanism.embed(wrong, cfg), path)
    with pytest.raises(ConfigError, match="atom inventory"):
        mechanism._read_geometry(path, graph)
    # Equal atom count but different chemistry is also refused.
    graph = mechanism.parse_smiles("[NH2:1][OH:2]", mapped=True)
    with pytest.raises(ConfigError, match="does not match"):
        mechanism._read_geometry(path, graph)


def test_xyz_symmetry_limit_and_failed_correspondences(tmp_path, monkeypatch):
    from rdkit import Chem

    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[OH2:1]", mapped=True)
    path = tmp_path / "water.xyz"
    io.write_single_xyz(mechanism.embed(graph, cfg), path)

    def reject(*a, **kw):
        raise ConfigError("wrong correspondence")

    monkeypatch.setattr(mechanism, "validate_geometry", reject)
    monkeypatch.setattr(Chem.Mol, "GetSubstructMatches", lambda *a, **kw: [(0, 1, 2)] * 10001)
    with pytest.raises(ConfigError, match="symmetry search limit"):
        mechanism._read_geometry(path, graph)
    monkeypatch.setattr(Chem.Mol, "GetSubstructMatches", lambda *a, **kw: [(0, 1, 2)])
    with pytest.raises(ConfigError, match="does not match"):
        mechanism._read_geometry(path, graph)


def test_xyz_correspondence_uses_resonance_fallback(tmp_path, monkeypatch):
    from rdkit import Chem

    cfg = configuration(tmp_path)
    graph = mechanism.parse_smiles("[O-:1][CH:2]=[O:3]", mapped=True)
    atoms = mechanism.embed(graph, cfg)
    path = tmp_path / "formate.xyz"
    io.write_single_xyz(atoms[[1, 0, 2, 3]], path)
    original = Chem.Mol.GetSubstructMatches
    calls = []

    def match(self, query, **kw):
        calls.append(query)
        # Model the valid case where only a later resonance form matches.
        return () if len(calls) < 3 else original(self, query, **kw)

    monkeypatch.setattr(Chem.Mol, "GetSubstructMatches", match)
    result = mechanism._read_geometry(path, graph)
    mechanism.validate_geometry(graph, result, label="resonance input")
    assert len(calls) >= 3


def test_source_xyz_without_whole_reactant_override_preserves_binding(tmp_path):
    cfg = configuration(tmp_path)
    item = mechanism.plan(cfg)[0]
    path = tmp_path / "r.xyz"
    io.write_single_xyz(mechanism.embed(item.reactant, cfg), path)
    raw = cfg.model_dump(by_alias=True)
    raw["mechanism"]["stages"] = raw["mechanism"]["stages"][:1]
    raw["mechanism"]["stages"][0]["reactant"]["sources"][0]["input"] = path
    pipeline.run(Config.model_validate(raw))
    handoff = json.loads((cfg.output_dir / "hydrate" / "handoff.json").read_text())
    assert handoff["source_geometries"][0]["input"] == str(path)
    assert handoff["source_geometries"][0]["input_atom_indices"] == list(range(7))


def test_runtime_selection_must_preserve_planned_identities(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    original = mechanism.select_component
    calls = 0

    def changed(*a, **kw):
        nonlocal calls
        calls += 1
        graph, indices = original(*a, **kw)
        if calls == 3:  # two static selections, then the first runtime handoff
            graph.GetAtomWithIdx(0).SetAtomMapNum(999)
        return graph, indices

    monkeypatch.setattr(mechanism, "select_component", changed)
    with pytest.raises(ConfigError, match="changed atom identities"):
        pipeline.run(cfg)
    assert not (cfg.output_dir / "dehydrate" / "reactant").exists()


@pytest.mark.parametrize(
    "source",
    [
        {"from": "hydrate", "smiles": "O"},
        {"from": "hydrate", "input": "water.xyz"},
        {"from": "hydrate", "select_smiles": ""},
        {"smiles": "O", "select_index": 0},
    ],
)
def test_sources_require_one_unambiguous_origin(tmp_path, source):
    raw = configuration(tmp_path).model_dump(by_alias=True)
    raw["mechanism"]["stages"][1]["reactant"]["sources"] = [source]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_missing_supplied_xyz_is_a_validation_warning(tmp_path):
    import yaml

    raw = configuration(tmp_path).model_dump(mode="json", by_alias=True)
    raw["mechanism"]["stages"][0]["product"]["input"] = str(tmp_path / "missing.xyz")
    report = validate_config_text(yaml.safe_dump(raw), base_dir=tmp_path)
    assert report.ok
    assert any(w.kind == "input" and "missing.xyz" in w.message for w in report.warnings)


@pytest.mark.parametrize("kind", ["cache_only", "overrides", "stop_after"])
def test_direct_scoped_run_plan_is_refused_before_output_mutation(tmp_path, kind):
    from chemrefine.step import RunPlan, StepMode

    cfg = configuration(tmp_path)
    plans = {
        "cache_only": RunPlan(default=StepMode.CACHE_ONLY),
        "overrides": RunPlan(overrides={1: StepMode.CACHE_ONLY}),
        "stop_after": RunPlan(stop_after=1),
    }
    with pytest.raises(ConfigError, match="whole-workflow run and resume"):
        mechanism.run(cfg, plans[kind])
    assert not cfg.output_dir.exists()


def test_sole_product_is_carried_without_selector_and_preserves_coordinates(tmp_path, monkeypatch):
    from fake_engine import FakeEngine

    raw = configuration(tmp_path).model_dump(by_alias=True)
    raw["mechanism"]["stages"] = raw["mechanism"]["stages"][:2]
    del raw["mechanism"]["stages"][1]["reactant"]["sources"][0]["select_smiles"]
    cfg = Config.model_validate(raw)
    first, second = mechanism.plan(cfg)
    assert mechanism.maps(second.sources[0]) == mechanism.maps(first.product)
    assert second.stage.reactant.sources[0].select_smiles is None
    pipeline.run(cfg)
    previous = io.read_xyz_frames(cfg.output_dir / "hydrate" / "endpoints" / "product.xyz")[0]
    carried = io.read_xyz_frames(cfg.output_dir / "dehydrate" / "_inputs" / "reactant.xyz")[0]
    np.testing.assert_allclose(previous.positions, carried.positions, atol=1e-6)
    handoff = json.loads((cfg.output_dir / "dehydrate" / "handoff.json").read_text())
    assert handoff["selections"][0]["select_smiles"] is None
    assert handoff["selections"][0]["atom_maps"] == mechanism.maps(first.product)
    text = (cfg.output_dir / "final" / "index.html").read_text()
    assert "product of hydrate" in text
    monkeypatch.setattr(FakeEngine, "submit", lambda *a: pytest.fail("resubmitted unchanged job"))
    recovery.execute(cfg, recovery.Action.RESUME)


def test_multiple_products_require_selector_even_with_occurrence(tmp_path, monkeypatch):
    graph = mechanism.parse_smiles(R, mapped=True)
    for occurrence in (None, 0):
        with pytest.raises(ConfigError, match=r"2 product molecules.*provide select_smiles"):
            mechanism.select_component(graph, occurrence=occurrence)
    raw = configuration(tmp_path).model_dump(by_alias=True)
    del raw["mechanism"]["stages"][2]["reactant"]["sources"][0]["select_smiles"]
    cfg = Config.model_validate(raw)
    monkeypatch.setattr(
        reaction, "run", lambda *a, **kw: pytest.fail("submitted ambiguous mechanism")
    )
    with pytest.raises(ConfigError, match="provide select_smiles"):
        pipeline.run(cfg)
    assert not cfg.output_dir.exists()


def test_automatic_selection_still_validates_occurrence():
    graph = mechanism.parse_smiles(P, mapped=True)
    selected, indices = mechanism.select_component(graph, occurrence=0)
    assert mechanism.maps(selected) == mechanism.maps(graph)
    assert indices == list(range(graph.GetNumAtoms()))
    with pytest.raises(ConfigError, match="select_index"):
        mechanism.select_component(graph, occurrence=1)
