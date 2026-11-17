"""Core-product stages integrate graph planning, optimized handoff, reports, and caches."""

import json
from dataclasses import replace

import numpy as np
import pytest
import yaml
from ase import Atoms
from pydantic import ValidationError
from rdkit import Chem
from test_mechanism import STEP, stage
from test_product_core import complex_atoms, core_fixture, write_xyz

from chemrefine import core_workflow, io, mechanism, product_core, reaction, recovery
from chemrefine.config import Config, load_config
from chemrefine.errors import ConfigError
from chemrefine.validate import validate_config_text


@pytest.mark.parametrize(
    "filename,halide,count,spin",
    [
        ("input.yaml", "Cl", 2, 2),
        ("input_CuBr.yaml", "Br", 1, 1),
        ("input_CuI.yaml", "I", 1, 1),
    ],
)
def test_geometry_only_a3_reuses_cores_and_carries_changed_catalyst(
    tmp_path, monkeypatch, filename, halide, count, spin
):
    """The real example plans, assembles, runs and resumes with no declared bonds or IDs."""
    from pathlib import Path

    from fake_engine import FakeEngine

    from chemrefine.config import resolve_relative_paths

    example = Path(__file__).resolve().parents[1] / "examples/a3_geometry_cores"
    raw = yaml.safe_load((example / filename).read_text())
    assert "connectivity" not in (example / filename).read_text()
    raw["output_dir"] = tmp_path / "outputs"
    raw["executables"] = {}
    for declaration in raw["mechanism"]["stages"]:
        declaration["reactant"]["steps"] = [STEP]
        declaration["product"]["steps"] = [STEP]
        declaration["ts_steps"] = [STEP]
        declaration["ts_guesser"]["scoring_steps"] = [{**STEP, "operation": "sp"}]
    cfg = resolve_relative_paths(Config.model_validate(raw), base=example)
    plans = mechanism.plan(cfg)
    assert [p.stage.name for p in plans] == ["h_abstraction", "sigma_rearrangement", "cc_addition"]
    assert [p.reactant.GetNumAtoms() for p in plans] == [7 + count, 4 + count, 9 + count]
    for plan in plans:
        assert plan.config.multiplicity == spin and plan.config.charge == -1
        assert [a.GetSymbol() for a in plan.product.GetAtoms()].count(halide) == count
        assert plan.stage.product.core.input.parent == example / "inputs"
    outcomes = mechanism.run(cfg)
    for plan in plans:
        handoff = json.loads((plan.config.output_dir / "handoff.json").read_text())
        if plan.stage.name == "sigma_rearrangement":
            assert handoff["selections"][0]["select"] == "metal_complex"
            assert len(handoff["excluded_atom_maps_by_source_stage"]["h_abstraction"]) == 3
        for role, graph in [("reactant", plan.reactant), ("product", plan.product)]:
            atoms = io.read_xyz_frames(plan.config.output_dir / "endpoints" / f"{role}.xyz")[0]
            mechanism.validate_geometry(graph, atoms, label=role)
            metadata = json.loads(
                (plan.config.output_dir / "endpoints" / f"{role}.json").read_text()
            )
            assert metadata["graph_formal_charge"] is None
            assert metadata["bond_orders_established"] is False
    assert outcomes[-1].state.structures
    monkeypatch.setattr(FakeEngine, "submit", lambda *_: pytest.fail("resubmitted cached job"))
    mechanism.run(cfg, recovery.RunPlan(default=recovery.StepMode.RESUME))


def configuration(tmp_path):
    """A complete XYZ first pair, then a reduced product with an inherited spectator."""
    _, _, _, _, core = core_fixture(tmp_path)
    first = stage("first", [{"input": tmp_path / "reactant.xyz"}], None)
    first["product"] = {"input": tmp_path / "reactant.xyz", "steps": [STEP]}
    second = stage("second", [{"from": "first"}], None)
    second["product"] = {"core": core.model_dump(), "steps": [STEP]}
    return Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "mechanism": {"stages": [first, second]},
        }
    )


def test_complete_xyz_and_core_plan_without_runtime_outputs(tmp_path):
    cfg = configuration(tmp_path)
    first, second = mechanism.plan(cfg)
    assert first.mapping["method"] == "xyz_graph"
    assert second.mapping["method"] == "product_core"
    assert second.mapping["core_atom_maps"] == [1, 2, 3]
    assert second.mapping["bond_edits"] == 2
    assert mechanism.maps(first.product) == mechanism.maps(second.product) == list(range(1, 8))
    assert not cfg.output_dir.exists()


def test_force_continues_changed_metal_contact_through_handoff_assembly_and_resume(
    tmp_path, monkeypatch, caplog
):
    from fake_engine import FakeEngine

    cfg = configuration(tmp_path)
    original = FakeEngine.parse

    def dissociated(self, inputs, ctx):
        result = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "product" and ctx.step_dir.parent.parent.name == "first":
            structures = []
            for structure in result.structures:
                atoms = structure.atoms.copy()
                atoms.positions[3:] += [-3, 0, 0]
                structures.append(replace(structure, atoms=atoms))
            return replace(result, structures=tuple(structures))
        return result

    monkeypatch.setattr(FakeEngine, "parse", dissociated)
    strict_plans = mechanism.plan(cfg)
    with pytest.raises(ConfigError, match="coordination changed"):
        mechanism.run(cfg)
    forced = cfg.model_copy(update={"mechanism": cfg.mechanism.model_copy(update={"force": True})})
    plans = mechanism.plan(forced)
    assert all(
        a.config.output_dir != b.config.output_dir for a, b in zip(strict_plans, plans, strict=True)
    )
    mechanism.run(forced)
    changes = json.loads((plans[0].config.output_dir / "coordination_overrides.json").read_text())
    assert changes["force"] is True
    assert changes["endpoints"]["product"]["removed_contacts_rows"] == [[1, 4]]
    assert changes["endpoints"]["product"]["added_contacts_rows"] == []
    assert changes["endpoints"]["product"]["verified_intended_endpoint"] is False
    assert "contact override accepted changed contacts" in caplog.text
    metadata = json.loads((cfg.output_dir / "mechanism.json").read_text())
    assert metadata["force_metal_coordination"] is True
    assert metadata["verified_transition_state"] is False
    assert "force=true" in (cfg.output_dir / "final/index.html").read_text()
    manifest = json.loads((cfg.output_dir / "final/manifest.json").read_text())
    assert all(r["force_metal_coordination"] for r in manifest["geometries"])
    monkeypatch.setattr(FakeEngine, "submit", lambda *_: pytest.fail("resubmitted force cache"))
    mechanism.run(forced, recovery.RunPlan(default=recovery.StepMode.RESUME))
    assert (
        json.loads((plans[0].config.output_dir / "coordination_overrides.json").read_text())
        == changes
    )


def test_force_does_not_relax_pure_organic_validation():
    graph = mechanism.parse_smiles("O", mapped=False)
    atoms = Atoms("OH2", positions=[[0, 0, 0], [5, 0, 0], [0, 5, 0]])
    with pytest.raises(ConfigError):
        mechanism.validate_geometry(graph, atoms, label="organic", allow_metal_changes=True)


def test_explicit_endpoint_override_records_ordinary_contacts_and_isolates_cache(
    tmp_path, monkeypatch
):
    from fake_engine import FakeEngine

    path = write_xyz(
        tmp_path, "iodide.xyz", Atoms("PdIO", positions=[[0, 0, 0], [2.65, 0, 0], [6.5, 0, 0]])
    )
    declaration = stage("contact", [{"input": path}], None)
    declaration["product"] = {"input": path, "steps": [STEP]}
    cfg = Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "multiplicity": 2,
            "mechanism": {"stages": [declaration]},
        }
    )
    original = FakeEngine.parse

    def changed(self, inputs, ctx):
        parsed = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "product":
            structure = parsed.structures[0]
            atoms = structure.atoms.copy()
            atoms.positions[2] = [5.2, 0, 0]
            return replace(parsed, structures=(replace(structure, atoms=atoms),))
        return parsed

    monkeypatch.setattr(FakeEngine, "parse", changed)
    strict = mechanism.plan(cfg)[0]
    with pytest.raises(ConfigError, match="coordination changed"):
        mechanism.run(cfg)
    broad = cfg.model_copy(
        update={"mechanism": cfg.mechanism.model_copy(update={"force_endpoint_contacts": True})}
    )
    planned = mechanism.plan(broad)[0]
    assert planned.config.output_dir != strict.config.output_dir
    mechanism.run(broad)
    report = json.loads((planned.config.output_dir / "coordination_overrides.json").read_text())
    assert report["force"] is False and report["force_endpoint_contacts"] is True
    assert report["endpoints"]["product"]["added_contacts_rows"] == [[2, 3]]
    manifest = json.loads((cfg.output_dir / "final/manifest.json").read_text())
    assert all(r["force_endpoint_contacts"] for r in manifest["geometries"])
    assert "force_endpoint_contacts=true" in (cfg.output_dir / "final/index.html").read_text()
    assert (
        json.loads((cfg.output_dir / "mechanism.json").read_text())["force_endpoint_contacts"]
        is True
    )


def test_force_records_new_metal_contact_without_calling_it_an_intended_endpoint(
    tmp_path, monkeypatch
):
    from fake_engine import FakeEngine

    atoms = Atoms("PdNO", positions=[[0, 0, 0], [2.1, 0, 0], [0, 4, 0]])
    path = write_xyz(tmp_path, "starting.xyz", atoms)
    declaration = stage("bind", [{"input": path}], None)
    declaration["product"] = {"input": path, "steps": [STEP]}
    cfg = Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "multiplicity": 2,
            "mechanism": {"force": True, "stages": [declaration]},
        }
    )
    original = FakeEngine.parse

    def coordinated(self, inputs, ctx):
        result = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "product":
            s = result.structures[0]
            geometry = s.atoms.copy()
            geometry.positions[2] = [0, 2.0, 0]
            return replace(result, structures=(replace(s, atoms=geometry),))
        return result

    monkeypatch.setattr(FakeEngine, "parse", coordinated)
    mechanism.run(cfg)
    planned = mechanism.plan(cfg)[0]
    override = json.loads((planned.config.output_dir / "coordination_overrides.json").read_text())
    assert override["endpoints"]["product"]["added_contacts_rows"] == [[1, 3]]
    assert override["endpoints"]["product"]["removed_contacts_rows"] == []


@pytest.mark.parametrize("value", ["yes", 1, "all"])
def test_force_requires_an_explicit_boolean(tmp_path, value):
    data = configuration(tmp_path).model_dump()
    data["mechanism"]["force"] = value
    with pytest.raises(ValidationError):
        Config.model_validate(data)


def test_core_runtime_uses_optimized_inherited_ligand_and_resume(tmp_path, monkeypatch):
    from fake_engine import FakeEngine

    cfg = configuration(tmp_path)
    original = FakeEngine.parse
    seen = []

    def optimized_ligand(self, inputs, ctx):
        """Rotate the first optimized spectator to distinguish it from the seed."""
        result = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "product" and ctx.step_dir.parent.parent.name == "first":
            updated = []
            angle = 0.15
            rotation = np.array(
                [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
            )
            for structure in result.structures:
                atoms = structure.atoms.copy()
                atoms.positions[3:] = atoms.positions[3:] @ rotation
                seen.append(atoms.copy())
                updated.append(replace(structure, atoms=atoms))
            return replace(result, structures=tuple(updated))
        return result

    monkeypatch.setattr(FakeEngine, "parse", optimized_ligand)
    # An identity first pair exercises endpoint handoff, not an interpolator's
    # singular zero-length path. Actual interpolation is already independently tested.
    monkeypatch.setattr(reaction, "interpolate", lambda r, p, g: [r.copy(), r.copy(), p.copy()])
    recovery.execute(cfg, recovery.Action.RUN)
    first, second = mechanism.plan(cfg)
    inherited = io.read_xyz_frames(first.config.output_dir / "endpoints/product.xyz")[0]
    assembled = io.read_xyz_frames(second.config.output_dir / "_inputs/product.xyz")[0]
    assert seen
    np.testing.assert_allclose(
        assembled.get_all_distances()[3:, 3:], inherited.get_all_distances()[3:, 3:], atol=1e-6
    )

    def angle_cos(atoms):
        """Measure spectator orientation relative to the unchanged Pd-C direction."""
        p, c = atoms.positions[3] - atoms.positions[0], atoms.positions[1] - atoms.positions[0]
        return np.dot(p, c) / np.linalg.norm(p) / np.linalg.norm(c)

    assert angle_cos(assembled) == pytest.approx(angle_cos(inherited), abs=1e-6)
    assert angle_cos(assembled) != pytest.approx(angle_cos(complex_atoms()), abs=1e-3)
    metadata = json.loads((second.config.output_dir / "assembly.json").read_text())
    assert metadata["inherited_atom_maps"] == [4, 5, 6, 7]
    assert metadata["score_kind"] == "geometric penalty, not electronic energy"
    report = json.loads((cfg.output_dir / "final/manifest.json").read_text())
    for record in report["geometries"]:
        assert (cfg.output_dir / record["source"]).exists()
    before = (cfg.output_dir / "final/mechanism.xyz").read_bytes()
    monkeypatch.setattr(FakeEngine, "submit", lambda *a: pytest.fail("unchanged job resubmitted"))
    recovery.execute(cfg, recovery.Action.RESUME)
    assert (cfg.output_dir / "final/mechanism.xyz").read_bytes() == before


def test_graph_or_core_changes_cannot_reuse_old_chemistry_namespace(tmp_path):
    cfg = configuration(tmp_path)
    before = mechanism.plan(cfg)[1].config.output_dir
    core = cfg.mechanism.stages[1].product.core
    changed = core.model_copy(update={"placements": 3})
    second = cfg.mechanism.stages[1].model_copy(
        update={"product": cfg.mechanism.stages[1].product.model_copy(update={"core": changed})}
    )
    cfg = cfg.model_copy(
        update={
            "mechanism": cfg.mechanism.model_copy(
                update={"stages": [cfg.mechanism.stages[0], second]}
            )
        }
    )
    assert mechanism.plan(cfg)[1].config.output_dir != before


def test_new_xyz_reagent_has_fresh_ids_and_is_added_only_when_explicit(tmp_path):
    cfg = configuration(tmp_path)
    ammonia = write_xyz(
        tmp_path,
        "ammonia.xyz",
        Atoms("NH3", positions=[[0, 0, 0], [1, 0, 0], [-0.35, 0.94, 0], [-0.35, -0.5, 0.8]]),
    )
    core_path = write_xyz(
        tmp_path, "coordination.xyz", Atoms("NiN", positions=[[0, 0, 0], [0, 0, 2.1]])
    )
    second = cfg.mechanism.stages[1].model_dump()
    second["reactant"]["sources"].append({"input": ammonia})
    second["product"]["core"] = {"input": core_path}
    cfg = cfg.model_copy(
        update={
            "mechanism": cfg.mechanism.model_copy(
                update={
                    "stages": [
                        cfg.mechanism.stages[0],
                        type(cfg.mechanism.stages[1]).model_validate(second),
                    ]
                }
            )
        }
    )
    first, second_plan = mechanism.plan(cfg)
    assert mechanism.maps(second_plan.sources[1]) == [8, 9, 10, 11]
    assert set(mechanism.maps(first.product)).isdisjoint(mechanism.maps(second_plan.sources[1]))
    assert second_plan.product.GetAtomWithIdx(0).GetSymbol() == "Pd"
    assert second_plan.product.GetBondBetweenAtoms(0, 7) is not None
    assert second_plan.product.GetBondBetweenAtoms(0, 3) is not None


def test_core_paths_are_anchored_to_yaml_not_working_directory(tmp_path):
    cfg = configuration(tmp_path)
    data = cfg.model_dump(mode="json", exclude_none=True)
    data["mechanism"]["stages"][0]["reactant"]["sources"][0]["input"] = "reactant.xyz"
    data["mechanism"]["stages"][0]["product"]["input"] = "reactant.xyz"
    data["mechanism"]["stages"][1]["product"]["core"]["input"] = "core.xyz"
    path = tmp_path / "input.yaml"
    path.write_text(yaml.safe_dump(data))
    loaded = load_config(path)
    assert loaded.mechanism.stages[1].product.core.input == tmp_path / "core.xyz"
    assert len(mechanism.plan(loaded)) == 2
    report = validate_config_text(path.read_text(), base_dir=tmp_path)
    assert report.config is not None, report.issues


@pytest.mark.parametrize(
    "product",
    [
        {},
        {"core": {"input": "x.xyz"}, "smiles": "O"},
        {"core": {"input": "x.xyz"}, "input": "y.xyz"},
    ],
)
def test_product_modes_are_exclusive(tmp_path, product):
    cfg = configuration(tmp_path).model_dump()
    cfg["mechanism"]["stages"][1]["product"] = {**product, "steps": [STEP]}
    with pytest.raises(ValidationError, match="product"):
        Config.model_validate(cfg)


def test_core_requires_a_metal_in_starting_or_carried_reactants(tmp_path):
    cfg = configuration(tmp_path).model_dump()
    cfg["mechanism"]["stages"][1]["reactant"]["sources"] = [{"smiles": "O"}]
    with pytest.raises(ConfigError, match="no compatible carried or new atom"):
        mechanism.plan(Config.model_validate(cfg))


@pytest.mark.parametrize("metal", ["Pd", "Pt"])
@pytest.mark.parametrize("ligand", ["P", "As"])
def test_first_core_inherits_starting_metal_and_ligands(tmp_path, monkeypatch, metal, ligand):
    """Config, matching, runtime, reporting and resume agree for a first-stage core."""
    from fake_engine import FakeEngine

    atoms, _, _, _, core = core_fixture(tmp_path, metal=metal, ligand=ligand)
    first = stage("first", [{"input": tmp_path / "reactant.xyz"}], None)
    first["product"] = {"core": core.model_dump(), "steps": [STEP]}
    cfg = Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "mechanism": {"stages": [first]},
        }
    )
    (planned,) = mechanism.plan(cfg)
    assert planned.mapping["core_atom_maps"] == [1, 2, 3]
    assert planned.mapping["sources"][0]["from"] is None
    assert planned.product.GetAtomWithIdx(0).GetSymbol() == metal
    assert mechanism.maps(planned.reactant) == mechanism.maps(planned.product) == list(range(1, 8))
    recovery.execute(cfg, recovery.Action.RUN)
    assembled = io.read_xyz_frames(planned.config.output_dir / "_inputs/product.xyz")[0]
    assert assembled.get_chemical_symbols() == atoms.get_chemical_symbols()
    np.testing.assert_allclose(
        assembled.get_all_distances()[3:, 3:], atoms.get_all_distances()[3:, 3:], atol=1e-6
    )
    assert assembled.get_distance(0, 3) == pytest.approx(atoms.get_distance(0, 3), abs=1e-6)
    mechanism.validate_geometry(planned.product, assembled, label="first core product")
    assert json.loads((planned.config.output_dir / "assembly.json").read_text())[
        "inherited_atom_maps"
    ] == [4, 5, 6, 7]
    before = (cfg.output_dir / "final/mechanism.xyz").read_bytes()
    monkeypatch.setattr(FakeEngine, "submit", lambda *a: pytest.fail("unchanged job resubmitted"))
    recovery.execute(cfg, recovery.Action.RESUME)
    assert (cfg.output_dir / "final/mechanism.xyz").read_bytes() == before


@pytest.mark.parametrize("with_smiles", [False, True])
def test_first_core_combines_starting_catalyst_and_fresh_reagent(tmp_path, with_smiles):
    """XYZ and SMILES-backed reagents receive distinct identities before first-core assembly."""
    core_fixture(tmp_path)
    ammonia = write_xyz(
        tmp_path,
        "ammonia.xyz",
        Atoms(
            "NH3",
            positions=[[0, 0, 0], [1, 0, 0], [-0.35, 0.94, 0], [-0.35, -0.5, 0.8]],
        ),
    )
    core = write_xyz(tmp_path, "coordination.xyz", Atoms("NiN", positions=[[0, 0, 0], [0, 0, 2.1]]))
    source = {"input": ammonia}
    if with_smiles:
        source["smiles"] = "N"
    first = stage("coordinate", [{"input": tmp_path / "reactant.xyz"}, source], None)
    first["product"] = {"core": {"input": core}, "steps": [STEP]}
    cfg = Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "mechanism": {"stages": [first]},
        }
    )
    (planned,) = mechanism.plan(cfg)
    assert mechanism.maps(planned.sources[0]) == list(range(1, 8))
    assert mechanism.maps(planned.sources[1]) == [8, 9, 10, 11]
    assert planned.mapping["core_atom_maps"] == [1, 8]
    assert planned.product.GetAtomWithIdx(0).GetSymbol() == "Pd"
    assert planned.product.GetBondBetweenAtoms(0, 7) is not None
    recovery.execute(cfg, recovery.Action.RUN)
    product = io.read_xyz_frames(planned.config.output_dir / "_inputs/product.xyz")[0]
    assert len(product) == 11
    mechanism.validate_geometry(planned.product, product, label="starting catalyst and reagent")


@pytest.mark.parametrize("extra", [{"charge": 1}, {"connectivity": {"bonds": []}}])
def test_carried_source_cannot_redefine_charge_or_connectivity(tmp_path, extra):
    cfg = configuration(tmp_path).model_dump()
    cfg["mechanism"]["stages"][1]["reactant"]["sources"] = [{"from": "first", **extra}]
    with pytest.raises(ValidationError, match="from cannot"):
        Config.model_validate(cfg)


def test_standalone_core_planner_without_mechanism_has_clear_error():
    with pytest.raises(ConfigError, match="no mechanism"):
        core_workflow.plan(Config(steps=[STEP]))


def test_complete_xyz_product_cannot_omit_byproducts(tmp_path):
    cfg = configuration(tmp_path)
    # The core preserves every inherited atom. A complete product must do so too.
    first, second = mechanism.plan(cfg)
    assert first.product.GetNumAtoms() == second.product.GetNumAtoms() == 7
    other = write_xyz(tmp_path, "missing.xyz", Atoms("PdC", positions=[[0, 0, 0], [2, 0, 0]]))
    cfg.mechanism.stages[0].product.__dict__["input"] = other
    with pytest.raises(ConfigError, match="every reactant atom"):
        mechanism.plan(cfg)


@pytest.mark.parametrize(
    "change,reason", [({"charge": 1}, "formal charge"), ({"multiplicity": 2}, "electron count")]
)
def test_stage_electronic_settings_validated(tmp_path, change, reason):
    cfg = configuration(tmp_path)
    plan = mechanism.plan(cfg)[0]
    # Explicit graph metadata establishes the charge, so it must agree with the job.
    graph = Chem.Mol(plan.reactant)
    graph.SetBoolProp("connectivity_only", False)
    with pytest.raises(ConfigError, match=reason):
        core_workflow.validate_stage(plan.config.model_copy(update=change), graph, graph)


def test_graph_inventory_validation_checks_elements_and_identities(tmp_path):
    first = mechanism.plan(configuration(tmp_path))[0]
    wrong = Chem.RWMol(first.product)
    wrong.GetAtomWithIdx(1).SetAtomicNum(7)
    with pytest.raises(ConfigError, match="inventories differ"):
        core_workflow.validate_stage(first.config, first.reactant, wrong.GetMol())


def test_step_electronic_overrides_must_agree(tmp_path):
    cfg = configuration(tmp_path)
    first = cfg.mechanism.stages[0]
    first.reactant.steps[0].__dict__["charge"] = 1
    with pytest.raises(ConfigError, match="overrides must match"):
        mechanism.plan(cfg)


def test_missing_xyz_is_a_validation_issue(tmp_path):
    cfg = configuration(tmp_path).model_dump(mode="json", exclude_none=True)
    cfg["mechanism"]["stages"][1]["product"]["core"]["input"] = str(tmp_path / "missing.xyz")
    report = validate_config_text(yaml.safe_dump(cfg), base_dir=tmp_path)
    assert report.config is None
    assert "cannot read XYZ" in report.issues[0].message


@pytest.mark.parametrize("smiles", ["O", "[O:1]([H:2])[H:3]"])
def test_xyz_source_can_use_existing_smiles_product_mapping(tmp_path, smiles):
    water = write_xyz(
        tmp_path, "water.xyz", Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    )
    cfg = Config.model_validate(
        {"mechanism": {"stages": [stage("water", [{"input": water}], smiles)]}}
    )
    (plan,) = mechanism.plan(cfg)
    assert mechanism.maps(plan.reactant) == mechanism.maps(plan.product) == [1, 2, 3]
    assert plan.product.GetNumBonds() == 2


def test_core_planner_preserves_legacy_aliases_and_rejects_conflicts(tmp_path):
    single = Config.model_validate(
        {"mechanism": {"stages": [stage("same", [{"smiles": "[OH2:10]"}], "[OH2:10]")]}}
    )
    (plan,) = core_workflow.plan(single)
    assert 10 in mechanism.maps(plan.product)
    duplicate = Config.model_validate(
        {
            "mechanism": {
                "stages": [
                    stage(
                        "conflict",
                        [{"smiles": "[OH2:10]"}, {"smiles": "[OH2:10]"}],
                        "[OH2:10].[OH2:11]",
                    )
                ]
            }
        }
    )
    with pytest.raises(ConfigError, match="map labels conflict"):
        core_workflow.plan(duplicate)


def branching_configuration(tmp_path):
    """Two inequivalent metal-bound carbon groups create independently tracked products."""
    smiles = "[Pd:1]([CH3:2])([CH2:3][CH3:4])"
    first = stage("start", [{"smiles": smiles}], smiles)
    path = write_xyz(tmp_path, "release.xyz", Atoms("PdC", positions=[[0, 0, 0], [5, 0, 0]]))
    second = stage("release", [{"from": "start"}], None)
    second["product"] = {"core": {"input": path}, "steps": [STEP]}
    return Config.model_validate(
        {"output_dir": tmp_path / "outputs", "mechanism": {"stages": [first, second]}}
    )


def test_chemical_candidates_have_separate_names_directories_and_lineages(tmp_path):
    plans = mechanism.plan(branching_configuration(tmp_path))
    assert [p.stage.name for p in plans] == [
        "start",
        "release__candidate_001",
        "release__candidate_002",
    ]
    assert plans[1].config.output_dir != plans[2].config.output_dir
    assert all(p.mapping["lineage"]["start"] == "start" for p in plans)
    assert plans[1].mapping["lineage"]["release"] == "release__candidate_001"
    assert plans[2].mapping["lineage"]["release"] == "release__candidate_002"


def test_expanded_stage_limit_accounts_for_parent_branches(tmp_path, monkeypatch):
    cfg = branching_configuration(tmp_path)
    original = core_workflow._products

    def many(*args, **kwargs):
        """Two separately resolved templates must still honor the stage-wide limit."""
        result = original(*args, **kwargs)
        return result * 17 if args[1].name == "release" else result

    monkeypatch.setattr(core_workflow, "_products", many)
    with pytest.raises(ConfigError, match="exceeds max_candidates"):
        mechanism.plan(cfg)


def test_generated_candidate_names_cannot_collide_with_declared_stages(tmp_path):
    cfg = branching_configuration(tmp_path)
    extra = stage("release__candidate_001", [{"smiles": "O"}], "O")
    data = cfg.model_dump()
    data["mechanism"]["stages"].append(extra)
    with pytest.raises(ConfigError, match="names collide"):
        mechanism.plan(Config.model_validate(data))


def test_duplicate_carries_never_duplicate_atoms(tmp_path):
    cfg = configuration(tmp_path)
    data = cfg.model_dump()
    data["mechanism"]["stages"][1]["reactant"]["sources"] *= 2
    with pytest.raises(ConfigError, match="repeat the same atoms"):
        mechanism.plan(Config.model_validate(data))


@pytest.mark.parametrize(
    "field,value,reason", [("charge", 2, "stage charge"), ("multiplicity", 3, "stage multiplicity")]
)
def test_changed_parent_electronic_state_requires_explicit_next_stage(
    tmp_path, field, value, reason
):
    data = configuration(tmp_path).model_dump()
    data["mechanism"]["stages"][0][field] = value
    with pytest.raises(ConfigError, match=reason):
        mechanism.plan(Config.model_validate(data))
    data["mechanism"]["stages"][1][field] = value
    assert len(mechanism.plan(Config.model_validate(data))) == 2


def test_charged_reagent_requires_explicit_stage_charge(tmp_path):
    data = configuration(tmp_path).model_dump()
    sodium = write_xyz(tmp_path, "sodium.xyz", Atoms("Na", positions=[[0, 0, 0]]))
    data["mechanism"]["stages"][1]["reactant"]["sources"].append({"input": sodium, "charge": 1})
    with pytest.raises(ConfigError, match="explicit stage charge"):
        mechanism.plan(Config.model_validate(data))
    data["mechanism"]["stages"][1]["charge"] = 1
    plans = mechanism.plan(Config.model_validate(data))
    assert len(plans) == 2  # fresh Na cannot impersonate the carried metal anchor
    assert plans[1].mapping["core_atom_maps"][0] == 1


def test_selected_charged_component_requires_explicit_charge(tmp_path):
    core_path = write_xyz(
        tmp_path, "sodium_core.xyz", Atoms("PdC", positions=[[0, 0, 0], [2.6, 0, 0]])
    )
    first = stage("salt", [{"smiles": "[Na+:1].[Cl-:2]"}], "[Na+:1].[Cl-:2]")
    second = stage("bind", [{"from": "salt", "select_smiles": "[Na+]"}, {"smiles": "[CH3]"}], None)
    second["product"] = {
        "core": {"input": core_path, "connectivity": {"bonds": [[1, 2, 1]]}},
        "steps": [STEP],
    }
    second["multiplicity"] = 2
    data = {"mechanism": {"stages": [first, second]}}
    with pytest.raises(ConfigError, match="explicit stage charge"):
        mechanism.plan(Config.model_validate(data))
    second["charge"] = 1
    plans = mechanism.plan(Config.model_validate(data))
    assert Chem.GetFormalCharge(plans[1].product) == 1


def test_full_xyz_product_graph_keeps_its_own_charges_and_bonds(tmp_path):
    water = write_xyz(
        tmp_path, "water.xyz", Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    )
    first = stage("water", [{"input": water}], None)
    first["product"] = {"input": water, "steps": [STEP]}
    cfg = Config.model_validate({"mechanism": {"stages": [first]}})
    (planned,) = mechanism.plan(cfg)
    assert Chem.GetFormalCharge(planned.product) == 0
    assert not planned.product.HasProp("connectivity_only")


def test_spin_cannot_exceed_electron_count(tmp_path):
    planned = mechanism.plan(configuration(tmp_path))[0]
    with pytest.raises(ConfigError, match="electron count"):
        core_workflow.validate_stage(
            planned.config.model_copy(update={"multiplicity": 999}),
            planned.reactant,
            planned.product,
        )


def test_forked_candidates_merge_only_with_matching_shared_ancestry(tmp_path):
    smiles = "[Pd:1]([C:2])([C:3][C:4])[O:5]"
    first = stage("start", [{"smiles": smiles}], smiles)
    path = write_xyz(
        tmp_path, "couple.xyz", Atoms("PdCO", positions=[[0, 0, 0], [2.05, 0, 0], [3.4, 0, 0]])
    )
    fork = stage("fork", [{"from": "start"}], None)
    fork["product"] = {"core": {"input": path}, "steps": [STEP]}
    coordination = write_xyz(tmp_path, "bind_o.xyz", Atoms("PdO", positions=[[0, 0, 0], [0, 2, 0]]))
    left = stage("left", [{"from": "fork"}, {"smiles": "O"}], None)
    right = stage("right", [{"from": "fork"}, {"smiles": "N"}], None)
    for branch in (left, right):
        branch["product"] = {"core": {"input": coordination, "match": {2: 5}}, "steps": [STEP]}
    merge = stage(
        "merge",
        [{"from": "left", "select_smiles": "O"}, {"from": "right", "select_smiles": "N"}],
        "O.N",
    )
    cfg = Config.model_validate({"mechanism": {"stages": [first, fork, left, right, merge]}})
    plans = mechanism.plan(cfg)
    merged = [p for p in plans if p.mapping["logical_stage"] == "merge"]
    assert len(merged) == 3
    by_name = {p.stage.name: p for p in plans}
    for item in merged:
        left_parent, right_parent = [
            by_name[s.from_stage] for s in item.stage.reactant.sources if s.from_stage is not None
        ]
        assert left_parent.mapping["lineage"]["fork"] == right_parent.mapping["lineage"]["fork"]
        assert left_parent.stage.name != right_parent.stage.name


def test_real_pd_example_completes_all_22_atoms_from_three_atom_core():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "examples/product_core/input.yaml")
    first, second = mechanism.plan(cfg)
    parent = mechanism._read_geometry(first.stage.product.input, first.product)
    # Shuffling the complete endpoint must not cause a factorial H search.
    shuffled = parent[list(reversed(range(len(parent))))]
    bound = product_core.bind_geometry(first.product, shuffled, label="reordered Pd complex")
    mechanism.validate_geometry(first.product, bound, label="bound Pd")
    core = Atoms(
        numbers=[
            second.reactant.GetAtomWithIdx(i).GetAtomicNum() for i in second.mapping["core_indices"]
        ],
        positions=second.mapping["core_coordinates"],
    )
    result, candidates, _ = product_core.assemble(
        parent,
        second.reactant,
        second.product,
        core,
        tuple(second.mapping["core_indices"]),
        second.stage.product.core,
    )
    assert len(result) == 22
    assert len(candidates) >= 1
    assert second.mapping["bond_edits"] == 3
    assert len(Chem.GetMolFrags(second.product)) == 2
    assert sorted(result.numbers) == sorted(parent.numbers)


def test_all_core_example_runs_both_stages_without_a_complete_product(tmp_path):
    """First-core assembly feeds a later core while preserving all 22 atom identities."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples/product_core_from_start/input.yaml"
    data = load_config(example).model_dump()
    data.update(output_dir=tmp_path / "outputs")
    for entry in data["mechanism"]["stages"]:
        entry["reactant"]["steps"] = [STEP]
        entry["product"]["steps"] = [STEP]
        entry["ts_steps"] = [STEP]
        entry["ts_guesser"] = {"method": "idpp", "n_images": 3, "scoring_steps": [STEP]}
    cfg = Config.model_validate(data)
    first, second = mechanism.plan(cfg)
    assert all(
        p.stage.product.input is None and p.stage.product.core is not None for p in [first, second]
    )
    assert len(first.mapping["core_atom_maps"]) == 5
    assert len(second.mapping["core_atom_maps"]) == 3
    recovery.execute(cfg, recovery.Action.RUN)
    first_product = io.read_xyz_frames(first.config.output_dir / "endpoints/product.xyz")[0]
    second_reactant = io.read_xyz_frames(second.config.output_dir / "endpoints/reactant.xyz")[0]
    np.testing.assert_allclose(first_product.positions, second_reactant.positions, atol=1e-6)
    for p in [first, second]:
        for role in ["reactant", "product"]:
            atoms = io.read_xyz_frames(p.config.output_dir / f"endpoints/{role}.xyz")[0]
            assert len(atoms) == 22
            mechanism.validate_geometry(getattr(p, role), atoms, label=f"{p.stage.name} {role}")
        assert mechanism.maps(p.reactant) == mechanism.maps(p.product) == list(range(1, 23))
    assert len(json.loads((cfg.output_dir / "final/manifest.json").read_text())["geometries"]) == 6


@pytest.mark.parametrize(
    "mode", ["source_charge", "source_connectivity", "product_connectivity", "core_connectivity"]
)
def test_unused_connectivity_or_charge_settings_cannot_be_silently_ignored(tmp_path, mode):
    data = configuration(tmp_path).model_dump()
    if mode.startswith("source"):
        source = {"smiles": "O", "input": tmp_path / "water.xyz"}
        source["charge" if mode == "source_charge" else "connectivity"] = (
            1 if mode == "source_charge" else {"bonds": []}
        )
        data["mechanism"]["stages"][0]["reactant"]["sources"] = [source]
    elif mode == "product_connectivity":
        data["mechanism"]["stages"][0]["product"] = {
            "smiles": "O",
            "connectivity": {"bonds": []},
            "steps": [STEP],
        }
    else:
        data["mechanism"]["stages"][1]["product"]["connectivity"] = {"bonds": []}
    with pytest.raises(ValidationError, match="connectivity"):
        Config.model_validate(data)


def test_charged_smiles_reagent_requires_explicit_stage_charge(tmp_path):
    data = configuration(tmp_path).model_dump()
    data["mechanism"]["stages"][1]["reactant"]["sources"].append({"smiles": "[Na+]"})
    with pytest.raises(ConfigError, match="explicit stage charge"):
        mechanism.plan(Config.model_validate(data))
    data["mechanism"]["stages"][1]["charge"] = 1
    assert len(mechanism.plan(Config.model_validate(data))) == 2


def test_selected_xyz_complex_requires_declared_electronic_state(tmp_path):
    data = configuration(tmp_path).model_dump()
    release = write_xyz(
        tmp_path, "release_co.xyz", Atoms("PdCO", positions=[[0, 0, 0], [4, 0, 0], [5.35, 0, 0]])
    )
    data["mechanism"]["stages"][1]["product"]["core"]["input"] = release
    bind = write_xyz(tmp_path, "bind_n.xyz", Atoms("PdN", positions=[[0, 0, 0], [0, 0, 2.1]]))
    third = stage(
        "bind", [{"from": "second", "select_smiles": "[PH3]->[Pd]"}, {"smiles": "N"}], None
    )
    third["product"] = {"core": {"input": bind}, "steps": [STEP]}
    data["mechanism"]["stages"].append(third)
    for declarations in ({}, {"charge": 0}, {"multiplicity": 1}):
        data["mechanism"]["stages"][2] = {**third, **declarations}
        with pytest.raises(ConfigError, match="explicit stage charge and multiplicity"):
            mechanism.plan(Config.model_validate(data))
    data["mechanism"]["stages"][2] = {**third, "charge": 0, "multiplicity": 1}
    plans = mechanism.plan(Config.model_validate(data))
    assert set(mechanism.maps(plans[2].reactant)) == {1, 4, 5, 6, 7, 8, 9, 10, 11}


def test_chemistry_namespace_survives_directory_relocation(tmp_path):
    import shutil

    first_dir = tmp_path / "first"
    first_dir.mkdir()
    cfg = configuration(first_dir)
    data = cfg.model_dump(mode="json", exclude_none=True)
    data["output_dir"] = "./outputs"
    data["mechanism"]["stages"][0]["reactant"]["sources"][0]["input"] = "./reactant.xyz"
    data["mechanism"]["stages"][0]["product"]["input"] = "./reactant.xyz"
    data["mechanism"]["stages"][1]["product"]["core"]["input"] = "./core.xyz"
    (first_dir / "input.yaml").write_text(yaml.safe_dump(data))
    second_dir = tmp_path / "moved"
    shutil.copytree(first_dir, second_dir)
    first = load_config(first_dir / "input.yaml")
    second = load_config(second_dir / "input.yaml")
    before = [p.config.output_dir.relative_to(first.output_dir) for p in mechanism.plan(first)]
    after = [p.config.output_dir.relative_to(second.output_dir) for p in mechanism.plan(second)]
    assert before == after


def test_xyz_product_provenance_names_its_own_file_and_row_binding(tmp_path, monkeypatch):
    cfg = configuration(tmp_path)
    shuffled = complex_atoms()[[3, 0, 6, 1, 5, 2, 4]]
    path = write_xyz(tmp_path, "distinct_product.xyz", shuffled)
    data = cfg.model_dump()
    data["mechanism"]["stages"][0]["product"]["input"] = path
    cfg = Config.model_validate(data)
    monkeypatch.setattr(reaction, "interpolate", lambda r, p, g: [r.copy(), r.copy(), p.copy()])
    mechanism.run(cfg)
    first, second = mechanism.plan(cfg)
    metadata = json.loads((first.config.output_dir / "_inputs/product.json").read_text())
    assert metadata["input_path"] == str(path)
    expected = io.read_xyz_frames(first.config.output_dir / "_inputs/product.xyz")[0]
    reconstructed = shuffled[metadata["input_atom_indices"]]
    # A whole-molecule alignment can rotate the persisted seed.
    np.testing.assert_allclose(
        expected.get_all_distances(), reconstructed.get_all_distances(), atol=1e-6
    )
    core_metadata = json.loads((second.config.output_dir / "_inputs/product.json").read_text())
    assert core_metadata["input_path"] is None
    assert core_metadata["input_atom_indices"] is None
