"""Reaction branches use real pipeline caching with a synthetic calculation engine."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from pydantic import ValidationError

from chemrefine import io, pipeline, reaction, recovery
from chemrefine.config import Config, TSGuesserConfig, load_config
from chemrefine.errors import ConfigError, JobFailureError
from chemrefine.validate import validate_config_text


def config(tmp_path: Path, method: str = "idpp") -> Config:
    """A small H-transfer pair, with one calculation on each endpoint and TS."""
    io.write_single_xyz(
        Atoms("H3", positions=[[0, 0, 0], [0.8, 0, 0], [2.5, 0.1, 0]]), tmp_path / "r.xyz"
    )
    io.write_single_xyz(
        Atoms("H3", positions=[[0, 0, 0], [1.7, 0, 0], [2.5, 0.1, 0]]), tmp_path / "p.xyz"
    )
    step = {"step": 1, "engine": "fake", "operation": "sp"}
    guesser = {"method": method, "n_images": 5}
    if method in ("idpp", "geodesic"):
        guesser["scoring_steps"] = [step]
    else:
        guesser["command"] = [
            sys.executable,
            "-c",
            "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])",
            "{reactant}",
            "{output}",
        ]
    return Config.model_validate(
        {
            "output_dir": tmp_path / "outputs",
            "dispatch": "local",
            "reaction": {
                "reactant": {"input": tmp_path / "r.xyz", "steps": [step]},
                "product": {"input": tmp_path / "p.xyz", "steps": [step]},
                "ts_guesser": guesser,
                "ts_steps": [step],
            },
        }
    )


def test_complete_reaction_and_resume(tmp_path, monkeypatch):
    """Run all four branches, then prove resume reuses guesses and calculation caches."""
    cfg = config(tmp_path)
    assert recovery.execute(cfg, recovery.Action.RUN) == 0
    report = json.loads((cfg.output_dir / "reaction.json").read_text())
    assert report["status"] == "refinement_completed"
    assert report["verified_transition_state"] is False
    assert len(io.read_xyz_frames(cfg.output_dir / "guesser" / "path.xyz")) == 5
    assert len(io.read_xyz_frames(cfg.output_dir / "guesser" / "candidates.xyz")) == 3
    seed = io.read_xyz_frames(cfg.output_dir / "guesser" / "ts_guess.xyz")[0]
    candidate = io.read_xyz_frames(cfg.output_dir / "guesser" / "candidates.xyz")[0]
    np.testing.assert_allclose(seed.positions, candidate.positions)
    monkeypatch.setattr(reaction, "interpolate", lambda *a: pytest.fail("regenerated cached guess"))
    from fake_engine import FakeEngine

    monkeypatch.setattr(FakeEngine, "submit", lambda *a: pytest.fail("resubmitted cached step"))
    assert recovery.execute(cfg, recovery.Action.RESUME) == 0
    assert not (cfg.output_dir / "step1").exists()


@pytest.mark.parametrize("method", ["fragmentflow", "react-ot", "external"])
def test_model_command_adapter(tmp_path, method):
    """Named models and generic adapters execute isolated wrappers and refine their outputs."""
    cfg = config(tmp_path, method)
    outcomes = pipeline.run(cfg)
    assert len(outcomes) == 3
    assert (cfg.output_dir / "guesser" / "model.log").exists()
    assert len(io.read_xyz_frames(cfg.output_dir / "guesser" / "ts_guess.xyz")) == 1


def test_changed_product_invalidates_guess(tmp_path, monkeypatch):
    """A changed endpoint cannot reuse a previous reaction's candidate cache."""
    cfg = config(tmp_path)
    r, p = [
        pipeline.bootstrap(b).structures[0].atoms
        for name, b in reaction.branches(cfg).items()
        if name in ("reactant", "product")
    ]
    reaction.generate_candidates(cfg, r, p, resume=True)
    p.positions[1, 0] += 0.1
    monkeypatch.setattr(
        reaction, "interpolate", lambda *a: pytest.fail("new endpoint was regenerated")
    )
    with pytest.raises(pytest.fail.Exception, match="regenerated"):
        reaction.generate_candidates(cfg, r, p, resume=True)


def test_corrupt_guess_regenerated(tmp_path):
    """Candidate bytes are checked as well as endpoint/config fingerprints."""
    cfg = config(tmp_path, "react-ot")
    r = pipeline.bootstrap(reaction.branches(cfg)["reactant"]).structures[0].atoms
    p = pipeline.bootstrap(reaction.branches(cfg)["product"]).structures[0].atoms
    path = reaction.generate_candidates(cfg, r, p, resume=True)
    path.write_text("bad")
    reaction.generate_candidates(cfg, r, p, resume=True)
    assert len(io.read_xyz_frames(path)) == 1


def test_invalid_mapping_before_calculations(tmp_path, monkeypatch):
    """Element/order mismatches must fail before submitting endpoint calculations."""
    cfg = config(tmp_path)
    io.write_single_xyz(Atoms("H2O"), tmp_path / "p.xyz")
    from fake_engine import FakeEngine

    monkeypatch.setattr(FakeEngine, "submit", lambda *a: pytest.fail("submitted invalid reaction"))
    with pytest.raises(ConfigError, match="mapped order"):
        pipeline.run(cfg)


def test_failed_command_cannot_reuse_old_output(tmp_path):
    """A failed model leaves neither a valid guess cache nor a completed report."""
    cfg = config(tmp_path, "react-ot")
    pipeline.run(cfg)
    updated = cfg.model_dump()
    updated["reaction"]["ts_guesser"]["command"] = [sys.executable, "-c", "raise SystemExit(4)"]
    cfg = Config.model_validate(updated)
    with pytest.raises(JobFailureError, match="failed"):
        pipeline.run(cfg)
    assert not (cfg.output_dir / "guesser" / "guess.json").exists()
    assert not (cfg.output_dir / "reaction.json").exists()


@pytest.mark.parametrize(
    "raw",
    [
        {"method": "fragmentflow"},
        {"method": "external"},
        {"method": "idpp"},
        {"method": "react-ot", "command": []},
        {"method": "idpp", "command": ["echo"]},
        {"method": "idpp", "scoring_steps": [{"step": 1, "engine": "fake", "operation": "opt"}]},
    ],
)
def test_invalid_guesser(raw):
    """Do not silently replace missing model integrations or unscored paths."""
    with pytest.raises(ValidationError):
        TSGuesserConfig.model_validate(raw)


def test_relative_paths_and_registry_validation(tmp_path):
    """Nested endpoint/model paths resolve beside the YAML, and all branches are checked."""
    cfg = config(tmp_path)
    raw = cfg.model_dump(mode="json")
    raw["reaction"]["reactant"]["input"] = "r.xyz"
    raw["reaction"]["ts_steps"][0]["options"] = {"model_path": "weights.model"}
    path = tmp_path / "reaction.yaml"
    path.write_text(json.dumps(raw))
    loaded = load_config(path)
    assert loaded.reaction.reactant.input == tmp_path / "r.xyz"
    assert loaded.reaction.ts_steps[0].options["model_path"] == str(tmp_path / "weights.model")
    raw["reaction"]["product"]["steps"][0]["engine"] = "missing-engine"
    report = validate_config_text(json.dumps(raw), base_dir=tmp_path)
    assert not report.ok
    assert report.issues[0].loc[:2] == ("reaction", "product")


def test_reject_mixed_workflow(tmp_path):
    """Existing top-level steps must never silently disappear into a reaction."""
    raw = config(tmp_path).model_dump()
    raw["steps"] = [{"step": 1, "engine": "fake"}]
    with pytest.raises(ValidationError, match="omit top-level"):
        Config.model_validate(raw)


def test_scoped_action_refused_before_mutation(tmp_path):
    """Unimplemented reaction recovery modes must not touch branch caches."""
    cfg = config(tmp_path)
    with pytest.raises(ConfigError, match="whole-workflow"):
        recovery.execute(cfg, recovery.Action.REBUILD_CACHE)
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize("kind", ["periodic", "nonfinite", "empty"])
def test_reject_invalid_endpoint_geometry(kind):
    """Periodicity, nonfinite coordinates and absent atoms are not molecular TS inputs."""
    atoms = Atoms("H")
    if kind == "periodic":
        atoms.pbc = True
    elif kind == "nonfinite":
        atoms.positions[:] = np.nan
    else:
        atoms = Atoms()
    with pytest.raises(ConfigError):
        reaction.check_pair(atoms, atoms)


def test_reject_ambiguous_endpoint():
    """An endpoint ensemble needs explicit conformer selection before guessing."""
    from chemrefine.state import PipelineState

    with pytest.raises(ConfigError, match="exactly one"):
        reaction._one(PipelineState(structures=()), "reactant")


@pytest.mark.parametrize("count", [3, 36])
def test_geodesic_adapter_dispatch_and_endpoint_preservation(monkeypatch, count):
    """Both geodesic algorithms consume the mapped endpoints without changing their inputs."""
    from types import SimpleNamespace

    calls = []
    smoother = SimpleNamespace(
        path=None,
        smooth=lambda **kw: calls.append("smooth"),
        sweep=lambda **kw: calls.append("sweep"),
    )

    def redistribute(symbols, endpoints, n_images, **kw):
        return np.linspace(endpoints[0], endpoints[-1], n_images)

    def geodesic(symbols, raw, *args, **kw):
        smoother.path = raw
        return smoother

    monkeypatch.setitem(
        sys.modules, "geodesic_interpolate.geodesic", SimpleNamespace(Geodesic=geodesic)
    )
    monkeypatch.setitem(
        sys.modules,
        "geodesic_interpolate.interpolation",
        SimpleNamespace(redistribute=redistribute),
    )
    r = Atoms("H" * count, positions=np.random.default_rng(42).normal(size=(count, 3)))
    p = r.copy()
    p.positions[0, 0] += 0.1
    original_r, original_p = r.positions.copy(), p.positions.copy()
    guesser = TSGuesserConfig(
        method="geodesic",
        n_images=5,
        scoring_steps=[{"step": 1, "engine": "fake", "operation": "sp"}],
    )
    images = reaction.interpolate(r, p, guesser)
    assert calls == (["smooth"] if count == 3 else ["sweep"])
    assert len(images) == 5
    np.testing.assert_array_equal(r.positions, original_r)
    np.testing.assert_array_equal(p.positions, original_p)
    np.testing.assert_allclose(images[0].positions, original_r)


def test_missing_geodesic_dependency(monkeypatch):
    """An absent optional package names its install extra instead of a traceback."""
    monkeypatch.setitem(sys.modules, "geodesic_interpolate.geodesic", None)
    guesser = TSGuesserConfig(
        method="geodesic",
        scoring_steps=[
            {"step": 1, "engine": "fake", "operation": "sp"},
        ],
    )
    with pytest.raises(ConfigError, match="chemrefine\\[reaction\\]"):
        reaction.interpolate(Atoms("H"), Atoms("H"), guesser)


def test_nonreaction_and_unsupported_plan(tmp_path):
    """The coordinator does not interpret a linear workflow or a scoped recovery plan."""
    from chemrefine.step import RunPlan, StepMode

    linear = Config(steps=[{"step": 1, "engine": "fake"}])
    with pytest.raises(ConfigError, match="no reaction"):
        reaction.branches(linear)
    with pytest.raises(ConfigError, match="no reaction"):
        reaction.generate_candidates(linear, Atoms("H"), Atoms("H"), resume=False)
    with pytest.raises(ConfigError, match="not an interpolation"):
        reaction.interpolate(
            Atoms("H"), Atoms("H"), TSGuesserConfig(method="react-ot", command=["model"])
        )
    with pytest.raises(ConfigError, match="whole-workflow"):
        reaction.run(config(tmp_path), RunPlan(default=StepMode.CACHE_ONLY))


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (["{typo}"], "placeholder"),
        ([sys.executable, "-c", "pass"], "valid candidates"),
        (
            [sys.executable, "-c", "open(__import__('sys').argv[1],'w').close()", "{output}"],
            "no TS candidates",
        ),
    ],
)
def test_invalid_model_output(tmp_path, command, message):
    """A successful process must still produce a nonempty, usable XYZ artifact."""
    raw = config(tmp_path, "react-ot").model_dump()
    raw["reaction"]["ts_guesser"]["command"] = command
    cfg = Config.model_validate(raw)
    r = pipeline.bootstrap(reaction.branches(cfg)["reactant"]).structures[0].atoms
    p = pipeline.bootstrap(reaction.branches(cfg)["product"]).structures[0].atoms
    with pytest.raises((ConfigError, JobFailureError), match=message):
        reaction.generate_candidates(cfg, r, p, resume=False)


@pytest.mark.parametrize("failure", ["missing_energy", "missing_image", "empty_ts"])
def test_incomplete_calculations_cannot_report_success(tmp_path, monkeypatch, failure):
    """Missing path energies/images or empty TS results must fail the reaction."""
    from chemrefine.state import PipelineState, Structure
    from chemrefine.step import StepOutcome

    raw = config(tmp_path).model_dump()
    raw["reaction"]["reactant"]["steps"] = []
    raw["reaction"]["product"]["steps"] = []
    cfg = Config.model_validate(raw)

    def run_branch(branch, plan):
        frames = io.read_xyz_frames(branch.input)
        structs = tuple(
            Structure(
                id=str(i),
                atoms=a,
                energy_hartree=(None if failure == "missing_energy" else float(i)),
            )
            for i, a in enumerate(frames)
        )
        if failure == "missing_image" or branch.output_dir.name == "ts":
            structs = ()
        return [StepOutcome(state=PipelineState(structures=structs), cache_hit=False)]

    monkeypatch.setattr(pipeline, "run", run_branch)
    with pytest.raises(JobFailureError, match=r"finite single-point|no surviving"):
        reaction.run(cfg)
    assert not (cfg.output_dir / "reaction.json").exists()


def test_guesser_uses_prepared_minima_before_ts_optimization(tmp_path, monkeypatch):
    """The guesser receives the final endpoint geometries, followed by scoring and TS work."""
    from dataclasses import replace

    from fake_engine import FakeEngine

    cfg = config(tmp_path)
    original_parse = FakeEngine.parse
    original_generate = reaction.generate_candidates
    events = []
    expected = {}
    for name, branch in reaction.branches(cfg).items():
        if name in ("reactant", "product"):
            atoms = pipeline.bootstrap(branch).structures[0].atoms.copy()
            atoms.positions[1, 0] += 0.05 if name == "reactant" else -0.05
            expected[name] = atoms.positions.copy()

    def parse_prepared(self, inputs, ctx):
        name = ctx.step_dir.parent.name
        events.append(name)
        result = original_parse(self, inputs, ctx)
        if name in expected:
            prepared = []
            for struct in result.structures:
                atoms = struct.atoms.copy()
                atoms.positions[:] = expected[name]
                prepared.append(replace(struct, atoms=atoms))
            result = replace(result, structures=tuple(prepared))
        return result

    def generate_from_minima(config, reactant, product, *, resume):
        events.append("guess")
        np.testing.assert_allclose(reactant.positions, expected["reactant"])
        np.testing.assert_allclose(product.positions, expected["product"])
        return original_generate(config, reactant, product, resume=resume)

    monkeypatch.setattr(FakeEngine, "parse", parse_prepared)
    monkeypatch.setattr(reaction, "generate_candidates", generate_from_minima)
    pipeline.run(cfg)
    assert events == ["reactant", "product", "guess", "scoring", "scoring", "scoring", "ts"]


def test_linker_provenance_survives_command_work_directory_cleanup(tmp_path):
    """Model metadata is copied beside the persistent guess artifacts before temp cleanup."""
    cfg = config(tmp_path, "react-ot")
    raw = cfg.model_dump()
    raw["reaction"]["ts_guesser"]["command"] = [
        sys.executable,
        "-c",
        "import pathlib,shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2]); "
        "pathlib.Path(sys.argv[2]+'.json').write_text('model provenance')",
        "{reactant}",
        "{output}",
    ]
    cfg = Config.model_validate(raw)
    pipeline.run(cfg)
    assert (cfg.output_dir / "guesser" / "model.json").read_text() == "model provenance"
    raw["reaction"]["ts_guesser"]["command"] = [
        sys.executable,
        "-c",
        "import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2])",
        "{reactant}",
        "{output}",
    ]
    pipeline.run(Config.model_validate(raw))
    assert not (cfg.output_dir / "guesser" / "model.json").exists()


@pytest.mark.parametrize("failure", ["numbering", "names", "continue_scoring", "both_modes"])
def test_reaction_config_rejects_ambiguous_workflows(tmp_path, failure):
    cfg = config(tmp_path)
    raw = cfg.model_dump(by_alias=True)
    if failure == "numbering":
        raw["reaction"]["reactant"]["steps"][0]["step"] = 2
    elif failure == "names":
        raw["reaction"]["reactant"]["steps"] = [
            {"step": i, "engine": "fake", "name": "repeated"} for i in (1, 2)
        ]
    elif failure == "continue_scoring":
        raw["reaction"]["ts_guesser"]["scoring_steps"][0]["on_failure"] = "skip"
    else:
        from test_mechanism import configuration

        raw["mechanism"] = configuration(tmp_path).model_dump(by_alias=True)["mechanism"]
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_missing_endpoint_xyz_is_a_validation_warning(tmp_path):
    import yaml

    raw = config(tmp_path).model_dump(mode="json", by_alias=True)
    raw["reaction"]["product"]["input"] = str(tmp_path / "missing.xyz")
    report = validate_config_text(yaml.safe_dump(raw), base_dir=tmp_path)
    assert report.ok
    assert any(
        w.loc == ("reaction", "product", "input") and w.kind == "input" for w in report.warnings
    )
