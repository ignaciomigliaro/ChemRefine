"""Directory variants preserve independent chemistry, resource limits and failure evidence."""

import json
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError
from test_core_workflow import configuration

from chemrefine import mechanism, mechanism_batch
from chemrefine.config import (
    Config,
    MechanismSource,
    MechanismVariantOverride,
    load_config,
    resolve_relative_paths,
)
from chemrefine.errors import ConfigError, JobFailureError
from chemrefine.step import RunPlan, StepMode


def batch_config(tmp_path, *, parallel=2):
    """Use real small metal XYZ/core inputs, with two alternative starting geometries."""
    cfg = configuration(tmp_path)
    directory = tmp_path / "catalysts"
    directory.mkdir()
    original = tmp_path / "reactant.xyz"
    for name in ["CuI2.XYZ", "CuBr2.xyz"]:
        (directory / name).write_bytes(original.read_bytes())
    (directory / "notes.txt").write_text("not a structure")
    (directory / "ignore.xyz").mkdir()
    stage = cfg.mechanism.stages[0]
    source = stage.reactant.sources[0].model_copy(update={"input": None, "input_dir": directory})
    stage = stage.model_copy(
        update={"reactant": stage.reactant.model_copy(update={"sources": [source]})}
    )
    return resolve_relative_paths(
        cfg.model_copy(
            update={
                "max_cores": 4,
                "max_gpus": 0,
                "mechanism": cfg.mechanism.model_copy(
                    update={"stages": [stage, cfg.mechanism.stages[1]], "parallel_runs": parallel}
                ),
            }
        ),
        base=tmp_path,
    )


def test_directory_expansion_is_sorted_isolated_and_preserves_single_sources(tmp_path):
    cfg = batch_config(tmp_path)
    variants = mechanism_batch.expand(cfg)
    assert [v.name for v in variants] == ["CuBr2", "CuI2"]
    assert mechanism_batch.enabled(cfg)
    assert not mechanism_batch.enabled(Config(steps=[{"step": 1, "engine": "fake"}]))
    assert not mechanism_batch.enabled(variants[0].config)
    assert [v.config.max_cores for v in variants] == [2, 2]
    assert all(v.config.max_gpus == 0 for v in variants)
    plans = mechanism.plan(cfg)
    assert len(plans) == 4
    for variant in variants:
        assert variant.config.output_dir == cfg.output_dir / "variants" / variant.name
        source = variant.config.mechanism.stages[0].reactant.sources[0]
        assert source.input_dir is None and source.input == variant.sources[0]
        assert variant.config.mechanism.stages[1] == cfg.mechanism.stages[1]
    assert cfg.mechanism.stages[0].reactant.sources[0].input is None


def test_single_master_applies_the_explicit_force_exception_to_one_catalyst_only(tmp_path):
    cfg = batch_config(tmp_path)
    cfg = cfg.model_copy(
        update={
            "mechanism": cfg.mechanism.model_copy(
                update={
                    "force": True,
                    "variant_overrides": {
                        "CuI2": MechanismVariantOverride(force=False, force_endpoint_contacts=True)
                    },
                }
            )
        }
    )
    bromide, iodide = mechanism_batch.expand(cfg)
    assert bromide.config.mechanism.force is True
    assert bromide.config.mechanism.force_endpoint_contacts is False
    assert iodide.config.mechanism.force is False
    assert iodide.config.mechanism.force_endpoint_contacts is True
    assert not iodide.config.mechanism.variant_overrides
    assert not bromide.config.mechanism.variant_overrides
    assert cfg.mechanism.variant_overrides["CuI2"].force_endpoint_contacts is True
    bad = cfg.model_copy(
        update={
            "mechanism": cfg.mechanism.model_copy(
                update={"variant_overrides": {"unknown": MechanismVariantOverride(force=True)}}
            )
        }
    )
    with pytest.raises(ConfigError, match="unknown variants"):
        mechanism_batch.expand(bad)


def test_variant_exception_requires_a_directory_and_cannot_change_unrelated_settings(tmp_path):
    cfg = configuration(tmp_path).model_dump()
    cfg["mechanism"]["variant_overrides"] = {"CuI2": {"force_endpoint_contacts": True}}
    with pytest.raises(ValidationError, match="requires a directory"):
        Config.model_validate(cfg)
    with pytest.raises(ValidationError):
        MechanismVariantOverride.model_validate({"charge": 5})


def test_multiple_directory_slots_expand_cartesian_product_with_stable_names(tmp_path):
    cfg = batch_config(tmp_path)
    stage = cfg.mechanism.stages[0]
    source = stage.reactant.sources[0]
    updated = stage.model_copy(
        update={"reactant": stage.reactant.model_copy(update={"sources": [source, source]})}
    )
    cfg = cfg.model_copy(
        update={"mechanism": cfg.mechanism.model_copy(update={"stages": [updated]})}
    )
    variants = mechanism_batch.expand(cfg)
    assert [v.name for v in variants] == [
        "s1_CuBr2__s2_CuBr2",
        "s1_CuBr2__s2_CuI2",
        "s1_CuI2__s2_CuBr2",
        "s1_CuI2__s2_CuI2",
    ]
    assert all(len(v.sources) == 2 for v in variants)
    for variant in variants:
        assert [s.input for s in variant.config.mechanism.stages[0].reactant.sources] == list(
            variant.sources
        )


@pytest.mark.parametrize("mode", ["empty", "unsafe", "duplicate", "limit", "gpu_auto"])
def test_expansion_refuses_invalid_or_unbounded_inputs(tmp_path, monkeypatch, mode):
    cfg = batch_config(tmp_path)
    directory = cfg.mechanism.stages[0].reactant.sources[0].input_dir
    if mode == "empty":
        for p in directory.iterdir():
            if p.is_file():
                p.unlink()
    elif mode == "unsafe":
        (directory / "bad name.xyz").write_text("bad")
    elif mode == "duplicate":
        entries = mechanism_batch.io.tree_entries(directory)
        monkeypatch.setattr(
            mechanism_batch.io,
            "tree_entries",
            lambda _: [*entries, (directory / "CuI2.xyz", False)],
        )
    elif mode == "limit":
        cfg = cfg.model_copy(
            update={"mechanism": cfg.mechanism.model_copy(update={"max_variants": 1})}
        )
    else:
        cfg = cfg.model_copy(update={"max_gpus": None})
    with pytest.raises(ConfigError):
        mechanism_batch.expand(cfg)


def test_nonbatch_expansion_requires_a_directory(tmp_path):
    with pytest.raises(ConfigError, match="require a mechanism"):
        mechanism_batch.expand(Config(steps=[{"step": 1, "engine": "fake"}]))
    cfg = configuration(tmp_path)
    with pytest.raises(ConfigError, match="no reactant input_dir"):
        mechanism_batch.expand(cfg)


def test_cartesian_names_cannot_alias_distinct_input_combinations(tmp_path):
    cfg = batch_config(tmp_path)
    first = tmp_path / "first_directory"
    second = tmp_path / "second_directory"
    first.mkdir()
    second.mkdir()
    for directory, names in [(first, ["A", "A__s2_B"]), (second, ["C", "B__s2_C"])]:
        for name in names:
            (directory / f"{name}.xyz").write_text("unused geometry")
    stage = cfg.mechanism.stages[0]
    source = stage.reactant.sources[0]
    sources = [
        source.model_copy(update={"input_dir": first}),
        source.model_copy(update={"input_dir": second}),
    ]
    stage = stage.model_copy(
        update={"reactant": stage.reactant.model_copy(update={"sources": sources})}
    )
    cfg = cfg.model_copy(update={"mechanism": cfg.mechanism.model_copy(update={"stages": [stage]})})
    with pytest.raises(ConfigError, match="colliding names"):
        mechanism_batch.expand(cfg)


def test_single_worker_and_explicit_gpu_budgets_are_not_oversubscribed(tmp_path):
    cfg = batch_config(tmp_path, parallel=8).model_copy(update={"max_cores": 1, "max_gpus": None})
    variants = mechanism_batch.expand(cfg)
    assert all(v.config.max_cores == 1 and v.config.max_gpus is None for v in variants)
    cfg = cfg.model_copy(update={"max_cores": 6, "max_gpus": 3})
    variants = mechanism_batch.expand(cfg)
    assert sum(v.config.max_cores for v in variants) == 6
    assert sum(v.config.max_gpus or 0 for v in variants) <= 3


@pytest.mark.parametrize("mode,action", [(StepMode.EXECUTE, "run"), (StepMode.RESUME, "resume")])
def test_worker_uses_isolated_cli_and_roundtrippable_config(tmp_path, monkeypatch, mode, action):
    cfg = batch_config(tmp_path)
    variant = mechanism_batch.expand(cfg)[0]

    def execute(command, **kwargs):
        assert command[1:4] == ["-m", "chemrefine", action]
        reloaded = load_config(command[4])
        assert reloaded == variant.config
        assert kwargs["stderr"] == subprocess.STDOUT and kwargs["check"] is False
        kwargs["stdout"].write("native child log\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(mechanism_batch.subprocess, "run", execute)
    record = mechanism_batch._execute(variant, mode)
    assert record == {"name": "CuBr2", "exit_code": 0, "status": "refinement_completed"}
    assert (variant.config.output_dir / "driver.log").read_text() == "native child log\n"


def test_parallel_batch_finishes_other_variants_and_reports_failed_worker(tmp_path, monkeypatch):
    cfg = batch_config(tmp_path)
    barrier = threading.Barrier(2)
    calls = []

    def execute(command, **kwargs):
        child = load_config(command[4])
        name = child.output_dir.name
        calls.append(name)
        barrier.wait(timeout=5)
        time.sleep(0.01)
        return SimpleNamespace(returncode=5 if name == "CuBr2" else 0)

    monkeypatch.setattr(mechanism_batch.subprocess, "run", execute)
    with pytest.raises(JobFailureError, match="variants failed"):
        mechanism.run(cfg)
    assert sorted(calls) == ["CuBr2", "CuI2"]
    report = json.loads((cfg.output_dir / "batch.json").read_text())
    assert report["status"] == "finished" and report["verified_transition_states"] is False
    assert [v["status"] for v in report["variants"]] == ["failed", "refinement_completed"]
    assert all(len(v["sources"][0]["sha256"]) == 64 for v in report["variants"])
    page = (cfg.output_dir / "index.html").read_text()
    assert "variants/CuBr2/final/index.html" not in page
    assert "variants/CuI2/final/index.html" in page


def test_batch_success_returns_without_running_the_unexpanded_parent(tmp_path, monkeypatch):
    cfg = batch_config(tmp_path)
    monkeypatch.setattr(
        mechanism_batch,
        "_execute",
        lambda v, mode: {"name": v.name, "exit_code": 0, "status": "refinement_completed"},
    )
    assert mechanism.run(cfg, RunPlan(default=StepMode.RESUME)) == []
    assert not (cfg.output_dir / "final").exists()


def test_every_variant_is_planned_before_any_child_process_starts(tmp_path, monkeypatch):
    cfg = batch_config(tmp_path)
    source = cfg.mechanism.stages[0].reactant.sources[0].input_dir / "CuI2.XYZ"
    source.write_text("not an XYZ")
    monkeypatch.setattr(
        mechanism_batch, "_execute", lambda *_: pytest.fail("submitted before preflight")
    )
    with pytest.raises(ConfigError):
        mechanism.run(cfg)
    assert not cfg.output_dir.exists()


@pytest.mark.parametrize(
    "source",
    [
        {"input_dir": "catalysts", "input": "a.xyz"},
        {"input_dir": "catalysts", "smiles": "O"},
        {"input_dir": "catalysts", "from": "earlier"},
        {"input_dir": "catalysts", "select": "metal_complex"},
    ],
)
def test_directory_source_is_an_alternative_not_an_implicit_mixture(source):
    with pytest.raises(ValidationError):
        MechanismSource.model_validate(source)


def test_relative_directory_is_anchored_by_the_yaml_not_the_current_directory(tmp_path):
    cfg = batch_config(tmp_path)
    raw = cfg.model_dump(mode="json")
    raw["mechanism"]["stages"][0]["reactant"]["sources"][0]["input_dir"] = "catalysts"
    path = tmp_path / "batch.yaml"
    path.write_text(yaml.safe_dump(raw))
    loaded = load_config(path)
    assert loaded.mechanism.stages[0].reactant.sources[0].input_dir == tmp_path / "catalysts"


def test_cli_dry_run_lists_every_core_variant_without_touching_outputs(tmp_path):
    from typer.testing import CliRunner

    from chemrefine.cli import app

    cfg = batch_config(tmp_path)
    path = tmp_path / "input.yaml"
    path.write_text(cfg.model_dump_json())
    result = CliRunner().invoke(app, ["run", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "variants/CuBr2/" in result.output and "variants/CuI2/" in result.output
    assert result.output.count("stage_output_dir=") == 4
    assert "mapped_reaction=" in result.output
    assert not cfg.output_dir.exists()
