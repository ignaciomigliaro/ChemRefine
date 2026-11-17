"""Opt-in real UMA/ORCA energy smoke test for the complete two-stage core workflow."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from chemrefine import mechanism
from chemrefine.config import Config, load_config
from chemrefine.engines.api import get_engine
from chemrefine.step import RunPlan, StepMode


@pytest.mark.integration
@pytest.mark.parametrize("example_name", ["product_core", "product_core_from_start"])
def test_uma_extopt_core_workflow_energy_smoke(tmp_path, monkeypatch, example_name):
    """Evaluate every branch with real tools; this does not certify minima or TSs."""
    names = ["CHEMREFINE_CORE_UMA_PYTHON", "CHEMREFINE_CORE_UMA_MODEL", "CHEMREFINE_CORE_ORCA"]
    if any(not os.environ.get(name) for name in names):
        message = "Set " + ", ".join(names) + " for the real product-core smoke test"
        if os.environ.get("CHEMREFINE_REQUIRE_LIVE"):
            pytest.fail(message)
        pytest.skip(message)
    python, model, orca = [os.environ[name] for name in names]
    example = Path(__file__).resolve().parents[1] / "examples" / example_name / "input.yaml"
    data = load_config(example).model_dump()
    data.update(output_dir=tmp_path / "outputs", executables={"orca": orca})
    options = {
        "task_name": "omol",
        "model_name": "uma-s-1p2",
        "model_path": model,
        "backend_python": python,
        "device": "cpu",
        "gradient_timeout_seconds": 600,
    }
    step = {
        "step": 1,
        "name": "energy",
        "engine": "mlip-extopt",
        "operation": "sp",
        "template": "single_point.inp",
        "options": options,
    }
    for stage in data["mechanism"]["stages"]:
        for branch in ("reactant", "product"):
            stage[branch]["steps"] = [step]
        stage["ts_steps"] = [step]
        stage["ts_guesser"].update(n_images=3, scoring_steps=[step])
    cfg = Config.model_validate(data)
    outcomes = mechanism.run(cfg)
    assert len(outcomes) == 8
    for outcome in outcomes:
        assert outcome.state.structures
        for structure in outcome.state.structures:
            assert structure.energy_hartree is not None and np.isfinite(structure.energy_hartree)
            assert len(structure.atoms) == 22
    report = cfg.output_dir / "final/manifest.json"
    before = report.read_bytes()
    summary = json.loads((cfg.output_dir / "mechanism.json").read_text())
    assert summary["verified_transition_state"] is False
    assert len(json.loads(before)["geometries"]) == 6
    engine = get_engine("mlip-extopt")
    monkeypatch.setattr(type(engine), "submit", lambda *a: pytest.fail("resume resubmitted a job"))
    mechanism.run(cfg, RunPlan(default=StepMode.RESUME))
    assert report.read_bytes() == before
