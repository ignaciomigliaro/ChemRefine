"""Opt-in CPU smoke tests against installed upstream models and real checkpoints."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write


@pytest.mark.integration
@pytest.mark.parametrize("model", ["react-ot", "fragmentflow"])
def test_pretrained_ts_linker(tmp_path, model):
    """Exercise a real loader/sampler; this smoke check does not establish TS accuracy."""
    prefix = "CHEMREFINE_" + model.upper().replace("-", "_")
    names = [prefix + suffix for suffix in ("_PYTHON", "_REPO", "_CHECKPOINT")]
    if any(not os.environ.get(name) for name in names):
        pytest.skip("Set " + ", ".join(names) + " for pretrained-model inference")
    interpreter, repo, checkpoint = [os.environ[name] for name in names]
    reactant, product = tmp_path / "r.xyz", tmp_path / "p.xyz"
    # Formaldehyde -> hydroxymethylene: mapped neutral-singlet endpoints.
    write(
        reactant,
        Atoms(
            "COHH",
            positions=[
                [0, 0, 0],
                [1.21, 0, 0],
                [-0.6, 0.94, 0],
                [-0.6, -0.94, 0],
            ],
        ),
    )
    write(
        product,
        Atoms(
            "COHH",
            positions=[
                [0, 0, 0],
                [1.3, 0, 0],
                [-0.8, 0.8, 0],
                [1.7, 0.85, 0],
            ],
        ),
    )
    output = tmp_path / "ts.xyz"
    script = Path(__file__).resolve().parents[1] / "scripts" / "ts_guess.py"
    command = [
        interpreter,
        str(script),
        "--model",
        model,
        "--repo",
        repo,
        "--checkpoint",
        checkpoint,
        "--reactant",
        str(reactant),
        "--product",
        str(product),
        "--output",
        str(output),
        "--device",
        "cpu",
        "--nfe",
        "10",
    ]
    if model == "fragmentflow":
        command += ["--full-system"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    ts = read(output)
    assert ts.get_chemical_symbols() == ["C", "O", "H", "H"]
    assert np.isfinite(ts.positions).all()
    metadata = json.loads(output.with_suffix(".xyz.json").read_text())
    assert metadata["model"] == model
    assert metadata["verified_transition_state"] is False
