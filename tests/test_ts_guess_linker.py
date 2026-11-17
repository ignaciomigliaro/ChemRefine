"""Linker input/output contracts and real tensor layout without pretrained model dependencies."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ts_guess.py"
spec = importlib.util.spec_from_file_location("ts_guess_linker", SCRIPT)
assert spec is not None and spec.loader is not None
linker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(linker)


def arguments(tmp_path, model="react-ot", extra=()):
    """Mapped neutral endpoints and a checkpoint path whose contents need not be unpickled."""
    r = tmp_path / "r.xyz"
    p = tmp_path / "p.xyz"
    write(r, Atoms("CHNO", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    write(p, Atoms("CHNO", positions=[[0, 0, 0], [1.2, 0, 0], [0, 1.1, 0], [0, 0, 1]]))
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"contract test; not a trained checkpoint")
    return linker.parser().parse_args(
        [
            "--model",
            model,
            "--reactant",
            str(r),
            "--product",
            str(p),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "ts.xyz"),
            *extra,
        ]
    )


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("charge", 1, "neutral singlet"),
        ("multiplicity", 3, "neutral singlet"),
        ("nfe", 0, "nfe"),
        ("epsilon", -1, "epsilon"),
        ("idpp_images", 2, "idpp-images"),
        ("seed", -1, "seed"),
        ("atol", np.nan, "tolerances"),
        ("rtol", 0, "tolerances"),
        ("core_indices", "0,1", "only to FragmentFlow"),
    ],
)
def test_invalid_model_inputs(tmp_path, flag, value, message):
    args = arguments(tmp_path)
    setattr(args, flag, value)
    with pytest.raises(ValueError, match=message):
        linker.read_endpoints(args)


def test_endpoint_mapping_and_alignment(tmp_path):
    args = arguments(tmp_path)
    original = args.product.read_bytes()
    r, p = linker.read_endpoints(args)
    np.testing.assert_array_equal(r.numbers, [6, 1, 7, 8])
    np.testing.assert_array_equal(p.numbers, r.numbers)
    np.testing.assert_allclose(r.positions.mean(0), 0, atol=1e-15)
    np.testing.assert_allclose(p.positions.mean(0), 0, atol=1e-15)
    assert args.product.read_bytes() == original


@pytest.mark.parametrize(
    "failure", ["missing_checkpoint", "overwrite", "ensemble", "mapping", "element", "nan"]
)
def test_unusable_files_refused_before_model_loading(tmp_path, failure):
    args = arguments(tmp_path)
    if failure == "missing_checkpoint":
        args.checkpoint.unlink()
    elif failure == "overwrite":
        args.output = args.reactant
    elif failure == "ensemble":
        args.reactant.write_bytes(args.reactant.read_bytes() * 2)
    elif failure == "mapping":
        write(args.product, Atoms("NHCO"))
    elif failure == "element":
        write(args.product, Atoms("Cl"))
    else:
        write(args.product, Atoms("CHNO", positions=np.full((4, 3), np.nan)))
    with pytest.raises(ValueError):
        linker.read_endpoints(args)


@pytest.mark.parametrize("model", ["react-ot", "fragmentflow"])
def test_upstream_checkpoint_loaders_and_solver_options(tmp_path, monkeypatch, model):
    pytest.importorskip("torch")
    args = arguments(tmp_path, model, extra=["--solver", "ei", "--order", "3", "--normalize"])
    ddpm = SimpleNamespace(
        mapping="R+P->TS", mapping_initial="RP", idx=1, T=3000, eval=lambda: None
    )
    seen = []
    wrapper = SimpleNamespace(ddpm=ddpm)
    wrapper.eval = lambda: wrapper
    wrapper.to = lambda device: wrapper

    def load(*a, **kw):
        seen.append((a, kw))
        return wrapper

    module = SimpleNamespace(
        SBModule=SimpleNamespace(load_from_checkpoint=load), load_pytorch_checkpoint=load
    )
    monkeypatch.setattr(linker, "upstream_module", lambda *a: module)
    assert linker.load_model(args) is ddpm
    assert str(args.checkpoint.resolve()) in str(seen)
    assert ddpm.opt.order == 3 and ddpm.opt.normalize is True
    assert ddpm.opt.diz == "linear"
    args.nfe = 3000
    with pytest.raises(ValueError, match="timesteps"):
        linker.load_model(args)
    args.nfe = 100
    ddpm.mapping = "R->P"
    with pytest.raises(ValueError, match="predict TS"):
        linker.load_model(args)


class ContractSampler:
    """Assert the upstream sampler's actual tensor contract while returning known coordinates."""

    def sample_batch(self, batch, conditions, *, return_timesteps, training):
        import torch

        assert not torch.is_grad_enabled()
        assert not return_timesteps and not training
        assert conditions.shape == (1, 1)
        for fragment in batch:
            assert fragment["pos"].shape[1] == 3
            assert fragment["one_hot"].shape[1] == 5
            assert fragment["charge"].shape[1] == 1
            assert fragment["size"].tolist() == [len(fragment["pos"])]
            assert fragment["mask"].sum().item() == 0
            torch.testing.assert_close(fragment["pos"].mean(0), torch.zeros(3), atol=1e-7, rtol=0)
        self.initial = (batch[0]["pos"] + batch[2]["pos"]) / 2
        return None, self.initial, None, None, None

    def sample(self, initial, batch, conditions, *, nfe, ot_ode):
        import torch

        assert not torch.is_grad_enabled()
        assert ot_ode is True and nfe == 100
        torch.testing.assert_close(initial, self.initial)
        positions = initial.clone()
        positions[0, 0] += 0.2
        # A deliberately different last trajectory frame catches wrong frame extraction.
        return torch.stack([positions, positions + 10], dim=1), None


def test_real_tensor_sampling_and_atomic_output(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    args = arguments(tmp_path)
    monkeypatch.setattr(linker, "load_model", lambda args: ContractSampler())
    assert (
        linker.main(
            [
                "--model",
                args.model,
                "--reactant",
                str(args.reactant),
                "--product",
                str(args.product),
                "--checkpoint",
                str(args.checkpoint),
                "--output",
                str(args.output),
            ]
        )
        == 0
    )
    atoms = read(args.output)
    r, p = linker.read_endpoints(args)
    expected = (r.positions + p.positions) / 2
    expected[0, 0] += 0.2
    np.testing.assert_allclose(atoms.positions, expected, atol=1e-7)
    np.testing.assert_array_equal(atoms.numbers, r.numbers)
    report = json.loads(args.output.with_suffix(".xyz.json").read_text())
    assert report["checkpoint_sha256"]
    assert report["core_indices"] == [0, 1, 2, 3]
    assert report["verified_transition_state"] is False


def test_fragmentflow_predicts_core_then_reconstructs_full_atom_order(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    args = arguments(tmp_path, "fragmentflow", extra=["--core-indices", "2,0"])
    r, p = linker.read_endpoints(args)
    seen = []

    def complete(core, rn, rp, pn, pp, kept, *, num_idpp_images):
        assert kept == [0, 2]
        assert num_idpp_images == 11
        assert core.shape == (2, 3)
        np.testing.assert_array_equal(rn, r.numbers)
        np.testing.assert_array_equal(pn, p.numbers)
        positions = rp.copy()
        positions[kept] = core
        seen.append(kept)
        return rn, positions, 0, rn[kept], rp[kept]

    monkeypatch.setattr(
        linker,
        "upstream_module",
        lambda *a: SimpleNamespace(
            complete_fragmented_core_from_xyz=complete,
        ),
    )
    atoms, indices = linker.fragmentflow_ts(ContractSampler(), r, p, args)
    assert seen == [[0, 2]] and indices == [0, 2]
    assert len(atoms) == 4
    np.testing.assert_array_equal(atoms.numbers, r.numbers)
    # Substituents survive in their original positions, never as a truncated core XYZ.
    np.testing.assert_allclose(atoms.positions[[1, 3]], r.positions[[1, 3]])


@pytest.mark.parametrize("indices", ["0,0", "-1,2", "0,12", "one,two", ""])
def test_invalid_explicit_core_indices(tmp_path, indices):
    args = arguments(tmp_path, "fragmentflow", extra=["--core-indices", indices])
    r, p = linker.read_endpoints(args)
    with pytest.raises(ValueError, match="indices"):
        linker.core_indices(args, r, p)


@pytest.mark.parametrize("selection", ["core", "all", "failed"])
def test_upstream_automatic_fragmentation(tmp_path, monkeypatch, selection):
    args = arguments(tmp_path, "fragmentflow")
    r, p = linker.read_endpoints(args)
    seen = []

    def process(rn, rp, pn, pp, tn, tp, mapper, eps):
        assert eps == [0] and mapper == "mapper"
        np.testing.assert_array_equal(rn, pn)
        np.testing.assert_array_equal(tn, rn)
        seen.append(True)
        return {
            "error": "could not map reaction" if selection == "failed" else None,
            "epsilon_results": {0: {"backbone_indices": {0, 2}}} if selection == "core" else {},
        }

    module = SimpleNamespace(process_single_reaction=process, localmapper=lambda **kw: "mapper")
    monkeypatch.setattr(linker, "upstream_module", lambda *a: module)
    if selection == "failed":
        with pytest.raises(ValueError, match="could not map"):
            linker.core_indices(args, r, p)
    else:
        assert linker.core_indices(args, r, p) == ([0, 2] if selection == "core" else [0, 1, 2, 3])
    assert seen == [True]


def test_missing_optional_mapping_dependency_has_workaround(tmp_path, monkeypatch):
    args = arguments(tmp_path, "fragmentflow")
    r, p = linker.read_endpoints(args)

    def missing(*args):
        raise ValueError("No module named localmapper")

    monkeypatch.setattr(linker, "upstream_module", missing)
    with pytest.raises(ValueError, match="core-indices"):
        linker.core_indices(args, r, p)


def test_missing_model_environment_and_wrong_repo(tmp_path):
    args = arguments(tmp_path)
    args.repo = tmp_path
    with pytest.raises(ValueError, match="not the expected"):
        linker.upstream_module(args, "reactot.trainer.pl_trainer")
    args.repo = None
    with pytest.raises(ValueError, match="own environment"):
        linker.upstream_module(args, "chemrefine_missing_upstream_test_module")


def test_cli_failure_reports_nonzero_without_output(tmp_path, capsys):
    args = arguments(tmp_path)
    code = linker.main(
        [
            "--model",
            "react-ot",
            "--reactant",
            str(args.reactant),
            "--product",
            str(args.product),
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(args.output),
            "--charge",
            "1",
        ]
    )
    assert code == 1
    assert "neutral singlet" in capsys.readouterr().err
    assert not args.output.exists()


def test_fragmentflow_allows_other_elements_only_in_substituents(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    args = arguments(tmp_path, "fragmentflow", extra=["--core-indices", "0,1,2"])
    for path in (args.reactant, args.product):
        atoms = read(path)
        atoms.numbers[3] = 17
        write(path, atoms)
    r, p = linker.read_endpoints(args)

    def complete(core, rn, rp, pn, pp, kept, **kwargs):
        return rn, rp.copy(), 0, rn[kept], rp[kept]

    monkeypatch.setattr(
        linker,
        "upstream_module",
        lambda *a: SimpleNamespace(
            complete_fragmented_core_from_xyz=complete,
        ),
    )
    atoms, indices = linker.fragmentflow_ts(ContractSampler(), r, p, args)
    assert indices == [0, 1, 2]
    assert atoms.get_chemical_symbols() == ["C", "H", "N", "Cl"]
    args.full_system, args.core_indices = True, None
    with pytest.raises(ValueError, match="Predicted atoms/core"):
        linker.fragmentflow_ts(ContractSampler(), r, p, args)
