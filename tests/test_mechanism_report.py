"""Final geometry collections are portable, complete, and invalidated before reruns."""

import json
from dataclasses import replace
from html.parser import HTMLParser

import numpy as np
import pytest
from ase import Atoms
from test_mechanism import configuration

from chemrefine import io, mechanism_report, pipeline, reaction, recovery
from chemrefine.errors import CacheError, JobFailureError, OutputTreeError
from chemrefine.state import StepInputs, Structure


def stage():
    """Two distinct final TS candidates with known coordinates and calculation evidence."""
    atoms = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    first = Structure("0-1", atoms.copy(), energy_hartree=-1.2, imaginary_freqs={6: -400.0})
    second_atoms = atoms.copy()
    second_atoms.positions += [0, 0, 0.5]
    return mechanism_report.FinalStage(
        name="example",
        reaction="O & <test> → O",
        reactant=atoms,
        product=atoms.copy(),
        transition_states=(first, replace(first, id="0-2", atoms=second_atoms)),
        atom_maps=[1, 2, 3],
        charge=-1,
        multiplicity=2,
        ts_source="example/ts/step2_refined/step2_survivors.xyz",
    )


class Links(HTMLParser):
    """Collect index links for portable local-file checks."""

    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for name, value in attrs if name == "href")


def test_bundle_preserves_all_candidates_coordinates_maps_and_local_links(tmp_path):
    final_stage = stage()
    mechanism_report.begin(tmp_path)
    mechanism_report.publish(tmp_path, [final_stage])
    directory = tmp_path / "final"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "refinement_completed"
    assert manifest["verified_transition_state"] is False
    records = manifest["geometries"]
    assert [r["file"] for r in records] == [
        "001_example_reactant.xyz",
        "001_example_ts_001.xyz",
        "001_example_ts_002.xyz",
        "001_example_product.xyz",
    ]
    assert [r["frame"] for r in records] == list(range(4))
    frames = io.read_xyz_frames(directory / "mechanism.xyz")
    expected = [
        final_stage.reactant,
        *(s.atoms for s in final_stage.transition_states),
        final_stage.product,
    ]
    for record, frame, original in zip(records, frames, expected, strict=True):
        individual = io.read_xyz_frames(directory / record["file"])[0]
        np.testing.assert_allclose(frame.positions, original.positions)
        np.testing.assert_allclose(individual.positions, original.positions)
        assert record["atom_maps"] == [1, 2, 3]
        assert (record["charge"], record["multiplicity"]) == (-1, 2)
        assert f"1. example / {record['role']}" in (directory / record["file"]).read_text()
    assert records[1]["structure_id"] == "0-1"
    assert records[1]["energy_hartree"] == -1.2
    assert records[1]["imaginary_frequencies_cm_inverse"] == {"6": -400.0}
    assert records[2]["source"] == final_stage.ts_source
    assert records[0]["source"] == "example/endpoints/reactant.xyz"
    assert records[-1]["source"] == "example/endpoints/product.xyz"
    assert records[0]["energy_hartree"] is None
    text = (directory / "index.html").read_text()
    assert "O &amp; &lt;test&gt; → O" in text
    assert "<test>" not in text
    parser = Links()
    parser.feed(text)
    assert len(parser.links) == 11
    assert all((directory / link).is_file() for link in parser.links)
    assert "not a continuous trajectory" in (directory / "README.txt").read_text()


def test_new_run_invalidates_only_managed_files_and_preserves_user_notes(tmp_path):
    mechanism_report.begin(tmp_path)
    mechanism_report.publish(tmp_path, [stage()])
    directory = tmp_path / "final"
    note = directory / "my_notes.txt"
    note.write_text("keep this")
    nested = directory / "custom"
    nested.mkdir()
    bundle = directory / "001_example/reactant"
    (bundle / "my_notes.txt").write_text("retain local notes")
    (bundle / "custom").mkdir()
    (directory / "001_example/custom").mkdir()
    mechanism_report.begin(tmp_path)
    assert note.read_text() == "keep this" and nested.is_dir()
    assert not list(directory.glob("*.xyz"))
    assert not (directory / "README.txt").exists()
    assert not (bundle / "geometry.xyz").exists()
    assert not (bundle / "frequencies.json").exists()
    assert (bundle / "my_notes.txt").read_text() == "retain local notes"
    assert json.loads((directory / "manifest.json").read_text()) == {
        "status": "incomplete",
        "geometries": [],
    }
    assert "Mechanism incomplete" in (directory / "index.html").read_text()
    mechanism_report.publish(tmp_path, [replace(stage(), name="renamed")])
    assert not list(directory.glob("*example*.xyz"))
    assert len(list(directory.glob("*renamed*.xyz"))) == 4


def test_multiple_stages_support_different_atom_counts(tmp_path):
    first = stage()
    second = replace(
        first,
        name="different",
        reactant=Atoms("H"),
        product=Atoms("H"),
        atom_maps=[8],
        transition_states=(Structure("h", Atoms("H")),),
    )
    mechanism_report.publish(tmp_path, [first, second])
    frames = io.read_xyz_frames(tmp_path / "final" / "mechanism.xyz")
    assert [len(frame) for frame in frames] == [3, 3, 3, 3, 1, 1, 1]


def test_existing_mechanism_exports_optimized_endpoints_and_final_refinement(tmp_path, monkeypatch):
    from fake_engine import FakeEngine

    cfg = configuration(tmp_path)
    original = FakeEngine.parse

    def refined(self, inputs, ctx):
        result = original(self, inputs, ctx)
        if ctx.step_dir.parent.name == "ts":
            atoms = result.structures[0].atoms.copy()
            atoms.positions += [0.123, 0.456, 0.789]
            return replace(result, structures=(replace(result.structures[0], atoms=atoms),))
        return result

    monkeypatch.setattr(FakeEngine, "parse", refined)
    outcomes = pipeline.run(cfg)
    manifest = json.loads((cfg.output_dir / "final" / "manifest.json").read_text())
    assert len(manifest["geometries"]) == 9
    first_ts = io.read_xyz_frames(cfg.output_dir / "final" / "001_hydrate_ts_001.xyz")[0]
    np.testing.assert_allclose(
        first_ts.positions, outcomes[3].state.structures[0].atoms.positions, atol=1e-6
    )
    for role in ("reactant", "product"):
        expected = io.read_xyz_frames(cfg.output_dir / "hydrate" / "endpoints" / f"{role}.xyz")[0]
        exported = io.read_xyz_frames(cfg.output_dir / "final" / f"001_hydrate_{role}.xyz")[0]
        np.testing.assert_allclose(expected.positions, exported.positions)
    before = {
        str(p.relative_to(cfg.output_dir)): p.read_bytes()
        for p in (cfg.output_dir / "final").rglob("*")
        if p.is_file()
    }
    monkeypatch.setattr(FakeEngine, "submit", lambda *a: pytest.fail("resubmitted cached job"))
    recovery.execute(cfg, recovery.Action.RESUME)
    after = {
        str(p.relative_to(cfg.output_dir)): p.read_bytes()
        for p in (cfg.output_dir / "final").rglob("*")
        if p.is_file()
    }
    assert before == after
    assert (
        json.loads((cfg.output_dir / "mechanism.json").read_text())["final_geometries"]
        == "final/index.html"
    )

    def failed(*a, **kw):
        raise JobFailureError("TS failed")

    monkeypatch.setattr(reaction, "run", failed)
    with pytest.raises(JobFailureError):
        recovery.execute(cfg, recovery.Action.RESUME)
    assert not list((cfg.output_dir / "final").glob("*.xyz"))
    assert (
        json.loads((cfg.output_dir / "final" / "manifest.json").read_text())["status"]
        == "incomplete"
    )
    assert not (cfg.output_dir / "mechanism.json").exists()


def test_calculation_uses_exact_manifest_job_before_explicit_parent(tmp_path, monkeypatch):
    selected = stage().transition_states[0]
    parent = tmp_path / "parent.out"
    exact = tmp_path / "exact.out"
    monkeypatch.setattr(
        mechanism_report.cache,
        "load_manifest",
        lambda _: StepInputs(
            ((tmp_path / "p.inp", parent, "0"), (tmp_path / "e.inp", exact, "0-1"))
        ),
    )
    assert mechanism_report.calculation(tmp_path, replace(selected, parent_id="0")).output == exact
    assert (
        mechanism_report.calculation(tmp_path, replace(selected, id="child", parent_id="0")).output
        == parent
    )
    assert mechanism_report.calculation(tmp_path, replace(selected, id="unrelated")).output is None
    monkeypatch.setattr(mechanism_report.cache, "load_manifest", lambda _: None)
    assert mechanism_report.calculation(tmp_path, selected).output is None


def test_role_bundles_copy_selected_outputs_and_distinguish_missing_frequencies(tmp_path):
    final_stage = stage()
    r = Structure(
        "endpoint",
        final_stage.reactant,
        energy_hartree=-2,
        frequencies={6: 123.4},
        imaginary_freqs={},
    )
    output = tmp_path / "native.out"
    output.write_bytes(b"VIBRATIONAL FREQUENCIES\n6: 123.4 cm**-1\n\xff")
    absent = tmp_path / "unavailable.out"
    calculations = {
        "reactant": mechanism_report.Calculation(r, output),
        "product": mechanism_report.Calculation(
            replace(r, frequencies=None, imaginary_freqs=None), absent
        ),
        "ts_001": mechanism_report.Calculation(final_stage.transition_states[0], None),
    }
    mechanism_report.publish(tmp_path, [replace(final_stage, calculations=calculations)])
    directory = tmp_path / "final"
    records = json.loads((directory / "manifest.json").read_text())["geometries"]
    for record in records:
        np.testing.assert_allclose(
            io.read_xyz_frames(directory / record["file"])[0].positions,
            io.read_xyz_frames(directory / record["geometry"])[0].positions,
        )
    assert (directory / records[0]["output"]).read_bytes() == output.read_bytes()
    assert records[0]["output"] == "001_example/reactant/calculation.out"
    assert all(r["output"] is None for r in records[1:])
    minimum = json.loads((directory / records[0]["frequency_file"]).read_text())
    assert minimum["frequencies_cm_inverse"] == {"6": 123.4}
    assert minimum["imaginary_frequencies_cm_inverse"] == {}
    missing = json.loads((directory / records[-1]["frequency_file"]).read_text())
    assert missing["frequencies_cm_inverse"] is None
    assert missing["imaginary_frequencies_cm_inverse"] is None
    assert records[0]["energy_hartree"] == -2
    parser = Links()
    parser.feed((directory / "index.html").read_text())
    assert "001_example/reactant/calculation.out" in parser.links
    assert all((directory / link).is_file() for link in parser.links)
    mechanism_report.begin(tmp_path)
    assert not list(directory.rglob("calculation.out"))


def test_mechanism_bundle_outputs_match_each_branch_selected_structure(tmp_path, monkeypatch):
    from fake_engine import FakeEngine

    cfg = configuration(tmp_path)
    original = FakeEngine.parse

    def distinct(self, inputs, ctx):
        parsed = original(self, inputs, ctx)
        value = {"reactant": 101.0, "product": 202.0, "ts": -303.0, "scoring": 404.0}[
            ctx.step_dir.parent.name
        ]
        return replace(
            parsed,
            structures=tuple(
                replace(s, frequencies={6: value}, imaginary_freqs={6: value} if value < 0 else {})
                for s in parsed.structures
            ),
        )

    monkeypatch.setattr(FakeEngine, "parse", distinct)
    pipeline.run(cfg)
    directory = cfg.output_dir / "final"
    records = json.loads((directory / "manifest.json").read_text())["geometries"]
    for record in records:
        role = "ts" if record["role"].startswith("ts_") else record["role"]
        expected = {"reactant": 101.0, "product": 202.0, "ts": -303.0}[role]
        frequencies = json.loads((directory / record["frequency_file"]).read_text())
        assert frequencies["frequencies_cm_inverse"] == {"6": expected}
        originals = list((cfg.output_dir / record["stage"] / role).glob("step*/0/*.out"))
        assert len(originals) == 1
        assert (directory / record["output"]).read_bytes() == originals[0].read_bytes()


@pytest.mark.parametrize("role", ["reactant", "product"])
def test_endpoint_without_preparation_keeps_geometry_without_inventing_output(tmp_path, role):
    cfg = configuration(tmp_path)
    getattr(cfg.mechanism.stages[0], role).steps.clear()
    pipeline.run(cfg)
    directory = cfg.output_dir / "final"
    records = json.loads((directory / "manifest.json").read_text())["geometries"]
    record = next(r for r in records if r["stage"] == "hydrate" and r["role"] == role)
    assert record["output"] is None
    assert (directory / record["geometry"]).is_file()
    frequencies = json.loads((directory / record["frequency_file"]).read_text())
    assert frequencies["frequencies_cm_inverse"] is None


def test_collection_listing_refusal_uses_cache_error(tmp_path):
    (tmp_path / "final").write_text("a file cannot hold an output directory")
    with pytest.raises(CacheError):
        mechanism_report.begin(tmp_path)


def test_collection_write_refusal_uses_output_tree_error(tmp_path, monkeypatch):
    def refused(*a, **kw):
        raise PermissionError("read only output tree")

    monkeypatch.setattr(io, "atomic_write", refused)
    with pytest.raises(OutputTreeError):
        mechanism_report.begin(tmp_path)
