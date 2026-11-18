"""Reduced product templates conserve chemical identities and inherited attachments."""

import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem

from chemrefine import io, mechanism, product_core
from chemrefine.config import MolecularConnectivity, ProductCore
from chemrefine.errors import ConfigError


def write_xyz(tmp_path, name, atoms):
    """Write a user-owned molecular fixture."""
    path = tmp_path / name
    path.write_text(io.plain_xyz_text(atoms))
    return path


def complex_atoms(metal="Pd", ligand="P"):
    """A coordination graph with distinguishable reacting C/O and a spectator hydride ligand."""
    return Atoms(
        [metal, "C", "O", ligand, "H", "H", "H"],
        positions=[
            [0, 0, 0],
            [2.05, 0, 0],
            [0, 2, 0],
            [-2.25, 0, 0],
            [-2.8, 1.2, 0],
            [-2.8, -0.6, 1.05],
            [-2.8, -0.6, -1.05],
        ],
    )


def core_fixture(tmp_path, **kwargs):
    """Read a graph and a three-atom product core with one intended bond formed."""
    atoms = complex_atoms(**kwargs)
    path = write_xyz(tmp_path, "reactant.xyz", atoms)
    graph, _ = product_core.read_structure(path, MolecularConnectivity())
    for i, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    core_atoms = Atoms("PdCO", positions=[[0, 0, 0], [2.05, 0, 0], [3.4, 0, 0]])
    path = write_xyz(tmp_path, "core.xyz", core_atoms)
    settings = ProductCore(input=path, placements=2)
    core, _ = product_core.read_structure(path, settings.connectivity, reduced=True)
    return atoms, graph, core_atoms, core, settings


def matches(graph, core, **kwargs):
    """Run bounded matching with ordinary defaults."""
    return product_core.match_core(
        graph, core, anchor=0, hints={}, dissociate=[], max_candidates=16, **kwargs
    )


@pytest.mark.parametrize("metal", ["Pd", "Pt"])
@pytest.mark.parametrize("ligand", ["P", "As"])
def test_same_core_inherits_real_metal_and_spectator_ligand(tmp_path, metal, ligand):
    atoms, graph, core_atoms, core, settings = core_fixture(tmp_path, metal=metal, ligand=ligand)
    (match,) = matches(graph, core)
    assert match.indices == (0, 1, 2)
    assert match.edits == 2
    assert match.graph.GetAtomWithIdx(0).GetSymbol() == metal
    assert mechanism.maps(match.graph) == list(range(1, 8))
    result, candidates, scores = product_core.assemble(
        atoms, graph, match.graph, core_atoms, match.indices, settings
    )
    assert 1 <= len(candidates) == len(scores) <= 2
    np.testing.assert_allclose(result.positions[:3], core_atoms.positions)
    np.testing.assert_allclose(result.positions[3:], atoms.positions[3:], atol=1e-7)
    assert result.get_chemical_symbols() == atoms.get_chemical_symbols()
    assert product_core.distance_edges(result) == {(0, 1), (1, 2), (0, 3), (3, 4), (3, 5), (3, 6)}


def test_reordered_xyz_binds_by_graph_not_row_or_nearest_coordinate(tmp_path):
    atoms, graph, _, _, _ = core_fixture(tmp_path)
    permutation = [3, 6, 0, 2, 5, 1, 4]
    shuffled = atoms[permutation]
    shuffled.positions += [17, -8, 5]
    result = product_core.bind_geometry(graph, shuffled, label="shuffled")
    assert result.get_chemical_symbols() == atoms.get_chemical_symbols()
    # Equivalent H rows may permute, but their attached ligand and heavy rows cannot.
    np.testing.assert_allclose(result.positions[:4], np.add(atoms.positions[:4], [17, -8, 5]))
    mechanism.validate_geometry(graph, result, label="checked")
    path = write_xyz(tmp_path, "shuffled.xyz", shuffled)
    bound = mechanism._read_geometry(path, graph)
    assert bound.info["input_atom_indices"] == result.info["input_atom_indices"]


def test_bind_geometry_refuses_wrong_inventory_and_graph(tmp_path):
    atoms, graph, _, _, _ = core_fixture(tmp_path)
    with pytest.raises(ConfigError, match="inventory"):
        product_core.bind_geometry(graph, atoms[:-1], label="missing")
    wrong = atoms.copy()
    wrong.symbols[1] = "N"
    with pytest.raises(ConfigError, match="correspondence"):
        product_core.bind_geometry(graph, wrong, label="wrong element")


def test_explicit_bond_orders_and_charge_metadata(tmp_path):
    path = write_xyz(
        tmp_path, "explicit.xyz", Atoms("PdCO", positions=[[0, 0, 0], [2, 0, 0], [3.2, 0, 0]])
    )
    graph, _ = product_core.read_structure(
        path, MolecularConnectivity(bonds=[(2, 1, 0), (2, 3, 2)], formal_charges={1: 1})
    )
    assert graph.GetBondBetweenAtoms(1, 0).GetBondType() == Chem.BondType.DATIVE
    assert graph.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.DOUBLE
    assert Chem.GetFormalCharge(graph) == 1
    assert not graph.GetBoolProp("connectivity_only")


@pytest.mark.parametrize(
    "bonds,charges,reason",
    [
        ([(0, 1, 1)], {}, "in range"),
        ([(1, 1, 1)], {}, "distinct"),
        ([(1, 8, 1)], {}, "in range"),
        ([(1, 2, 1), (2, 1, 1)], {}, "duplicate"),
        ([], {4: 1}, "charge row"),
    ],
)
def test_invalid_connectivity_metadata(tmp_path, bonds, charges, reason):
    path = write_xyz(tmp_path, "bad.xyz", Atoms("PdC", positions=[[0, 0, 0], [2, 0, 0]]))
    with pytest.raises(ConfigError, match=reason):
        product_core.read_structure(
            path, MolecularConnectivity(bonds=bonds, formal_charges=charges)
        )


@pytest.mark.parametrize("positions", [[[0, 0, 0], [0.1, 0, 0]], [[0, 0, 0], [2.7, 0, 0]]])
def test_ambiguous_distances_require_clearer_geometry(tmp_path, positions):
    path = write_xyz(tmp_path, "ambiguous.xyz", Atoms("PdC", positions=positions))
    with pytest.raises(ConfigError, match="adjust the structure"):
        product_core.read_structure(path, MolecularConnectivity())
    graph, _ = product_core.read_structure(path, MolecularConnectivity(bonds=[]))
    assert graph.GetNumBonds() == 0


def test_organic_xyz_infers_real_bond_orders(tmp_path):
    path = write_xyz(
        tmp_path, "water.xyz", Atoms("OH2", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    )
    graph, _ = product_core.read_structure(path, MolecularConnectivity())
    assert graph.GetNumBonds() == 2
    assert Chem.GetFormalCharge(graph) == 0


def test_unresolved_organic_xyz_has_actionable_error(tmp_path, monkeypatch):
    path = write_xyz(tmp_path, "water.xyz", Atoms("O", positions=[[0, 0, 0]]))

    def fail(*args, **kwargs):
        """Simulate a bond-perception refusal."""
        raise ValueError("no valence")

    monkeypatch.setattr(product_core.rdDetermineBonds, "DetermineBonds", fail)
    with pytest.raises(ConfigError, match="check geometry and charge"):
        product_core.read_structure(path, MolecularConnectivity())


@pytest.mark.parametrize("frames", [[], [Atoms()], [Atoms("Pd"), Atoms("Pd")]])
def test_requires_one_nonempty_frame(tmp_path, monkeypatch, frames):
    monkeypatch.setattr(io, "read_xyz_frames", lambda path: frames)
    with pytest.raises(ConfigError, match="one nonempty"):
        product_core.read_structure(tmp_path / "x.xyz", MolecularConnectivity())


@pytest.mark.parametrize("kind", ["pbc", "nan"])
def test_read_and_validation_reject_bad_coordinates(tmp_path, monkeypatch, kind):
    atoms, graph, _, _, _ = core_fixture(tmp_path)
    if kind == "pbc":
        atoms.pbc = True
    else:
        atoms.positions[0, 0] = np.nan
    monkeypatch.setattr(io, "read_xyz_frames", lambda path: [atoms])
    with pytest.raises(ConfigError, match="finite nonperiodic"):
        product_core.read_structure(tmp_path / "x.xyz", MolecularConnectivity())
    with pytest.raises(ConfigError, match="identities or finite"):
        product_core.validate_connectivity(graph, atoms, label="bad")


def test_connectivity_change_and_wrong_elements_are_rejected(tmp_path):
    atoms, graph, _, _, _ = core_fixture(tmp_path)
    wrong = atoms.copy()
    wrong.positions[3:] += [10, 0, 0]
    with pytest.raises(ConfigError, match="coordination changed"):
        product_core.validate_connectivity(graph, wrong, label="lost ligand")
    atoms.symbols[0] = "Pt"
    with pytest.raises(ConfigError, match="identities"):
        product_core.validate_connectivity(graph, atoms, label="changed metal")


@pytest.mark.parametrize(
    "anchor,hints,reason",
    [
        (-1, {}, "inventory or anchor"),
        (8, {}, "inventory or anchor"),
        (1, {}, "metal placeholder"),
        (0, {0: 1}, "match hints"),
        (0, {1: 99}, "match hints"),
        (0, {2: 3}, "no compatible"),
    ],
)
def test_invalid_core_anchor_and_hints(tmp_path, anchor, hints, reason):
    _, graph, _, core, _ = core_fixture(tmp_path)
    with pytest.raises(ConfigError, match=reason):
        product_core.match_core(
            graph, core, anchor=anchor, hints=hints, dissociate=[], max_candidates=16
        )


def test_explicit_hints_use_persistent_identities(tmp_path):
    _, graph, _, core, _ = core_fixture(tmp_path)
    for atom in graph.GetAtoms():
        atom.SetAtomMapNum(atom.GetAtomMapNum() + 100)
    (match,) = product_core.match_core(
        graph, core, anchor=0, hints={2: 102}, dissociate=[], max_candidates=16
    )
    assert match.indices == (0, 1, 2)


def test_inventory_and_injective_mapping_required(tmp_path):
    _, graph, _, core, _ = core_fixture(tmp_path)
    with pytest.raises(ConfigError, match="inventory"):
        product_core.match_core(core, graph, anchor=0, hints={}, dissociate=[], max_candidates=16)
    duplicate = Chem.MolFromSmiles("[Pd].[Pd]")
    with pytest.raises(ConfigError, match="no atom-conserving"):
        matches(graph, duplicate)


def test_core_cannot_introduce_undeclared_elements(tmp_path):
    _, graph, _, _, _ = core_fixture(tmp_path)
    with pytest.raises(ConfigError, match="no compatible"):
        matches(graph, Chem.MolFromSmiles("[Pd].[N]"))


def test_symmetric_matches_deduplicate_but_distinct_sites_branch():
    graph = Chem.MolFromSmiles("[Pd]([CH3])([CH3])([CH2]C)")
    graph = Chem.AddHs(graph)
    for i, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    core = Chem.MolFromSmiles("[Pd].[C]")
    results = matches(graph, core)
    assert len(results) == 2  # methyl and ethyl alpha; beta would change no bonds
    assert sorted(r.edits for r in results) == [1, 1]
    with pytest.raises(ConfigError, match="chemical candidates"):
        product_core.match_core(graph, core, anchor=0, hints={}, dissociate=[], max_candidates=1)


def test_dissociation_is_explicit_and_keeps_ligand_atoms(tmp_path):
    atoms, graph, core_atoms, core, settings = core_fixture(tmp_path)
    (match,) = product_core.match_core(
        graph, core, anchor=0, hints={}, dissociate=["[P]([H])([H])[H]"], max_candidates=16
    )
    assert match.graph.GetNumAtoms() == graph.GetNumAtoms()
    assert match.graph.GetBondBetweenAtoms(0, 3) is None
    result, _, _ = product_core.assemble(
        atoms, graph, match.graph, core_atoms, match.indices, settings
    )
    assert result.get_distance(0, 3) > 3.5
    np.testing.assert_allclose(
        result.get_all_distances()[3:, 3:], atoms.get_all_distances()[3:, 3:], atol=1e-6
    )


@pytest.mark.parametrize("selector", ["invalid[", "[N]", "[H]"])
def test_dissociation_refuses_invalid_or_partial_ligands(tmp_path, selector):
    _, graph, _, core, _ = core_fixture(tmp_path)
    with pytest.raises(ConfigError, match="dissociate"):
        product_core.match_core(
            graph, core, anchor=0, hints={}, dissociate=[selector], max_candidates=16
        )


def test_proper_rigid_fit_cannot_reflect_chirality():
    source = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    target = source * [-1, 1, 1] + [2, 3, 4]
    rotation, _ = product_core.fit_rigid(source, target)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_impossible_assembly_is_not_reported_as_product(tmp_path):
    atoms, graph, core_atoms, core, settings = core_fixture(tmp_path)
    (match,) = matches(graph, core)
    core_atoms.positions[2] = core_atoms.positions[1] + [8, 0, 0]
    with pytest.raises(ConfigError, match="could not resolve clashes"):
        product_core.assemble(atoms, graph, match.graph, core_atoms, match.indices, settings)


def test_full_core_without_spectators(tmp_path):
    _, _, core_atoms, core, settings = core_fixture(tmp_path)
    full = Chem.Mol(core)
    for i, a in enumerate(full.GetAtoms(), 1):
        a.SetAtomMapNum(i)
    result, candidates, scores = product_core.assemble(
        core_atoms, full, full, core_atoms, (0, 1, 2), settings
    )
    assert len(candidates) == 1 and scores == [0.0]
    np.testing.assert_equal(result.positions, core_atoms.positions)


def test_matching_budget_never_silently_truncates(tmp_path, monkeypatch):
    _, graph, _, core, _ = core_fixture(tmp_path)
    monkeypatch.setattr(product_core, "MATCH_SEARCH_BUDGET", 1)
    with pytest.raises(ConfigError, match="exceeded budget"):
        matches(graph, core)


def test_dissociation_cannot_contradict_core_or_omit_anchor(tmp_path):
    _, graph, _, _, _ = core_fixture(tmp_path)
    core = Chem.MolFromSmiles("[Pd][P]")
    with pytest.raises(ConfigError, match="still bonded"):
        product_core.match_core(
            graph, core, anchor=0, hints={}, dissociate=["[PH3]"], max_candidates=16
        )
    with pytest.raises(ConfigError, match="requires a metal anchor"):
        product_core.match_core(
            graph, graph, anchor=None, hints={}, dissociate=["[PH3]"], max_candidates=16
        )


def test_explicit_core_charge_overrides_only_declared_rows(tmp_path):
    _, graph, _, core, _ = core_fixture(tmp_path)
    core.GetAtomWithIdx(0).SetFormalCharge(2)
    core.GetAtomWithIdx(0).SetBoolProp("formal_charge_explicit", True)
    (match,) = matches(graph, core)
    assert match.graph.GetAtomWithIdx(0).GetFormalCharge() == 2
    assert match.graph.GetAtomWithIdx(3).GetFormalCharge() == 0


def test_complete_endpoint_matching_preserves_h_parents_and_minimizes_edits():
    graph = mechanism.parse_smiles("[CH3:1][OH:2]", mapped=True)
    target = mechanism.parse_smiles("CO", mapped=False)
    (match,) = product_core.match_core(
        graph, target, anchor=None, hints={}, dissociate=[], max_candidates=16
    )
    assert match.edits == 0
    assert mechanism.maps(match.graph) == mechanism.maps(graph)


def test_complete_endpoint_can_transfer_a_hydrogen():
    graph = mechanism.parse_smiles("[CH3:1][OH:2]", mapped=True)
    target = mechanism.parse_smiles("[CH2-][OH2+]", mapped=False)
    results = product_core.match_core(
        graph, target, anchor=None, hints={}, dissociate=[], max_candidates=16
    )
    assert results[0].edits == 2
    assert results[0].graph.GetNumAtoms() == graph.GetNumAtoms()


def test_core_bond_overlap_is_rejected_even_with_declared_connectivity():
    graph = Chem.MolFromSmiles("[Pd][C]")
    with pytest.raises(ConfigError, match="severe atomic overlap"):
        product_core.validate_connectivity(
            graph, Atoms("PdC", positions=[[0, 0, 0], [0.1, 0, 0]]), label="collapsed"
        )


@pytest.mark.parametrize("metal_row", [0, 1])
def test_force_allows_only_changed_metal_contacts(metal_row):
    symbols = ["Pd", "N"] if metal_row == 0 else ["N", "Pd"]
    graph = Chem.MolFromSmiles("[Pd][N]" if metal_row == 0 else "[N][Pd]")
    atoms = Atoms(symbols, positions=[[0, 0, 0], [4, 0, 0]])
    with pytest.raises(ConfigError, match="coordination changed"):
        product_core.validate_connectivity(graph, atoms, label="strict")
    product_core.validate_connectivity(graph, atoms, label="forced", allow_metal_changes=True)


def test_force_keeps_ordinary_bonds_atom_identity_finiteness_and_overlap_checks():
    graph = Chem.MolFromSmiles("[Pd].[I].[O]")
    atoms = Atoms("PdIO", positions=[[0, 0, 0], [5, 0, 0], [7.55, 0, 0]])
    # I-O involves no metal, exactly like the stopped CuI2 model.
    with pytest.raises(ConfigError, match="coordination changed"):
        product_core.validate_connectivity(graph, atoms, label="I-O", allow_metal_changes=True)
    graph = Chem.MolFromSmiles("[Pd][N]")
    overlap = Atoms("PdN", positions=[[0, 0, 0], [0.1, 0, 0]])
    with pytest.raises(ConfigError, match="overlap"):
        product_core.validate_connectivity(
            graph, overlap, label="overlap", allow_metal_changes=True
        )
    for invalid in [Atoms("PdO"), Atoms("PdN", positions=[[0, 0, 0], [float("nan"), 0, 0]])]:
        with pytest.raises(ConfigError, match="identities or finite"):
            product_core.validate_connectivity(
                graph, invalid, label="invalid", allow_metal_changes=True
            )


def test_explicit_broader_contact_override_still_checks_overlap_and_identity():
    graph = Chem.MolFromSmiles("[Pd].[I].[O]")
    atoms = Atoms("PdIO", positions=[[0, 0, 0], [5, 0, 0], [7.55, 0, 0]])
    product_core.validate_connectivity(
        graph, atoms, label="explicit I-O override", allow_contact_changes=True
    )
    for bad in [Atoms("PdIO", positions=[[0, 0, 0], [0.1, 0, 0], [7, 0, 0]]), Atoms("PdIN")]:
        with pytest.raises(ConfigError):
            product_core.validate_connectivity(graph, bad, label="bad", allow_contact_changes=True)


def test_chelate_group_moves_as_one_connected_fragment(tmp_path):
    # A synthetic bidentate S-C-C-S ring; distance preservation, not an energy model.
    atoms = Atoms(
        "PdCOSCCS",
        positions=[
            [0, 0, 0],
            [2.05, 0, 0],
            [0, 2, 0],
            [-2.15, 1.4, 0],
            [-3.55, 0.75, 0],
            [-3.55, -0.75, 0],
            [-2.15, -1.4, 0],
        ],
    )
    path = write_xyz(tmp_path, "chelate.xyz", atoms)
    bonds = [(1, 2, 1), (1, 3, 1), (1, 4, 0), (4, 5, 1), (5, 6, 1), (6, 7, 1), (7, 1, 0)]
    graph, _ = product_core.read_structure(path, MolecularConnectivity(bonds=bonds))
    for i, a in enumerate(graph.GetAtoms(), 1):
        a.SetAtomMapNum(i)
    core_atoms = Atoms("PdCO", positions=[[0, 0, 0], [2.05, 0, 0], [3.4, 0, 0]])
    core_path = write_xyz(tmp_path, "chelate_core.xyz", core_atoms)
    core, _ = product_core.read_structure(core_path, MolecularConnectivity(), reduced=True)
    (match,) = product_core.match_core(
        graph, core, anchor=0, hints={2: 2}, dissociate=[], max_candidates=16
    )
    result, _, _ = product_core.assemble(
        atoms,
        graph,
        match.graph,
        core_atoms,
        match.indices,
        ProductCore(input=core_path, placements=2),
    )
    np.testing.assert_allclose(
        result.get_all_distances()[3:, 3:], atoms.get_all_distances()[3:, 3:], atol=0.05
    )


def test_inherited_group_with_multiple_core_attachments(tmp_path):
    atoms, graph, core_atoms, core, settings = core_fixture(tmp_path)
    graph = Chem.RWMol(graph)
    graph.AddBond(1, 3, Chem.BondType.SINGLE)
    graph = graph.GetMol()
    # Move P close enough to both Pd and C; H follows the same displacement.
    atoms.positions[3:] = [[1, -2, 0], [1, -3.3, 0], [0, -2.5, 0.7], [2, -2.5, -0.7]]
    (match,) = matches(graph, core)
    result, _, _ = product_core.assemble(
        atoms, graph, match.graph, core_atoms, match.indices, settings
    )
    assert result.get_distance(1, 3) == pytest.approx(atoms.get_distance(1, 3), abs=0.15)


def test_relaxation_cannot_destroy_inherited_geometry(tmp_path, monkeypatch):
    from types import SimpleNamespace

    atoms, graph, core_atoms, core, settings = core_fixture(tmp_path)
    (match,) = matches(graph, core)

    def distorted(function, values, **kwargs):
        """An unconverged relaxation that tears apart the ligand must be refused."""
        values = values.copy()
        values[0] += 2
        return SimpleNamespace(x=values)

    monkeypatch.setattr(product_core, "least_squares", distorted)
    with pytest.raises(ConfigError, match="could not resolve clashes"):
        product_core.assemble(atoms, graph, match.graph, core_atoms, match.indices, settings)


def test_organic_multiple_bonds_in_metal_xyz_are_inferred_without_metadata(tmp_path):
    path = write_xyz(
        tmp_path, "carbonyl.xyz", Atoms("PdCO", positions=[[0, 0, 0], [2.05, 0, 0], [3.25, 0, 0]])
    )
    graph, _ = product_core.read_structure(path, MolecularConnectivity())
    assert graph.GetBondBetweenAtoms(1, 2).GetBondType() == Chem.BondType.DOUBLE
    assert graph.GetBoolProp("connectivity_only")


def test_automatic_bond_orders_respect_valence_in_aromatic_ligands():
    from rdkit.Chem import AllChem

    ring = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(ring, randomSeed=42)
    AllChem.UFFOptimizeMolecule(ring)
    atoms = Atoms(
        numbers=[a.GetAtomicNum() for a in ring.GetAtoms()],
        positions=ring.GetConformer().GetPositions(),
    )
    bonds = product_core._perceived_bonds(atoms)
    assert sum(order == 2 for _, _, order in bonds) == 3
    for carbon in range(6):
        assert sum(order for a, b, order in bonds if carbon + 1 in (a, b)) == 4


def test_automatic_alkyne_bonds_distinguish_pi_from_sigma_metal_attachment():
    pi = Atoms(
        "CuCCHH",
        positions=[[0, 0, 0], [2.05, -0.6, 0], [2.05, 0.6, 0], [2.05, -1.66, 0], [2.05, 1.66, 0]],
    )
    bonds = product_core._perceived_bonds(pi)
    assert (2, 3, 3) in bonds
    assert (1, 2, 0) in bonds and (1, 3, 0) in bonds
    sigma = Atoms("CuCCH", positions=[[0, 0, 0], [1.9, 0, 0], [3.1, 0, 0], [4.16, 0, 0]])
    assert product_core._perceived_bonds(sigma) == [(1, 2, 1), (2, 3, 3), (3, 4, 1)]


def test_bond_perception_failure_stops_instead_of_using_arbitrary_orders(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(product_core, "milp", lambda **_: SimpleNamespace(success=False))
    with pytest.raises(ConfigError, match="automatic organic bond perception failed"):
        product_core._perceived_bonds(Atoms("CC", positions=[[0, 0, 0], [1.2, 0, 0]]))


def test_impossible_carbon_valence_cannot_be_a_candidate():
    graph = mechanism.parse_smiles("[Pd:1][CH3:2].[O:3]", mapped=True)
    core = Chem.MolFromSmiles("[Pd][C][O]")
    with pytest.raises(ConfigError, match="no atom-conserving"):
        matches(graph, core)


def test_complete_endpoint_rejects_worse_correspondences():
    graph = mechanism.parse_smiles("[CH3:1][CH2:2][OH:3]", mapped=True)
    core = mechanism.parse_smiles("CCO", mapped=False)
    (result,) = product_core.match_core(
        graph, core, anchor=None, hints={}, dissociate=[], max_candidates=16
    )
    assert result.edits == 0


def test_symmetric_group_centroid_at_anchor_is_finite(tmp_path):
    # An algebraic stress case, not a proposed chemical structure.
    atoms = Atoms(
        "PdCOI2", positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.4, 0], [0, 0, 1.6], [0, 0, -1.6]]
    )
    path = write_xyz(tmp_path, "centered.xyz", atoms)
    graph, _ = product_core.read_structure(
        path, MolecularConnectivity(bonds=[(1, 2, 1), (1, 3, 1), (1, 4, 1), (1, 5, 1), (4, 5, 1)])
    )
    for i, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    core_atoms = Atoms("PdCO", positions=[[0, 0, 0], [2.5, 0, 0], [3.85, 0, 0]])
    core_path = write_xyz(tmp_path, "centered_core.xyz", core_atoms)
    connectivity = MolecularConnectivity(bonds=[(1, 2, 1), (2, 3, 1)])
    core, _ = product_core.read_structure(core_path, connectivity, reduced=True)
    (match,) = matches(graph, core)
    result, _, _ = product_core.assemble(
        atoms,
        graph,
        match.graph,
        core_atoms,
        match.indices,
        ProductCore(input=core_path, connectivity=connectivity, placements=2),
    )
    assert np.isfinite(result.positions).all()


@pytest.mark.parametrize("substituent", ["F", "N", "C"])
def test_substrate_substituents_are_inherited_and_clashes_resolved(tmp_path, substituent):
    atoms, _, core_atoms, core, settings = core_fixture(tmp_path)
    from ase.data import atomic_numbers, covalent_radii

    bond_length = covalent_radii[6] + covalent_radii[atomic_numbers[substituent]]
    atoms += Atoms(substituent, positions=[[2.05 + bond_length, 0, 0]])
    path = write_xyz(tmp_path, "substituted.xyz", atoms)
    graph, _ = product_core.read_structure(path, MolecularConnectivity())
    for i, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    match = next(m for m in matches(graph, core) if m.indices == (0, 1, 2))
    result, _, _ = product_core.assemble(
        atoms,
        graph,
        match.graph,
        core_atoms,
        match.indices,
        settings.model_copy(update={"placements": 8}),
    )
    assert result.get_chemical_symbols()[-1] == substituent
    assert result.get_distance(1, 7) == pytest.approx(bond_length, abs=0.15)
    assert result.get_distance(2, 7) > 1.25 * (
        covalent_radii[8] + covalent_radii[atomic_numbers[substituent]]
    )
    assert mechanism.maps(match.graph) == list(range(1, 9))


def test_heavy_atom_symmetry_budget_is_explicit(tmp_path, monkeypatch):
    atoms, graph, _, _, _ = core_fixture(tmp_path)
    monkeypatch.setattr(product_core, "GEOMETRY_MATCH_LIMIT", 0)
    with pytest.raises(ConfigError, match="symmetry limit"):
        product_core.bind_geometry(graph, atoms, label="bounded")


def test_bridging_hydrogen_connectivity_cannot_be_ignored(tmp_path):
    atoms = Atoms(
        "PdPtNiH3",
        positions=[
            [0, 0, 0],
            [2.8, 0, 0],
            [1.4, 2, 0],
            [1.4, 0.65, 0],
            [1.4, -0.7, 0],
            [1.4, 3.3, 0],
        ],
    )
    path = write_xyz(tmp_path, "bridge.xyz", atoms)
    graph, _ = product_core.read_structure(
        path,
        MolecularConnectivity(
            bonds=[
                (1, 2, 1),
                (1, 3, 1),
                (2, 3, 1),
                (1, 4, 1),
                (2, 4, 1),
                (2, 5, 1),
                (3, 5, 1),
                (1, 6, 1),
                (3, 6, 1),
            ]
        ),
    )
    with pytest.raises(ConfigError, match="correspondence is missing"):
        product_core.bind_geometry(graph, atoms, label="wrong bridges")


def test_metal_mixture_selector_uses_only_established_graph_properties(tmp_path):
    _, graph, _, _, _ = core_fixture(tmp_path)
    # Release CO; the remaining Pd-PH3 component is represented by distance
    # adjacency, so a chemically equivalent dative SMILES must still select it.
    core_atoms = Atoms("PdCO", positions=[[0, 0, 0], [4, 0, 0], [5.35, 0, 0]])
    core_path = write_xyz(tmp_path, "released_core.xyz", core_atoms)
    core, _ = product_core.read_structure(core_path, MolecularConnectivity(), reduced=True)
    (match,) = matches(graph, core)
    selected, indices = mechanism.select_component(match.graph, "[PH3]->[Pd]")
    assert set(mechanism.maps(selected)) == {1, 4, 5, 6, 7}
    assert indices == [0, 3, 4, 5, 6]
    assert selected.GetBoolProp("connectivity_only")


def test_reduced_proton_transfer_preserves_the_unreacting_alkyne_hydrogen():
    reactant = mechanism.parse_smiles("[Cl-]->[Cu+2]1(<-[Cl-])<-[CH]#[CH]->1.[OH-]", mapped=False)
    core = mechanism.parse_smiles("[Cl-]->[Cu+2]1(<-[Cl-])<-[C-]#[CH]->1.O", mapped=False)
    for identity, atom in enumerate(reactant.GetAtoms(), 1):
        atom.SetAtomMapNum(identity)
    for atom in core.GetAtoms():
        atom.SetBoolProp("formal_charge_explicit", True)
    anchor = next(atom.GetIdx() for atom in core.GetAtoms() if atom.GetSymbol() == "Cu")
    (match,) = product_core.match_core(
        reactant, core, anchor=anchor, hints={}, dissociate=[], max_candidates=16
    )
    # One C-H bond breaks and one O-H bond forms. The other alkyne H
    # and the original hydroxide H must keep their original parent atoms.
    assert match.edits == 2
    hydrogen_parent_changes = 0
    for atom in reactant.GetAtoms():
        if atom.GetAtomicNum() == 1:
            original = {neighbor.GetIdx() for neighbor in atom.GetNeighbors()}
            current = {
                neighbor.GetIdx()
                for neighbor in match.graph.GetAtomWithIdx(atom.GetIdx()).GetNeighbors()
            }
            hydrogen_parent_changes += original != current
    assert hydrogen_parent_changes == 1


@pytest.mark.parametrize("substituent", ["H", "CH3"])
def test_omitted_alkyne_h_or_r_rotates_with_the_conserved_triple_bond(tmp_path, substituent):
    atoms = Atoms("CuCC", positions=[[0, 0, 0], [-0.6, 0, 1.9], [0.6, 0, 1.9]])
    bonds = [(1, 2, 0), (1, 3, 0), (2, 3, 3)]
    if substituent == "H":
        atoms += Atoms("H", positions=[[1.66, 0, 1.9]])
        bonds.append((3, 4, 1))
    else:
        atoms += Atoms(
            "CH3",
            positions=[
                [2.14, 0, 1.9],
                [2.5033, 1.0277, 1.9],
                [2.5033, -0.5138, 2.7900],
                [2.5033, -0.5138, 1.0100],
            ],
        )
        bonds.extend([(3, 4, 1), (4, 5, 1), (4, 6, 1), (4, 7, 1)])
    path = write_xyz(tmp_path, "alkyne.xyz", atoms)
    graph, _ = product_core.read_structure(
        path, MolecularConnectivity(bonds=bonds, formal_charges={1: 1, 2: -1})
    )
    for i, atom in enumerate(graph.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    core_atoms = Atoms("CuCC", positions=[[0, 0, 0], [0, 1.9, 0], [0, 3.1, 0]])
    core_path = write_xyz(tmp_path, "rotated_sigma.xyz", core_atoms)
    connectivity = MolecularConnectivity(bonds=[(1, 2, 0), (2, 3, 3)], formal_charges={2: -1, 3: 0})
    core, _ = product_core.read_structure(core_path, connectivity, reduced=True)
    (match,) = matches(graph, core)
    result, _, _ = product_core.assemble(
        atoms,
        graph,
        match.graph,
        core_atoms,
        match.indices,
        ProductCore(input=core_path, connectivity=connectivity, placements=1),
    )
    triple = result.positions[2] - result.positions[1]
    inherited = result.positions[3] - result.positions[2]
    cosine = float(np.dot(triple, inherited) / np.linalg.norm(triple) / np.linalg.norm(inherited))
    assert cosine == pytest.approx(1.0, abs=1e-6)
    assert result.get_distance(2, 3) == pytest.approx(atoms.get_distance(2, 3), abs=1e-6)
    np.testing.assert_allclose(
        result.get_all_distances()[3:, 3:], atoms.get_all_distances()[3:, 3:], atol=1e-6
    )
