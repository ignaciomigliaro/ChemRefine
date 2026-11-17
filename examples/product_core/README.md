# Later products from a small reaction core

This example starts with complete XYZ endpoints for
`Pd(Me)2(PH3)2 + MeI -> Pd(Me)3(I)(PH3)2`, then uses only a three-atom
product core for ethane elimination. The full second product retains all 22
atoms. Its omitted phosphines, methyl group, iodide, and hydrogens come from the
first stage's optimized product.

Change the catalyst, ligands, or substrate in the first endpoint pair. The same
later core can be reused when it still describes the intended bond changes and
can identify the reacting fragment. The core's Pd is a metal placeholder; the
actual metal comes from the carried catalyst.

Configure `executables.orca`, then set `backend_python` and `model_path` under
the shared UMA options for your installation. All preparation and TS templates
use UMA energies/gradients through ORCA ExtOpt. TS Hessians and frequencies are
numerical. From this directory:

```bash
chemrefine validate input.yaml
chemrefine run input.yaml --dry-run
chemrefine run input.yaml
chemrefine resume input.yaml
```

Add each incoming reagent explicitly as a SMILES or a separate XYZ source.
Include its reacting atoms in the new product core. Omitted existing ligands are
retained automatically. A departing ligand can be represented by including its
donor and the metal in the core with their contact broken, or by a whole-ligand
`dissociate` selector. Ambiguous selectors require a more specific declaration.

The core supplies the intended product geometry, not a discovered mechanism.
Include enough reacting context to distinguish substrate atoms from ligand atoms;
chemically different matches become separate candidates. Provide connectivity
metadata for ambiguous XYZ bonding and explicit charge/spin when required.

Final structures are collected in `outputs/final/index.html` and
`outputs/final/mechanism.xyz`. Per-candidate assembly and atom correspondence are
recorded under `outputs/_resolved/`. The inputs do not certify TS convergence,
reaction barriers, or IRC connectivity.

The opt-in regression test `tests/test_core_workflow_live.py` exercises this
assembly and all calculation branches with real UMA/ORCA **single points**, then
checks that resume submits no new jobs. Set `CHEMREFINE_CORE_UMA_PYTHON`,
`CHEMREFINE_CORE_UMA_MODEL`, and `CHEMREFINE_CORE_ORCA`, then run from the repository:

```bash
pytest -m integration tests/test_core_workflow_live.py
```

See `docs/tutorials/linked-reactions.md` for the complete schema and behavior.
