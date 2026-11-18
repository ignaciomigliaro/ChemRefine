# Separate A3 reagents and an initial coordination core

This example batches propyne, 1-butyne and 1-pentyne with the same CuCl2 catalyst
and hydroxide. Every reagent is a separate, complete XYZ file. The six-atom
`inputs/pi_core.xyz` places Cu, both alkyne carbons, the terminal alkyne H,
and hydroxide's O and H. Catalyst ligands and alkyl chains come from the sources;
core atoms identify existing source atoms and never add atoms to the system.

`reactant.initial_core` adds only inter-fragment metal coordination. It preserves
source bonds and charges and cannot transfer H or dissociate a ligand. Each
connected source fragment must contribute at least one core atom. Atom identity
comes from graph correspondence, not XYZ row proximity. Chemically distinct
matches become separate mechanism candidates, bounded by `max_candidates`;
`match` hints can restrict correspondence using persistent atom identities.

Assembly fits and rotates intact fragments, preserving their internal distances.
Template coordinates are targets with a maximum fitting deviation of 0.15 Å,
allowing small differences in independently prepared bond lengths. Use more core
atoms to constrain a ligand's orientation. `placements` samples unresolved
rotations; the selected candidate minimizes a geometric clash penalty, not an
electronic energy. Optimization still determines whether the assembled complex
is a stable structure.

All variants retain the terminal H and use the same stages: H abstraction,
pi-to-sigma rearrangement, then addition to methanimine. Hydroxide is OH−
(`charge: -1`), rather than the neutral OH radical. Water is retained in the
first product and excluded by the next stage's `select: metal_complex`.
The illustrative combined model has charge −1 and multiplicity 2.

Configure ORCA and the local UMA model/backend before running:

```sh
chemrefine validate examples/a3_initial_core/input.yaml
chemrefine run examples/a3_initial_core/input.yaml --dry-run
chemrefine run examples/a3_initial_core/input.yaml
```

The substrate `input_dir` produces three independent runs under
`outputs/variants/<substrate>/`. Each first-stage directory contains
`initial_assembly_candidates.xyz`, `initial_assembly.json` (source membership,
core correspondence and placement scores), and `_inputs/reactant.xyz` (the
assembled complex). The unchanged product cores and templates are shared with
`../a3_geometry_cores`.

These are unoptimized starting models. Workflow tests check assembly, atom
conservation and cached handoff; they do not validate a mechanism or TS.
