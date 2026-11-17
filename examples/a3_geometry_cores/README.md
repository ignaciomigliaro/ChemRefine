# Geometry-driven A3 inputs

Use `input.yaml` for the existing illustrative CuCl2 doublet model. Use
`input_CuBr.yaml` or `input_CuI.yaml` for the monomeric Cu(I) monohalide
singlet starting models. All three reuse the same product-core XYZs.

```sh
chemrefine validate examples/a3_geometry_cores/input_CuBr.yaml
chemrefine run examples/a3_geometry_cores/input_CuBr.yaml --dry-run
chemrefine run examples/a3_geometry_cores/input_CuBr.yaml
```

The YAMLs contain no atom-by-atom bond lists and no catalyst-specific selection
SMILES. `select: metal_complex` carries the single metal-containing component,
leaving the water byproduct in the preceding stage's results. The full starting
XYZ determines which metal and ligands are present. A core's metal is a placement
placeholder; omitted spectator ligands are inherited, not copied from the core.
The shared TS templates use the optimizer's Hessian, with no fixed atom indices.

Both monohalide examples contain Cu, one halide, alkyne and hydroxide initially;
they use charge -1 and multiplicity 1. The previous CuCl2 model uses two halides
and multiplicity 2. Replacing Cl with Br twice would give a CuBr2 starting model,
not CuBr. Supply the intended full complex and choose charge/spin for that model.

The examples have passed graph planning, coordinate assembly and cached workflow
tests using the test engine. The CuBr/CuI endpoints and transition states have
not been optimized or validated with UMA. A change of catalyst can change the
pathway; a TS must have the appropriate imaginary mode and connecting IRC.

Set the ORCA executable and local UMA backend/model paths before running.
