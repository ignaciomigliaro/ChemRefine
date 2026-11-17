One input, independent XYZ alternatives

This portable example uses the same two product cores with two full starting
reactants: reactants/Pd.xyz and reactants/Pt.xyz. Set the ORCA executable and UMA
backend/model paths before a real run. The geometries are illustrative, not
validated reaction pathways.

The configuration sets max_cores=2, max_gpus=0 and mechanism.parallel_runs=2.
Each mechanism receives one CPU core and uses its own driver process and cache.

  chemrefine run examples/product_core_batch/input.yaml --dry-run
  chemrefine run examples/product_core_batch/input.yaml
  chemrefine resume examples/product_core_batch/input.yaml

Open outputs/index.html for the batch report. Each alternative has a generated
variant.yaml, driver.log, full native calculations, and a final/ collection after
refinement completes. A failed alternative is reported without cancelling the
others. More than one input_dir source forms independent Cartesian combinations;
max_variants bounds the expansion. Common fixed XYZ/SMILES sources can be added
beside an input_dir source. Charges and multiplicities are shared across variants.
