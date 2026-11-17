# A product core for every stage

This example supplies a complete starting reactant once and uses reduced XYZ
cores for both products. It reuses the starting reactant and calculation templates
in `../product_core/` and does not read `product_step1.xyz`.

- `oxidative_addition_core.xyz`: metal placeholder, three reacting alkyl attachment
  carbons, and iodide. Including all three carbons specifies which C–I bond breaks
  without confusing the incoming alkyl with an existing metal-bound alkyl.
- `../product_core/product_core.xyz`: metal placeholder and two coupling carbons
  for reductive elimination.

Both complete endpoints contain all 22 atoms. Omitted phosphine ligands and
substrate hydrogens are inherited automatically. No atom identifiers are required.
Change the full starting reactant to supply a different compatible metal, ligand,
or substituent; omitted groups cannot be inferred from the cores alone.

Configure your ORCA executable and UMA environment/model paths in `input.yaml`, then:

```bash
chemrefine validate input.yaml
chemrefine run input.yaml --dry-run
chemrefine run input.yaml
```

This is an assembly/execution example. The proposed chemistry must still be
appropriate for the supplied reactants, and successful assembly does not establish
a minimum or transition state. Inspect frequencies and IRC connectivity.
