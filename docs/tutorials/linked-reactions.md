# Linking reactions through optimized intermediates

A `mechanism` workflow declares a sequence before any calculations run. A later
reactant can select a complete molecule from an earlier stage's **optimized product
endpoint**. The geometry is resolved at runtime, so no intermediate XYZ is needed
in advance. Existing linear and single-`reaction` workflows remain available. Every product,
including the first, can use a small [product core](#product-cores-from-the-first-stage).
The first product inherits omitted ligands and substituents from the supplied starting
reactants; later products inherit them from optimized intermediates.

Start with [the linked-reaction example](https://github.com/sterling-group/ChemRefine/tree/main/examples/linked_reactions).
It illustrates methanediol dehydration followed by rehydration with a new water
molecule. The example demonstrates handoff and atom accounting; it is not a
verified elementary mechanism. Configure the executable/backend installations and
appropriate chemistry before calculating it. Install the automatic mapper in the
driver environment (its pretrained model is included in the package):

```bash
pip install -e ".[mapping]"  # From the ChemRefine source checkout
```

Each stage runs its reactant preparation, product preparation, TS guessing/scoring,
and TS refinement branches. Only after that stage completes does the next stage
resolve its product selections. The carried coordinates come from the product
minimum branch, never from the TS geometry.

## Declare a mechanism

At the top level, use `mechanism` instead of `input`, `steps`, or `reaction`.
Usual machine settings, templates, dispatch, and budgets still apply.

```yaml
mechanism:
  conformers: 8
  seed: 42
  fragment_gap_angstrom: 3.0
  stages:
    - name: dehydration
      reactant:
        sources:
          - smiles: 'OC(O)'
        steps:
          - step: 1
            engine: orca
            template: minimum.inp
      product:
        smiles: 'C=O.O'
        steps:
          - step: 1
            engine: orca
            template: minimum.inp
      ts_guesser:
        method: idpp
        scoring_steps:
          - step: 1
            engine: mlip
            operation: sp
            template: score.py
            options: {task_name: omol, device: cpu}
      ts_steps:
        - step: 1
          engine: orca
          template: ts.inp

    - name: rehydration
      reactant:
        sources:
          - from: dehydration
            select_smiles: 'C=O'
          - smiles: 'O'
        steps:
          - step: 1
            engine: orca
            template: minimum.inp
      product:
        smiles: 'OC(O)'
        steps:
          - step: 1
            engine: orca
            template: minimum.inp
      ts_guesser:
        method: idpp
        scoring_steps:
          - step: 1
            engine: mlip
            operation: sp
            template: score.py
            options: {task_name: omol, device: cpu}
      ts_steps:
        - step: 1
          engine: orca
          template: ts.inp
```

`minimum.inp`, `score.py`, and `ts.inp` are supplied by the elementary-reaction
example. The full linked example also includes a separate frequency calculation
in each stage's TS refinement. You can use the existing guessers and multiple
calculation steps in every preparation/refinement branch.

Stage names must be unique filesystem-safe names. `from` names an **earlier stage**;
forward references and cycles are refused. Each branch numbers its own steps from
1. A stage may set `charge` and `multiplicity`; otherwise it inherits the workflow
values. Endpoint formal charges must match the stage charge, and step charge/spin
overrides must agree with it. Individual molecules selected from a mixture may
have different charges or spin states from the earlier full system: specify the
new stage's values explicitly rather than treating the mixture's spin as a property
of its selected component.

## Ordinary SMILES and automatic identities

**Users do not need to choose atom identifiers.** The declaration determines
whether a molecule is carried or new:

- `from` alone carries an earlier optimized product when it is a single molecule.
  With multiple product molecules, `from` plus `select_smiles` selects one complete molecule. Its existing identities and calculated coordinates are retained.
- `smiles` declares a **new reagent instance**. Every atom receives a fresh identity,
  even if the same SMILES appeared earlier. Old byproduct identities are never
  recycled. Supplied legacy labels do not request reuse; use `from` for that.
- `product.smiles` defines the complete target connectivity, including byproducts.
  RXNMapper maps its heavy atoms to the assembled reactants, and ChemRefine translates
  that reaction-local correspondence into the persistent identities of carried and
  newly introduced atoms.

The two waters in the example are distinct instances: the old water remains in
`dehydration`, while `smiles: O` introduces a new water in `rehydration`.

The mapper uses the real packaged [RXNMapper model](https://github.com/rxn4chemistry/rxnmapper).
It receives canonically serialized heavy-atom reaction SMILES with preserved
traversal-index bindings. Returned correspondences must form an element-preserving
bijection over all heavy atoms. Atom inventories, charge, and multiplicity are
checked before submission. A mapping is a proposal for a TS search, not evidence
that the chemical pathway or transition state is correct.

`mechanism.min_mapping_confidence` optionally rejects mapper scores below a chosen
threshold (default `0.0`, no score-based rejection). Scores are recorded for review;
they are not calibrated probabilities of a correct elementary mechanism. A missing
model installation, invalid correspondence, or failed confidence gate stops before
any calculation jobs run.

### Hydrogen transfers

Hydrogens are expanded automatically. The resolver first preserves unchanged
hydrogen-only species and hydrogens that can remain on the same mapped parent,
then assigns remaining transfers when there is one donor group or one acceptor
group. Equivalent H positions are assigned deterministically at the molecular-graph
level. This does not enumerate every stereochemical or geometric reaction channel.

If several distinct donor groups and several distinct acceptor groups leave
multiple transfer assignments, the resolver stops instead of silently selecting a
proton pathway. Split the chemistry into elementary steps, or supply a complete
mapped override for that particular reaction. Unbalanced reactions still require
the missing products/byproducts to be specified; automatic mapping does not invent
them. Isotopically labelled mechanism SMILES remain unsupported.

### Carrying a single product automatically

When a previous stage has exactly one connected product molecule, omit the selector:

```yaml
reactant:
  sources:
    - from: reaction_1
```

ChemRefine carries that entire optimized molecule and its persistent atom identities.
The choice is resolved from the declared graph before any jobs run and checked again
at runtime. If there are multiple product molecules (including separate counterions),
`select_smiles` is required; ChemRefine never silently picks the first fragment.
`select_index` without a selector cannot bypass this requirement.

This omits only the **intermediate selector**. A later stage must still define
its target using complete `product.smiles`, a complete product XYZ `input`, or a
small `product.core`. A sole previous product determines the next reactant; it
does not determine the intended next transformation.

### Selecting one of several matching molecules

A selector matches a **whole molecule**, with connectivity, bond orders, formal
charge, radicals, hydrogen count, and any specified stereo. It cannot cut a
substructure out of a larger product. Ordinary `select_smiles: C=O` is sufficient
when one formaldehyde molecule matches.

If several complete molecules match, use `select_index: 0` (zero-based occurrence)
to select an instance without typing atom identifiers. Occurrences follow the
resolved product's internal component order; inspect the mapping/geometry records
when distinguishing different poses of chemically identical molecules matters.
Existing explicitly mapped selectors also remain supported.

### Compatibility with mapped inputs

Fully mapped product declarations remain an explicit override and do not require
RXNMapper. Their labels are translated through carried identities and any fresh
reagent allocations. Existing mapped workflows retain their input format, including
the parent-based reserved identifiers used for implicit Hs in that format. Complete
ordinary SMILES are the recommended user-facing input; partial product map labels
are not constraints on the automatic mapper.

Persistent identifiers are stored in sidecars and diagnostic diagrams. Calculation
engines and custom templates must still preserve the canonical input atom order
when returning optimized geometries. Graph checks catch inequivalent permutations;
symmetry-equivalent atom permutations cannot always be detected from an unlabeled
XYZ alone.

## Geometry generation and validation

For a new reagent or product supplied only as SMILES, the runner samples
`conformers` RDKit ETKDG conformers
per molecule and selects the lowest-energy converged MMFF conformer, using UFF
when MMFF parameters are unavailable. The random `seed` makes generation
reproducible. Failure to embed, missing force-field parameters, or failure to
converge stops the stage with instructions to supply a geometry.

Additional molecules are translated along the x axis with the configured
`fragment_gap_angstrom` between their bounding boxes. Internal molecular
geometries are preserved; the first source retains its coordinates. This is a
deterministic starting arrangement, **not an exhaustive encounter-complex or
orientation search**. Product conformers are rigidly aligned to the mapped
reactants before the calculation branches start. Conformational changes and
reaction approach geometries can still determine whether a TS search succeeds.

Supply `input: /path/to/complex.xyz` under `reactant` or `product` to use a prepared
full endpoint geometry. A new reagent source may also supply `input` together with
its `smiles`. To use an XYZ-only first reactant, declare
`reactant.sources: [{input: ./reactant.xyz}]`; a complete XYZ product uses
`product.input` without `smiles`. XYZ rows can be in arbitrary order: the runner perceives the input
graph and matches it to the declared SMILES, including stereo, then reorders rows
internally. Include all atoms and explicit hydrogens. A full reactant `input` takes
precedence over per-source geometry generation;
product selections are still resolved and recorded. Supplied geometries are
validated too. The original row-to-internal-index binding is recorded in endpoint
JSON (`input_atom_indices`), and in `handoff.json` for file-backed reagent sources.
Inspect `--dry-run` for the resolved reaction mapping and confidence score.

For ordinary covalent systems, seed geometries and optimized endpoints are checked
against their declared graphs using coordinate-based bond perception at the
specified charge. Expected stereochemistry is checked; equivalent resonance forms
are allowed. Metal endpoint checks validate element identities, adjacency, and
coordination distances; they do not establish oxidation states or measure bond
orders from the optimized electronic structure. Changed
connectivity, failed perception, mismatched atom order, or incompatible stereo
stops the workflow before guessing or forwarding the affected endpoint. Bond
perception is most suitable for ordinary covalent molecular structures; metal
coordination, ambiguous bonding, and unusual electronic structures may be refused.
A matching graph establishes product identity, not a verified local minimum or TS:
frequency calculations and IRC checks remain necessary.

## Outputs, byproducts, and resume

```bash
chemrefine validate input.yaml
chemrefine run input.yaml --dry-run
chemrefine run input.yaml
chemrefine resume input.yaml
```

Each named stage has its own `reactant/`, `product/`, `scoring/`, and `ts/` calculation
branches, plus:

- `mapping.json`, with backend version, confidence, resolved reaction, fresh source
  allocations, and hydrogen transfers;
- `_inputs/reactant.xyz` and `product.xyz`, with JSON graph/map bindings;
- `endpoints/reactant.xyz` and `product.xyz`, containing validated optimized
  endpoint geometries with their bindings;
- `handoff.json`, recording previous-stage selections, original coordinate
  indices, retained map identifiers, and identifiers excluded from each source stage;
- ordinary guesser, cache, failure-ledger, and TS refinement artifacts.

In the example, the first water remains in the first stage's optimized product
record. The second stage carries only formaldehyde and introduces a new water with
a disjoint set of identities. Dropping a byproduct between reactions does not
remove it from the earlier TS search or its record.

Resume resolves references again through each stage's fingerprint-checked
optimized outputs. Deterministic seed generation permits unchanged calculations
to cache-hit; changed upstream coordinates invalidate affected downstream
calculation inputs. Selected species are new geometry seeds with **no inherited
energy from the full product complex**. Their energies must be calculated in the
new stage. No mechanism-wide energy or barrier is inferred by subtracting energies
of systems with different compositions.

`mechanism.json` records overall refinement completion and explicitly marks TS
verification as false. Later stages do not start if an earlier stage fails.
Whole-mechanism `run` and `resume` are supported; scoped recovery commands remain
available for ordinary linear workflows. The GUI can load, edit, validate, and run
a mechanism through a dedicated YAML settings panel.

## Final geometries in one place

### Run the same template for a directory of reactants

Use `input_dir` in any fresh reactant source to run each XYZ file as an independent
alternative. Each file must contain one complete starting structure for that
source. Do not put alternatives into one multi-frame XYZ: a directory creates
independent mechanisms rather than a conformer ensemble.

```yaml
max_cores: 3           # Total CPU budget across all concurrent variants.
max_gpus: 0           # Explicit for parallel batches; these are CPU runs.
mechanism:
  parallel_runs: 3
  max_variants: 64
  stages:
    - name: first_reaction
      charge: -1
      multiplicity: 2
      reactant:
        sources:
          - input_dir: ./reactants_by_catalyst
            charge: -1
          # Fixed reagents may also be supplied as additional XYZ or SMILES sources.
        steps: &batch_minimum
          - step: 1
            engine: mlip-extopt
            template: minimum.inp
            options: &batch_uma {task_name: omol, model_name: uma-s-1p2, device: cpu}
      product:
        core: {input: ./product_core.xyz, anchor: 1}
        steps: *batch_minimum
      ts_guesser:
        method: idpp
        n_images: 9
        scoring_steps:
          - step: 1
            engine: mlip-extopt
            operation: sp
            template: single_point.inp
            options: *batch_uma
      ts_steps:
        - step: 1
          engine: mlip-extopt
          template: ts.inp
          options: *batch_uma
```

If the directory contains `CuCl2.xyz`, `CuBr2.xyz`, and `CuI2.xyz`, the same
mechanism is evaluated separately under `output_dir/variants/CuCl2/`,
`variants/CuBr2/`, and `variants/CuI2/`. Paths are relative to the YAML.
Only direct child files ending in `.xyz` (case-insensitive) are included, in
sorted order. File stems must be unique and use letters, digits, `_` or `-`.
Multiple directory sources form a Cartesian product, capped by `max_variants`.
All alternatives share the configured charge/spin: use a separate batch for a
different electronic state. Each elementary reaction still conserves all atoms.

Explicit validation exceptions can be kept in the **same master YAML**, keyed by
the XYZ filename stem:

```yaml
mechanism:
  parallel_runs: 3
  force: false
  variant_overrides:
    CuI2:
      force_endpoint_contacts: true
  # Keep the same directory source, cores and stages for all three catalysts.
```

`force_endpoint_contacts` is broader than the metal-only `force` option: in
metal-containing endpoints it permits ordinary as well as metal contact changes,
including an I–O contact. Atom identities, finite coordinates, nonperiodicity,
severe-overlap, charge/spin and calculation-success checks remain active. Pure
organic endpoint validation is not relaxed. The result is exploratory and still
requires independent endpoint/TS and IRC checks. This option does not prove that
the template's intended chemistry occurred. Accepted changes and the override
scope are written to the stage logs and final manifests.

Only `force` and `force_endpoint_contacts` can be overridden per variant. The
exception applies only to the named alternative; the other catalysts retain the
master defaults. Unknown variant names are rejected before jobs start. Concrete
worker inputs are generated automatically; no separate user-maintained YAML or
wrapper is needed. Broader contact overrides use their own core-cache namespace.

`parallel_runs` caps simultaneous **mechanisms**. Every variant runs in its own
driver process and output tree, with separate caches and native job logs. The
total CPU budget is divided among workers; a GPU budget is divided the same way
and must be explicit for parallel batches (`0` for CPU). Configure each step to
fit the per-worker budget. Failures in one mechanism do not cancel the others;
the batch exits nonzero if any variant fails and records each result in
`batch.json` and `index.html`. A failed refinement is never marked TS-validated.
Each successful variant exports its usual `final/` structures, outputs and
frequency summaries. `variant.yaml` and `driver.log` record the actual input and
log of every variant, including failures.

Use the normal `chemrefine run input.yaml`, `resume input.yaml`, and `--dry-run`
commands. Resume reuses each variant's own cache. Planning validates all expanded
chemistries before submitting any jobs. Changing XYZ contents goes through the
existing geometry cache identity; selecting another catalyst never borrows a
different variant's results.

Custom TS generators, including force-relaxed path searches, can use
`ts_guesser.method: external` with a `command` argument vector. The adapter receives
`{reactant}`, `{product}`, `{output}`, `{charge}` and `{multiplicity}` placeholders
and must write ordered, atom-conserving candidate XYZs to `{output}`. The same
endpoint caching, TS refinement and candidate validation used by the named model
adapters apply. An interpolated or force-relaxed path image remains a guess until
the refined saddle passes frequency and IRC checks.

After every stage completes, open **`output_dir/final/index.html`**. This standalone
page needs no server or internet connection. It lists stages in reaction order,
with download links for the optimized reactant, every surviving final TS
candidate, and optimized product. Each structure also has a directory containing
its geometry, frequency summary and matching native calculation output:

```text
outputs/final/
  index.html
  001_dehydration_reactant.xyz
  001_dehydration_ts_001.xyz
  001_dehydration_product.xyz
  002_rehydration_reactant.xyz
  002_rehydration_ts_001.xyz
  002_rehydration_product.xyz
  001_dehydration/
    reactant/
      geometry.xyz
      calculation.out
      frequencies.json
    ts_001/
      geometry.xyz
      calculation.out
      frequencies.json
    product/
      geometry.xyz
      calculation.out
      frequencies.json
  mechanism.xyz
  manifest.json
  README.txt
```

TS geometries come from the **last TS refinement step's survivors**, never from
the guesser. Multiple surviving candidates receive separate numbered files.
Endpoint files are the same optimized geometries used by runtime handoffs.
The calculation output is copied from the selected structure's canonical job,
identified through the step manifest. Ensemble children use their explicit parent
job; outputs from other candidates, archived retries or earlier steps are never
substituted. The output retains its original extension (`.out` above).
Open it to inspect the program's full frequency table. `frequencies.json` also
contains all parsed vibrational frequencies and the imaginary subset, for both
endpoints and TS candidates. `null` means frequency data were unavailable;
an empty imaginary table means no imaginary frequencies were found. If no endpoint
optimization was requested, or a native output was not retained, the manifest's
`output` is `null` and the geometry and available summary are still exported.
`mechanism.xyz` combines the stage's reactant, TS candidates, and product in that
order, with labeled XYZ comments. Because stages can gain or discard molecules,
frame atom counts may differ; this collection is not a continuous trajectory.

`manifest.json` binds each file and combined frame to its stage, role, persistent
atom identifiers, charge, multiplicity, and source path relative to `output_dir`.
Entries also link the per-structure geometry, output and frequency summary;
electronic energies and imaginary frequencies are included when available.
TS entries retain the structure ID. These fields are evidence to inspect, not automatic
TS verification or a mechanism-wide energy profile.

The `final` folder is managed: `run` and `resume` invalidate earlier generated
results after acquiring locks and passing preflight, and regenerate them only
when the entire mechanism completes. An interrupted or failed run leaves an
explicitly incomplete index without stale generated XYZs. Original calculation
folders remain available. Resume an existing completed mechanism to generate this
collection from cached results without submitting unchanged calculations.

<a id="complete-a-later-product-from-a-small-core"></a>

## Product cores from the first stage

Any stage can supply `product.core` instead of a complete product SMILES or
XYZ. The core contains a metal placeholder and the reacting atoms in their intended
product arrangement. Provide the full catalyst and starting reactants once, as XYZ
or SMILES sources. The first core is matched to those atoms, and omitted ligands and
substituents are retained from their starting geometries. No complete first product
is required. Later stages use carried **optimized product coordinates** together with
explicitly supplied new reagents.

The [all-core mechanism example](https://github.com/sterling-group/ChemRefine/tree/main/examples/product_core_from_start)
supplies one complete starting XYZ, a five-atom oxidative-addition core, and a
three-atom reductive-elimination core. Each stage retains all 22 atoms, including
the omitted phosphines and substrate hydrogens. Reuse the same cores when replacing
compatible metals, spectator ligands, or substituents in the starting structures;
the chosen transformation and reacting sites must still be chemically applicable.
The cores alone cannot specify omitted catalyst ligands, substrate substituents,
or missing reagents: their full compositions must be supplied as reactant sources.

The first stage can use the same core syntax as later stages:

```yaml
- name: first_reaction
  reactant:
    sources:
      - input: ./starting_catalyst_and_reactants.xyz
  product:
    core:
      input: ./first_product_core.xyz
      anchor: 1
```

Add your existing endpoint preparation, TS guesser, and TS refinement settings,
as in the complete example. Separate catalyst and substrate sources are also
accepted. Default fragment placement remains a starting arrangement; it does not
search every possible reactant encounter geometry.

See the [complete product-core example](https://github.com/sterling-group/ChemRefine/tree/main/examples/product_core).
It supplies complete XYZ endpoints for Pd oxidative addition, followed by a
three-atom core for ethane reductive elimination. All 22 atoms, including the
inherited phosphines, remaining methyl group, iodide, and hydrogens, remain in the
assembled product. These inputs illustrate assembly and execution; they do not
establish a verified catalytic mechanism or guarantee TS convergence.

The following fragment supplies the sources and product core for a later stage.
Reuse your existing endpoint preparation, TS guesser, and TS refinement settings
on this stage, as shown in the complete example.

```yaml
- name: next_reaction
  reactant:
    sources:
      - from: previous_reaction
      # Optional NEW reagent; choose SMILES or an XYZ file:
      - smiles: N
      # - input: ./new_reagent.xyz
  product:
    core:
      input: ./next_product_core.xyz
      anchor: 1
      placements: 8
```

Paths are relative to the input YAML. `anchor` is a **one-based row of the small core**,
not a manually assigned catalyst identifier. Its real element and persistent
identity come from the supplied starting catalyst when there is no carried source.
When a stage carries an intermediate, the anchor must come from that intermediate;
a fresh metal-containing reagent cannot silently replace it. Every other core atom
must exist in the assembled reactants.
A new ligand must therefore be supplied as a new reagent, even if it appears in
the core. Include atoms involved in hydrogen transfers explicitly in the core.

Omitted attachments are retained. The template changes bonds **between its matched
atoms** and preserves bonds crossing from the core into omitted groups. Include
enough of the reacting fragment to distinguish it from ligand backbone atoms and
to specify all intended bond changes. For example, reductive C–C elimination
includes the metal and both coupling carbons, with the carbons bonded to each
other and separated from the metal in the product core.

For XYZ-only sources and complete products, `input` no longer requires accompanying
SMILES or atom-by-atom bond lists. Ordinary molecular bond perception is used
for complete organic molecules. For metal complexes and reduced cores, ChemRefine
infers contacts from geometry and proposes common organic multiple bonds jointly
under valence limits. A metal's contacts to both ends of a multiple bond are
treated as pi coordination; an isolated metal-carbon attachment is sigma bonded.
This lets an alkyne core distinguish the reacting carbons without user bond indices.

To carry the catalyst-containing molecule and leave water or other disconnected
byproducts behind, use a catalyst-independent selector:

```yaml
reactant:
  sources:
    - from: previous_stage
      select: metal_complex
    - input: ./new_reagent.xyz
      charge: 0
```

`metal_complex` requires exactly one metal-containing component; it refuses an
ambiguous mixture with two metal complexes. This selector also works with SMILES
sources. It does not delete byproducts from the elementary reaction's atom inventory.

Geometry-derived bond orders guide correspondence; they do not prove an electronic
structure. Oxidation states and individual formal charges are not inferred for
the metal complex. The exported endpoint metadata marks that distinction.
Contacts near the adjacency threshold or severe overlaps stop with a geometry
diagnostic instead of asking the user to write a bond table. Provide a clearer
structure when that happens. Legacy explicit bond metadata remains supported.
Set stage charge/multiplicity when the electronic state changes; charged
new reagents require an explicit next-stage charge. Selecting only part of a
distance-only XYZ mixture requires an explicit stage charge and multiplicity,
because the selected component's electronic state cannot be inferred. Metadata
belongs to XYZ-only inputs; SMILES already defines its own connectivity and charge.

The [geometry-driven A3 example](https://github.com/sterling-group/ChemRefine/tree/main/examples/a3_geometry_cores)
contains three stages and no bond lists. Its `input_CuBr.yaml` and `input_CuI.yaml`
reuse exactly the same product-core XYZs. They supply a monohalide starting complex
and a singlet state in place of the illustrative CuCl2 doublet model. These are
starting models, not validated CuBr/CuI pathways: optimize endpoints and verify
the TS frequency and IRC for each catalyst. Changing the catalyst can change the
preferred pathway, coordination and spin state.

### Inherited ligands, alternatives, and failures

To continue when an optimized endpoint changes **metal contacts**, opt in with:

```yaml
mechanism:
  force: true
  # Keep the existing stages and other mechanism settings here.
```

The default is `false`. Force permits only contact changes involving a metal
atom, including at carried endpoints and core assembly. Ordinary-atom adjacency,
atom identities, finite coordinates and severe-overlap checks remain enforced.
A new I–O contact, for example, is not a metal-contact change and still stops
under `force` alone. Only the separate, explicit `force_endpoint_contacts` option
described above permits that broader exception.
This setting does not override failed calculations, charge/spin validation or
missing TS candidates, and does not relax separate IRC-validation tools.

The intended graph remains the template's nominal target; it is not silently
rewritten to certify whatever coordination optimization produced. Each stage
writes `coordination_overrides.json` with the accepted added/removed contacts,
one-based rows and persistent atom maps. Actual overrides emit a warning. The
final results and `mechanism.json` mark force mode explicitly; endpoint/TS
identity still requires independent inspection and IRC validation. Core workflows
use a separate cache namespace in force mode, so strict and forced assemblies
cannot be mistaken for each other. Directory variants inherit the setting.

Inherited connected groups move together initially, including chelating ligands.
ChemRefine samples placements, keeps the core fixed, and relaxes nonbonded overlaps
with restraints on inherited distances. Placements that lose connectivity, form
unintended bonds, or deform inherited distances by more than 0.15 Å are rejected.
The selected seed minimizes a **geometric penalty**, not electronic energy. Your
configured minimum steps then optimize the complete endpoint before TS guessing.

Omission never removes a ligand. To dissociate one, use a selector that matches
exactly one whole ligand after cutting its metal contacts:

```yaml
core:
  input: ./next_product_core.xyz
  dissociate: ['[PH3]']
```

The ligand remains in the stage as a separate molecule. To replace it, also supply
the incoming ligand as a new reagent and include its metal contact in the core.
Unwanted byproducts can be excluded only when selecting the next stage's sources.
A SMARTS selector is also accepted; it must match the entire explicit-H ligand.

Exact graph symmetries are deduplicated. Chemically distinct matches become named
`__candidate_001`, `__candidate_002`, … branches, and later stages follow each
parent candidate. Shared ancestry must agree when branches merge. More than
`max_candidates` alternatives or an excessive mapping search stops execution with
a diagnostic; candidates are never silently truncated. Narrow the core or provide
optional `match` hints mapping core rows to persistent identities from `mapping.json`
when automatic matching cannot distinguish the intended site.

Each resolved stage records `mapping.json`, `handoff.json`, `assembly.json`, and
`assembly_candidates.xyz` under `outputs/_resolved/<chemistry hash>/<stage>/`.
The assembly record identifies core atoms, inherited atoms, new sources, the
selected placement, and geometric scores. Input graphs, core contents, matches,
and assembly settings participate in cache identity. Moving an intact workflow
to another directory does not change this chemistry identity.

The existing `outputs/final/index.html` and combined `mechanism.xyz` remain the
flat entry points for final structures across every candidate. Failed assembly or
endpoint validation stops the workflow and leaves that final collection incomplete.
A converged TS still needs frequency and, when required, IRC connectivity checks;
product assembly does not verify a mechanism.
