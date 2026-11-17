# Elementary reactions from reactants and products

A `reaction:` workflow prepares two endpoints, generates transition-state (TS)
candidates, and refines them using the same calculation steps as a conventional
ChemRefine workflow. Existing linear workflows and their recovery commands keep
working. The GUI includes an elementary-reaction setup button: it moves the
current steps into TS refinement and provides separate branch settings.

Start with [the complete example](https://github.com/sterling-group/ChemRefine/tree/main/examples/elementary_reaction).
It includes illustrative mapped H + H₂ endpoints, an MLIP single-point template,
and ORCA minimum optimization, saddle-point, and frequency templates. Configure your ORCA executable,
install the chosen MLIP backend (`chemrefine backends install mlip-fairchem` for FAIRChem),
and replace the endpoints and electronic-structure settings for your chemistry.
A geometrical path and a highest-energy image are initial guesses; convergence
to a chemically relevant saddle point is not guaranteed.

```yaml
reaction:
  reactant:
    input: reactant.xyz
    steps:
      - step: 1
        name: reactant_minimum
        engine: orca
        template: minimum.inp
  product:
    input: product.xyz
    steps:
      - step: 1
        name: product_minimum
        engine: orca
        template: minimum.inp
  ts_guesser:
    method: idpp
    n_images: 17
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
    - step: 2
      engine: orca
      template: freq.inp
```

The execution order is **reactant minimum preparation → product minimum
preparation → TS guessing from the final optimized endpoints → path scoring
when needed → TS optimization → TS frequency analysis**. Each preparation branch
can contain several steps, such as a low-cost optimization followed by a DFT
optimization. `minimum.inp` uses `Opt Freq`; inspect its frequency results to
confirm that the endpoints are minima. Use `steps: []` only when an endpoint is
already prepared. The generated TS guess enters the separate `ts_steps` pipeline.

Retain the usual top-level machine settings (`executables`, resource budgets,
SLURM headers, and directories). Omit top-level `input` and `steps` when declaring
`reaction`. Each branch numbers steps from 1 and has its own step-name namespace.
Endpoint preparation accepts existing engines, templates, filters, and failure
policies. Each endpoint input must initially contain exactly one structure; if
preparation creates conformers, filter them down to one before guessing.

Both endpoints must contain the **same atoms in the same mapped order**, including
all explicit hydrogens, and share total charge and multiplicity. Separate fragments
belong in one XYZ frame per endpoint. ChemRefine checks element order, finite
coordinates, and nonperiodic boundaries. It cannot infer correspondence between
same-element atoms: supplying the correct mapping remains essential. This first
implementation handles one elementary reaction at a time, without automated atom
mapping or reaction-network enumeration.

For multiple reactions with optimized intermediate handoffs and optional new
reagents, use a [linked mechanism](linked-reactions.md).

## Guessers

- `idpp` uses ASE's image-dependent pair potential interpolation. It requires no
  additional guesser package.
- `geodesic` uses the Martinez group's
  [geodesic-interpolate implementation](https://github.com/virtualzx-nad/geodesic-interpolate).
  Install it in the driver environment with `pip install 'chemrefine[reaction]'`.
  This package constructs a geometric path; the **separate scoring steps** evaluate
  the path with your selected MLIP or quantum-chemistry engine.
- `fragmentflow` and `react-ot` run the supplied
  [`scripts/ts_guess.py`](https://github.com/sterling-group/ChemRefine/blob/main/scripts/ts_guess.py)
  linker in each model's own environment. It loads the upstream checkpoint, prepares
  mapped endpoint tensors, samples a TS, and (for a fragmented core) reconstructs
  the full geometry. Checkpoints and the model environments are installed separately.

IDPP and geodesic generate `n_images` frames, align endpoint rotation/translation,
and evaluate every **interior** image. Scoring steps must explicitly use
`operation: sp`, `on_failure: stop`, no sample filters, and no NMS. The final
scoring step's highest electronic energy selects the TS seed; all interior images
must return finite energies. These calculations use the existing managed backend
environments and local/SLURM dispatch. This is path scoring, not climbing-image NEB.

## React-OT and FragmentFlow linker

The supplied **`scripts/ts_guess.py`** is a standalone linker for both repositories.
It does not require installing ChemRefine in the model environment. The two upstream
projects both install a package named `reactot`, so **use a separate environment for
each model**. Use the environment files from
[React-OT](https://github.com/deepprinciple/react-ot) and
[FragmentFlow](https://github.com/ronsh9/FragmentFlow), install their source code as
instructed there, and obtain a compatible pretrained checkpoint. For example, in
the chosen repository:

```bash
conda env create -f env.yaml           # React-OT
# For FragmentFlow, use fragmentflow.yaml instead (it also installs the project).
conda activate reactot                # fragmentflow for the other model
pip install -e .
```

Configure the linker using **absolute paths**. Keep the reactant and product minimum
preparation and TS optimization branches from the example; replace only
`reaction.ts_guesser`:

```yaml
  ts_guesser:
    method: react-ot
    command:
      - /absolute/path/to/reactot-env/bin/python
      - /absolute/path/to/ChemRefine/scripts/ts_guess.py
      - --model
      - react-ot
      - --repo
      - /absolute/path/to/react-ot
      - --checkpoint
      - /absolute/path/to/reactot-pretrained.ckpt
      - --reactant
      - '{reactant}'
      - --product
      - '{product}'
      - --output
      - '{output}'
      - --charge
      - '{charge}'
      - --multiplicity
      - '{multiplicity}'
      - --device
      - cpu
      - --solver
      - ode
      - --nfe
      - '100'
    timeout_seconds: 600
```

For **FragmentFlow**, change the environment interpreter, repository path,
checkpoint, `method`, and `--model` to their FragmentFlow values. The same script
then loads `reactot.utils.inference_tools.load_pytorch_checkpoint` (including EMA
weights when available), samples the core, and calls the authors'
`complete_fragmented_core_from_xyz` to reattach substituents in the original XYZ
atom order. The checkpoint must be a plain PyTorch checkpoint supported by that
upstream loader (`model` or `ema_model`, with its architecture arguments).

Core selection has three modes:

- Default: call upstream `process_single_reaction` with `--epsilon 0`, including
  its backbone, reactive-center, and hydrogen rules. This requires the optional
  `localmapper` stack, which the authors exclude from their base model environment
  because of dependency conflicts. Supply an environment where those optional
  dependencies work if using this mode.
- `--core-indices 0,2,3,7`: use explicit **zero-based original XYZ indices** (including
  any participating hydrogens) and bypass `localmapper`. Indices are sorted so the
  core and reconstruction use the same atom order. This is useful with an existing
  mapped reaction and a known core.
- `--full-system`: explicitly predict every atom without fragmentation. This can
  be useful for a small system; it does not provide the large-molecule benefit of
  reactive-core prediction.

For example, add these two strings to the FragmentFlow `command` list when the
core is already known:

```yaml
      - --core-indices
      - '0,2,3,7'
```

`--idpp-images` controls reconstruction (default 11). `--device cuda:0` requests a
GPU; CPU is the default. `--solver`, `--nfe`, `--method`, `--order`, `--diz`,
`--normalize`, `--atol`, `--rtol`, and `--seed` configure upstream sampling.
The linker checks that the checkpoint maps `R+P->TS` using `RP` initialization and
that `nfe` is below its diffusion timestep count. It builds the upstream five-type
H/C/N/O/F encoding for atoms sent to the neural network, and centers/aligns coordinates
without permuting atoms. FragmentFlow may retain other elements in substituents
when they are excluded from the predicted core; React-OT and full-system prediction
require every atom to fit that vocabulary.
The middle TS tensor slot is a midpoint placeholder: no reference TS is needed,
and the script does not compute evaluation metrics against that placeholder.

These initial linkers accept **neutral singlet reactions only**. The upstream
inference contracts do not condition on total molecular charge or spin, so passing
a nonzero charge or another multiplicity fails explicitly. This excludes the
illustrative H + H₂ doublet endpoint files in the IDPP example: supply suitable
neutral-singlet endpoints when switching that example to one of these models.
A model's atom vocabulary is not a guarantee of accuracy for all those chemistries;
consult its training-domain documentation.

The linker writes a single full-system TS frame to `--output` and a provenance
sidecar `<output>.json` with checkpoint hash, core indices, solver, seed, and the
upstream revision used as the API reference. ChemRefine preserves the sidecar as
`guesser/model.json` and captures console output in `guesser/model.log`.
The implementation was inspected against React-OT revision
`6dfccd0112cf6b504da80ad2516ff8e5ab8bade1` and FragmentFlow revision
`bf7a09fd61cec67828182d8442cd3cb351387878`; these are API reference revisions, not
claims about the revision of a user's installed checkout. Environments and weights
remain separate installations, and pretrained numerical inference needs validation
in the intended model environment before production use.

An opt-in pretrained-model smoke test is available in
`tests/test_ts_guess_linker_live.py`. Set
`CHEMREFINE_REACT_OT_PYTHON`, `CHEMREFINE_REACT_OT_REPO`, and
`CHEMREFINE_REACT_OT_CHECKPOINT` for React-OT; use the
`CHEMREFINE_FRAGMENTFLOW_` prefix for FragmentFlow, then run:

```bash
pytest -m integration tests/test_ts_guess_linker_live.py
```

It validates model loading and mapped finite XYZ output on a small neutral-singlet
pair. It does not establish saddle-point identity or prediction accuracy.

The command runs locally in a fresh work directory without a shell. Without
`scoring_steps`, its candidate enters `ts_steps` directly. Existing custom command
adapters remain supported.

## Running and inspecting

```bash
chemrefine validate input.yaml
chemrefine run input.yaml --dry-run
chemrefine run input.yaml
chemrefine resume input.yaml
```

`run` recomputes all reaction calculation branches and guesses. `resume` uses the
existing per-step fingerprints, retries pending failures, and resumes downstream
work. Guesses are cached using the endpoint geometries, charge, multiplicity,
and guesser settings; their saved XYZ bytes are checked before reuse. Changes to
external wrapper code or checkpoints at the same path require `run` to regenerate
guesses. Whole-reaction `run` and `resume` are supported; targeted recovery actions
are currently limited to ordinary linear workflows.

Outputs are separated into `reactant/`, `product/`, `scoring/`, and `ts/`, each with
its ordinary step caches, failure ledgers, reports, and job artifacts. `guesser/`
contains prepared endpoint snapshots, `path.xyz` for interpolation methods,
`candidates.xyz`, and `ts_guess.xyz`. `reaction.json` is written only after a
nonempty TS refinement completes. It explicitly records
`verified_transition_state: false`: inspect the frequency result for one
imaginary mode and establish IRC connectivity to both endpoints before assigning
the elementary mechanism. Frequency/IRC verification is not inferred from a
successful optimization, and no reaction barriers are reported automatically.
