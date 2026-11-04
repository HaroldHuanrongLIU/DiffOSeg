# E11 LIDC-IDRI adaptation

Base: 3b877e7ccd9366a8f404a4e8cb9bff1cc9330ec7. The LIDC data are
`data/lidc_attributes.pkl`. The current changes reuse and narrow relevant
components from repository commit 3369610b6548e6e7ee018779c0cb87921098bcce;
that commit is not checked out and is not the training source by itself.

The configured data path expands `HOME` when loaded. The in-progress E11
experiment was frozen before this account-path portability change, so its
source fingerprint differs from the current checkout even though the data
path resolves to the same location. Resume that experiment only with its
original frozen source and output directory.

## Scope and protocol

This is an LIDC-only adaptation; NPC, baseline methods and ablations are not
included. Four patient-level folds follow the predecessor's first-occurrence
patient ordering, seed 1337 and four-fold split. The exact split is recorded;
the DiffOSeg paper does not publish its own seed or patient manifests. Each
fold is independent, and Stage II initializes from that fold's Stage I
checkpoint. The Stage I consensus target samples a nonempty expert subset and
averages its masks into a two-class soft distribution. Timestep sampling is
uniform across 1..250; cosine cumulative alpha is the product of individual
alphas. LIDC uint8 inputs are normalized to [0,1].

In the original data, 54 of 1609 cases have masks already ordered by
nondecreasing foreground area. Following the paper's simulated-preference
protocol, each sample's four masks are stably ranked by foreground area
before both training and evaluation; equal-area masks retain their supplied
order. These ranks are simulated style identities, not stable identities of
the radiologists across patients. The data file is not modified. The cited
predecessor ranks masks by area:
https://raw.githubusercontent.com/ycwu1997/D-Persona/70689229044dabc006743f439540473df1551abe/D-Persona/code/dataloader/utils.py

The reference configuration uses batch 12, Adam 1e-4, 250 cosine diffusion
steps, and 120000 / 90000 optimizer updates per fold. The paper prints the
Stage II count as `9,0000`; 90000 is an explicitly recorded interpretation.
The released image augmentation omits the paper's unspecified intensity
scaling. The linear learning-rate decay and EMA, preference averaging over ten
predictions, final-checkpoint choice, expert subset distribution, and fixed
patient split are documented implementation choices rather than verified
author settings. Held-out patients do not select checkpoints. Identity-aligned
expert Dice is reported separately from assignment-based consensus metrics.

Resume stores model, EMA, optimizer, scheduler, iteration and Python, NumPy,
Torch CPU and CUDA random states. The shuffled loader is restarted rather than
restored at its exact position, so a resumed training run is not bitwise
equivalent to an uninterrupted run. Evaluation stores case-level records and
masks to allow continuation. Source and dataset hashes and resolved config
must match before resuming. Do not run concurrent processes against one output
directory; the records and checkpoints have a single writer.

## Short validation

Installations and GPU execution must follow the E11 guide. The smoke script
requests one scheduler-assigned GPU and does not set CUDA_VISIBLE_DEVICES.
Submit from this directory with a **new** output directory and an output log
path that already exists, for example:

```bash
sbatch --output="$HOME/.cache/diffoseg-setup/lidc-smoke-%j.log" \
  scripts/e11_lidc_smoke.sbatch "$HOME/.local/share/diffoseg-runs/UNIQUE_RUN"
```

It uses fold 0, 20 training updates per stage, and one held-out case per
stage. Stage I deliberately checkpoints and stops after step 10, checks saved
state, and then resumes the same run to step 20. Stage II initializes from
that checkpoint and is followed by both evaluations. Smoke results are
engineering validation only and are excluded from paper comparisons.

The runner requires `--fold` and uses batch 12 in the supplied configuration.
Its non-smoke mode contains the full budget and must only be launched when
separately authorized. For SLURM execution source
`$HOME/.local/share/diffoseg-runtime/activate.sh` before invoking the virtual
environment Python. Do not use legacy `ddpm_train.py` or `ddpm_eval.py` as
E11 entry points: they override CUDA visibility, and their data routing and
evaluation do not implement this protocol.
