# Reproduce and inspect the evidence

## Portable CPU check

Python 3 standard library is sufficient:

```bash
python3 -B scripts/check_course_project.py
python3 -B -m unittest discover -s tests -p test_course_evidence.py -v
```

Expected: SMOKE_PASS and five passing targeted tests. The check validates bundle hashes, frozen P/A source identity, six runs, paired initialization, all 2700 score rows, patient clusters and the declared decision. It does not run training or infer masks.

The existing project environment in INSTALL.md supplies NumPy for the exact bootstrap:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -B scripts/check_course_project.py --bootstrap
```

To compare against local original artifacts, add `--archive-root /path/to/DiffOSeg_runs`. The portable CSV removes original patient identifiers and keeps pairing.

## Inspect the exact historical source

```bash
CLAIM_SOURCE=$(mktemp -d)
tar -xzf results/course_project/pa_source.tar.gz -C "$CLAIM_SOURCE"
cp results/course_project/pa_metrics_set.py "$CLAIM_SOURCE/metrics_set.py"
```

The 49-member archive preserves its historical source fingerprint. metrics_set.py was originally tracked as a separate dependency. Do not replace those files with current E11 modules. For independent CPU tests use the repository's installed interpreter with the extracted source as the working directory:

```bash
PROJECT_PY="$PWD/.venv/bin/python"
(cd "$CLAIM_SOURCE" && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PROJECT_PY" -B -m unittest discover -s tests -v)
```

After inspecting the isolated directory, remove only that directory when no longer needed. No training is part of the course smoke.

## Full historical replay prerequisites

The original P/A protocol, data checksum, seeds, six final checkpoints, case masks, split and gpu_usage.jsonl remain under 20261025_table3_pa_claim_v1. Its frozen runner requires the original path configuration and writes to its output. Do not invoke summarize/coordinator on that immutable directory. A new training reproduction needs a new output directory, resolved data paths, re-frozen source/configuration and an explicit compute allocation. This package provides analysis replay; it does not claim that paths can be changed while retaining the historical source identity.

For the extensions follow their existing protocol/results commands using each run's frozen source. Final E also depends on public runtime receipts, teacher outputs, feature caches and earlier guidance. A single checkpoint is insufficient.

## Data and environment

The packaged LIDC data contain 1609 patches from 214 patients, each 128x128 with four masks. Obtain the preprocessed file through the source in INSTALL.md, and verify SHA-256 against manifest.json. The preprocessed file's redistribution permission remains pending confirmation. No download occurs during smoke.

The original P/A environment specifies Python 3.11, torch 2.14.0+cu132 and the archived uv.lock. Current source installation uses uv sync --locked. No clean installation was performed during packaging; tests used the existing environment.

## Regenerate the package and documents

```bash
python3 -B scripts/prepare_course_evidence.py --archive-root /path/to/DiffOSeg_runs
python3 -B scripts/render_course_tables.py
bash scripts/build_course_documents.sh
```

The exporter reads old runs and writes only the local course evidence directory. The document builder uses the pinned Typst compiler and UM_Typst_Template submodule and private temporary build files; it does not change shared environments. PDF sources are editable. Instructor template/length acceptance remains a release item.

## User-selected Typst template

The template is a Git submodule at templates/UM_Typst_Template, pinned to ac1964afca6a99bb60cb29ed2c2ddece00eb2eea. Initialize it with `git submodule update --init templates/UM_Typst_Template`. The submodule is kept unchanged. Course content lives in report/ and presentation/.

The template has slides and posters, not an official paper-report template. The A4 report adapts its palette and logo. The slides retain its Touying 0.7.4 / Numbly 0.1.0 imports, white 16:9 canvas, UM logo and fixed template copyright footer.

`presentation/toolchain/uv.lock` pins the document exporter and Python Typst binding separately from the training environment. The build uses a private `.course-build/` environment/cache and regenerates both slide PDF and PPTX. The PPTX has full-slide images at 144 PPI; edit the `.typ` source, then rebuild. This is the selected template's export design, not an editable PowerPoint chart package.
