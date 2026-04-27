# RT-A3C-MLN

Experimental code for **Real-time Markov Logic Network Structure Optimization
with Asynchronous Advantage Actor-Critic**.

The project studies online Markov Logic Network (MLN) rule-pool optimization for
SWaT anomaly-detection windows. The main implementation combines:

- adaptive MLN rule mining over stream windows,
- a warm-start classifier for rule keep/disable decisions,
- A3C fine-tuning for sequential rule selection,
- protocol-level comparisons, ablations, and paper-figure generation.

The paper-aligned implementation is under `src/v2/` and `scripts/v2/`. The
older pipeline is kept under `src/pipeline/` and top-level `scripts/` for
traceability.

## Repository Layout

```text
RT_A3C_MLN/
  data/
    raw/swat/            # local raw SWaT CSV files, not committed
    processed/swat/      # generated processed CSVs, not committed
    stream/swat/         # generated window plans/features, not committed
  notebooks/             # exploratory preprocessing and prototype notebooks
  scripts/
    v2/                  # paper-aligned experiment drivers
    select_formal_windows.py
    run_a3c_training.py  # legacy v1 driver
  src/
    adapters/            # named comparison-adapter stubs/wrappers
    analysis/            # formal window selection helpers
    data/                # transaction utilities
    mln/                 # MLN rule utilities
    pipeline/            # legacy v1 experiment pipeline
    plot/                # legacy plotting helpers
    rl/                  # rule-selection environments and policies
    v2/                  # paper-aligned pipeline, policy, and rule pool
  outputs/               # generated logs, figures, checkpoints, not committed
  requirements.txt
```

Generated data, figures, logs, and model checkpoints are ignored by Git. The
directory placeholders are committed so a fresh clone has the expected layout.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

CUDA is optional. The scripts use PyTorch and will run on CPU unless your local
PyTorch installation can see a CUDA device.

## Data

The experiments use the SWaT cyber-physical testbed dataset. SWaT is released
under a data-use agreement, so this repository does not include raw or processed
SWaT files.

Place the raw files under:

```text
data/raw/swat/
  Normal.csv
  Attack.csv
```

If your SWaT download uses longer official filenames, either rename the two
CSVs to `Normal.csv` and `Attack.csv`, or update the first preprocessing
notebook cells accordingly.

Then run the preprocessing notebooks in order:

```text
notebooks/02_inspect_swat.ipynb
notebooks/03_build_transactions_fixed_clean.ipynb
notebooks/05_make_stream_windows.ipynb
```

The v2 scripts expect these generated artifacts:

```text
data/processed/swat/y_filled.csv
data/stream/swat/windows_mixed.csv
data/stream/swat/window_binary_data/window_<window_id>_binary.csv
```

## Reproducing The Main v2 Results

First build the formal mixed-window plan:

```bash
python scripts/select_formal_windows.py \
  --per-bin 12 \
  --min-window-gap 2 \
  --out-name formal_mixed_windows_plan.csv
```

Then run the high-heldout v2 pipeline used for the main paper tables and
figures:

```bash
python scripts/v2/run_all.py \
  --high-heldout \
  --seeds 0 1 2 3 4 42 \
  --episodes 100 \
  --workers 4
```

For a quick smoke run before a full reproduction:

```bash
python scripts/v2/run_all.py --high-heldout --seeds 42 --episodes 2 --workers 1
```

The orchestrator runs Protocol A, high-heldout Protocol B, baselines,
comparison adapters, ablations, detection-utility calibration, runtime
measurement, and figure generation.

## Running v2 Steps Individually

```bash
python scripts/v2/run_protocolA.py
python scripts/v2/run_protocolB_high_heldout.py --seeds 0 1 2 3 4 42 --episodes 100 --workers 4
python scripts/v2/run_baselines.py --high-heldout --seeds 0 1 2 3 4 42
python scripts/v2/run_adapter_methods.py --high-heldout --seeds 0 1 2 3 4 42 --episodes 100
python scripts/v2/run_ablation_pool_high.py --seeds 0 1 2 3 4 42 --episodes 100 --workers 4
python scripts/v2/run_detection_utility.py --high-heldout --seeds 0 1 2 3 4 42
python scripts/v2/run_runtime.py --high-heldout --episodes 100
python scripts/v2/make_paper_figures_v2.py
```

Outputs are written to:

```text
outputs/logs/v2/
outputs/figures/v2/
outputs/v2/models/
```

## Comparison Adapters

The named adapters in `src/adapters/` provide the plumbing for comparison
tables:

```text
src/adapters/mln_family/adapters.py
src/adapters/detector_family/adapters.py
src/adapters/rl_family/adapters.py
```

They are intentionally lightweight stubs or deterministic ranking heuristics so
the full pipeline can run without third-party competitor code. To report real
external baselines, replace each adapter's prediction/training method with a
wrapper around the corresponding authors' implementation, then rerun the v2
comparison scripts.

## Legacy v1 Pipeline

The legacy scripts are still available for auditability and for reproducing
older intermediate experiments:

```bash
python scripts/run_formal_mixed_experiment_strict.py \
  --plan-csv data/stream/swat/formal_mixed_windows_plan.csv

python scripts/run_cross_window_experiment.py \
  --plan-csv data/stream/swat/formal_mixed_windows_plan.csv

python scripts/run_a3c_training.py \
  --plan-csv data/stream/swat/formal_mixed_windows_plan.csv

python scripts/make_paper_figures.py
```

Each of these scripts now defaults `--project-root` to the repository root, so
the shorter commands above work when run from this checkout.

## Development Notes

- Keep raw SWaT data, generated windows, logs, figures, and checkpoints out of
  Git; `.gitignore` already covers them.
- Notebooks are retained as preprocessing/prototyping history. The script-based
  v2 pipeline is the preferred reproducibility path.
- Add any camera-ready BibTeX, DOI, and license information before the public
  release if required by the venue.
