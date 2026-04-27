# RT-A3C-MLN

Experimental code for **Real-time Markov Logic Network Structure Optimizationwith Asynchronous Advantage Actor-Critic**.

The project studies online Markov Logic Network (MLN) rule-pool optimization forSWaT anomaly-detection windows. The main implementation combines:

* adaptive MLN rule mining over stream windows,
* a warm-start classifier for rule keep/disable decisions,
* A3C fine-tuning for sequential rule selection,
* protocol-level comparisons, ablations, and paper-figure generation.

The paper-aligned implementation is under `src/v2/` and `scripts/v2/`. Theolder pipeline is kept under `src/pipeline/` and top-level `scripts/` fortraceability.

## Repository Layout

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

Generated data, figures, logs, and model checkpoints are ignored by Git. Thedirectory placeholders are committed so a fresh clone has the expected layout.

## Environment

Python 3.10 or newer is recommended.

    python -m venv .venv
    # Windows:
    .venv\Scripts\activate
    # macOS/Linux:
    # source .venv/bin/activate
    
    python -m pip install --upgrade pip
    pip install -r requirements.txt

CUDA is optional. The scripts use PyTorch and will run on CPU unless your localPyTorch installation can see a CUDA device.

## Data

The experiments use the SWaT cyber-physical testbed dataset. SWaT is releasedunder a data-use agreement, so this repository does not include raw or processedSWaT files.

Place the raw files under:

    data/raw/swat/
      Normal.csv
      Attack.csv

If your SWaT download uses longer official filenames, either rename the twoCSVs to `Normal.csv` and `Attack.csv`, or update the first preprocessingnotebook cells accordingly.

Then run the preprocessing notebooks in order:

    notebooks/02_inspect_swat.ipynb
    notebooks/03_build_transactions_fixed_clean.ipynb
    notebooks/05_make_stream_windows.ipynb

The v2 scripts expect these generated artifacts:

    data/processed/swat/y_filled.csv
    data/stream/swat/windows_mixed.csv
    data/stream/swat/window_binary_data/window_<window_id>_binary.csv

## Reproducing The Main v2 Results

First build the formal mixed-window plan:

    python scripts/select_formal_windows.py \
      --per-bin 12 \
      --min-window-gap 2 \
      --out-name formal_mixed_windows_plan.csv

Then run the high-heldout v2 pipeline used for the main paper tables andfigures:

    python scripts/v2/run_all.py \
      --high-heldout \
      --seeds 0 1 2 3 4 42 \
      --episodes 100 \
      --workers 4

For a quick smoke run before a full reproduction:

    python scripts/v2/run_all.py --high-heldout --seeds 42 --episodes 2 --workers 1

The orchestrator runs Protocol A, high-heldout Protocol B, baselines,comparison adapters, ablations, detection-utility calibration, runtimemeasurement, and figure generation.

## Running v2 Steps Individually

    python scripts/v2/run_protocolA.py
    python scripts/v2/run_protocolB_high_heldout.py --seeds 0 1 2 3 4 42 --episodes 100 --workers 4
    python scripts/v2/run_baselines.py --high-heldout --seeds 0 1 2 3 4 42
    python scripts/v2/run_adapter_methods.py --high-heldout --seeds 0 1 2 3 4 42 --episodes 100
    python scripts/v2/run_ablation_pool_high.py --seeds 0 1 2 3 4 42 --episodes 100 --workers 4
    python scripts/v2/run_detection_utility.py --high-heldout --seeds 0 1 2 3 4 42
    python scripts/v2/run_runtime.py --high-heldout --episodes 100
    python scripts/v2/make_paper_figures_v2.py

Outputs are written to:

    outputs/logs/v2/
    outputs/figures/v2/
    outputs/v2/models/

## Comparison Adapters

The named adapters in `src/adapters/` provide the plumbing for comparisontables:

    src/adapters/mln_family/adapters.py
    src/adapters/detector_family/adapters.py
    src/adapters/rl_family/adapters.py

They are intentionally lightweight stubs or deterministic ranking heuristics sothe full pipeline can run without third-party competitor code. To report realexternal baselines, replace each adapter's prediction/training method with awrapper around the corresponding authors' implementation, then rerun the v2comparison scripts.

## Legacy v1 Pipeline

The legacy scripts are still available for auditability and for reproducingolder intermediate experiments:

    python scripts/run_formal_mixed_experiment_strict.py \
      --plan-csv data/stream/swat/formal_mixed_windows_plan.csv
    
    python scripts/run_cross_window_experiment.py \
      --plan-csv data/stream/swat/formal_mixed_windows_plan.csv
    
    python scripts/run_a3c_training.py \
      --plan-csv data/stream/swat/formal_mixed_windows_plan.csv
    
    python scripts/make_paper_figures.py

Each of these scripts now defaults `--project-root` to the repository root, sothe shorter commands above work when run from this checkout.

## Development Notes

* Keep raw SWaT data, generated windows, logs, figures, and checkpoints out ofGit; `.gitignore` already covers them.
* Notebooks are retained as preprocessing/prototyping history. The script-basedv2 pipeline is the preferred reproducibility path.
* Add any camera-ready BibTeX, DOI, and license information before the publicrelease if required by the venue.
