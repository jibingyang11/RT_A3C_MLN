# RT-A3C-MLN Core Realtime Experiments

This workspace now keeps only the latest experiments designed to support thecentral claim: MLN rules can adapt to changing data streams while the foregroundresponse path remains realtime.

## Core Protocol

The main experiment uses prequential streaming evaluation:

1. A model predicts the current stream chunk.
2. The chunk labels arrive.
3. Adaptive methods update before the next chunk.

Batch MLN baselines must retrain or reselect rules on the rolling window aftereach chunk. `Static_MaxEnt_Reference` is reported only as a non-adaptivereference and is not treated as evidence of realtime adaptation.

## Run

    python scripts/run_core_realtime_experiment.py
    python scripts/run_core_ablation_experiment.py
    python scripts/plot_core_claim_results.py

Outputs:

* `results/core_claim/core_realtime_summary.csv`
* `results/core_ablation/core_ablation_summary.csv`
* `results/core_figures/`

## Latest Result

`RT_A3C_Realtime_MLN` reaches:

* foreground update time: about `0.0044 s/chunk`;
* blocking response cycle: about `0.0059 s/chunk`;
* 10ms realtime-cycle success rate: `100%`;
* mean F1: `0.7034`.

Compared with adaptive batch MLN baselines, foreground update speedup is:

* `2.70x` over rolling L1 MLN;
* `3.59x` over rolling MaxEnt MLN;
* `61.41x` over rolling Boosted MLN;
* `378.02x` over rolling BeamSearch MLN.

`OnlineWeight_MLN` is faster but keeps a fixed rule structure and has much worselog loss, so it is a useful lower-bound online baseline rather than an MLNstructure-adaptation competitor.
