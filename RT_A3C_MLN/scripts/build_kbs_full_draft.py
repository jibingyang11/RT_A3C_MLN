# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import shutil
import textwrap
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "paper_artifacts"
TABLE_DIR = ARTIFACTS / "tables"
FIG_DIR = ARTIFACTS / "figures"
OUT_ROOT = ROOT / "kbs_full_draft"
TEMPLATE_ZIP = Path("C:/Users/Administrator/Desktop/temp/我的论文-MLN/els-cas-templates.zip")
FALLBACK_TEMPLATE = ROOT / "kbs_partitioned_draft" / "els-cas-templates"
ZIP_PATH = ARTIFACTS / "RT_A3C_MLN_KBS_full_draft.zip"


def latex_escape(value):
    text = "" if pd.isna(value) else str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def read_table(name, columns=None):
    path = TABLE_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if columns:
        keep = [col for col in columns if col in df.columns]
        df = df[keep]
    return df


def table_tex(df, caption, label, size="\\scriptsize"):
    if df.empty:
        return ""
    colspec = "l" * len(df.columns)
    header = " & ".join(latex_escape(col) for col in df.columns) + r" \\"
    rows = []
    for _, row in df.iterrows():
        rows.append(" & ".join(latex_escape(row[col]) for col in df.columns) + r" \\")
    body = "\n".join(rows)
    return rf"""
\begin{{table*}}[!t]
\centering
\caption{{{caption}}}
\label{{{label}}}
{size}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{{colspec}}}
\toprule
{header}
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table*}}
"""


def copy_template():
    if OUT_ROOT.exists():
        resolved = OUT_ROOT.resolve()
        if resolved.parent != ROOT.resolve() or resolved.name != "kbs_full_draft":
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if TEMPLATE_ZIP.exists():
        with zipfile.ZipFile(TEMPLATE_ZIP) as zf:
            zf.extractall(OUT_ROOT)
        candidates = [p for p in OUT_ROOT.iterdir() if p.is_dir()]
        if len(candidates) == 1 and candidates[0].name != "els-cas-templates":
            candidates[0].rename(OUT_ROOT / "els-cas-templates")
    elif ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(OUT_ROOT)
    else:
        shutil.copytree(FALLBACK_TEMPLATE, OUT_ROOT / "els-cas-templates")
    template_dir = OUT_ROOT / "els-cas-templates"
    (template_dir / "figs").mkdir(parents=True, exist_ok=True)
    return template_dir


def copy_figures(template_dir):
    for fig in FIG_DIR.glob("*.pdf"):
        shutil.copy2(fig, template_dir / "figs" / fig.name)


def build_bib(template_dir):
    src = ROOT / "kbs_partitioned_draft" / "els-cas-templates" / "rt_a3c_mln_refs.bib"
    if src.exists():
        text = src.read_text(encoding="utf-8")
    elif ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as zf:
            try:
                text = zf.read("els-cas-templates/rt_a3c_mln_refs.bib").decode("utf-8")
            except KeyError:
                text = ""
    else:
        text = ""
    extra = r"""

@article{hollmann2025tabpfn,
  title={Accurate Predictions on Small Data with a Tabular Foundation Model},
  author={Hollmann, Noah and Müller, Samuel and Eggensperger, Katharina and Hutter, Frank},
  journal={Nature},
  volume={637},
  pages={319--326},
  year={2025},
  doi={10.1038/s41586-024-08328-6}
}

@inproceedings{gorishniy2025tabm,
  title={TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling},
  author={Gorishniy, Yury and Rubachev, Ivan and Babenko, Artem},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
"""
    if "hollmann2025tabpfn" not in text:
        text += extra
    (template_dir / "rt_a3c_mln_refs.bib").write_text(text, encoding="utf-8")


def build_tex(template_dir):
    main_table = read_table(
        "Table1_Formal_All_Methods",
        [
            "Method",
            "Group",
            "Year/type",
            "Core F1",
            "Core cycle ms",
            "10ms %",
            "Strong F1",
            "Strong Log-loss",
            "Recovery",
            "Struct %",
            "Rule +/-",
            "Emerging rules",
        ],
    )
    ablation_table = read_table("Table2_Formal_Ablation")
    actor_table = read_table(
        "Table3_Actor_Count",
        [
            "Actors",
            "Rule scope",
            "Scope/actor",
            "Total eps",
            "Actor eps max",
            "Search sec",
            "Base T90",
            "Search speedup",
            "Base T90 speedup",
            "Val F1",
            "Signal recall",
        ],
    )
    update_table = read_table("Table4_Update_Frequency")

    tex = rf"""
\documentclass[a4paper,fleqn]{{cas-sc}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{adjustbox}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage{{algorithm}}
\usepackage{{algorithmic}}
\usepackage{{hyperref}}
\usepackage{{float}}
\begin{{document}}
\shorttitle{{RT-A3C-MLN}}
\shortauthors{{Anonymous}}

\title [mode = title]{{RT-A3C-MLN: Real-Time Markov Logic Network Rule Learning with Asynchronous Actor-Critic Search}}
\author[1]{{Anonymous Author}}
\address[1]{{Anonymous Institution}}

\begin{{abstract}}
Markov logic networks (MLNs) provide an expressive framework for combining first-order logic and probabilistic reasoning, but conventional structure learning depends heavily on global candidate-rule search and batch weight optimization. This makes MLNs difficult to deploy in data streams where the active logical rules and data distribution may change rapidly. This paper proposes RT-A3C-MLN, a real-time MLN rule learning framework based on asynchronous advantage actor-critic learning. In the proposed model, multiple Actors independently explore different concept boxes of the candidate rule space, perform rule addition, deletion, replacement, and weight adjustment, and send asynchronous updates to a global Critic and shared probabilistic rule memory. Foreground inference uses the latest accepted MLN structure and is not blocked by background rule refresh. Experiments on four real tabular data streams and a strong emerging-rule drift stress test compare RT-A3C-MLN with 21 baselines, including classic MLN structure learners, fixed-rule online MLNs, modern data-stream learners, recent rule-adaptive methods, and 2024--2026 tabular foundation/deep models. The results show that RT-A3C-MLN is most valuable when fast adaptation and explicit rule-structure updating are required: it keeps low foreground response time while activating new rules under drift, whereas fixed-rule or purely predictive baselines either lack structure adaptation or incur larger update costs.
\end{{abstract}}

\begin{{keywords}}
Markov logic network \sep structure learning \sep actor-critic learning \sep data stream \sep real-time reasoning \sep neuro-symbolic learning
\end{{keywords}}

\maketitle

\section{{Introduction}}
Markov logic networks (MLNs) attach real-valued weights to first-order formulas and define a probabilistic distribution over possible worlds \citep{{richardson2006markov}}. They are attractive for knowledge-intensive prediction because their rules can express relational dependencies while their weights capture uncertainty. However, MLN structure learning is expensive: a learner must search a large formula space and estimate corresponding weights \citep{{kok2005learning,lowd2007efficient}}. This dependence on global rule search is especially problematic in data streams, where a useful model must respond to new chunks before full retraining is possible.

Real-time applications such as financial monitoring, autonomous perception, multi-sensor fusion, and industrial IoT require models that can both infer quickly and revise their knowledge when the generating rules change. A static MLN may give stable predictions but cannot discover rules that become active after a drift point. A fixed-rule online MLN can adjust weights quickly, but its structure remains constrained by the initial rule set. Streaming learners such as adaptive trees and random patches react to drift \citep{{bifet2009adaptive,gomes2017adaptive,gomes2019streaming,montiel2021river}}, yet they do not provide explicit weighted logical rule updates. This gap motivates a real-time MLN learner with both low blocking latency and adaptive rule-structure search.

RT-A3C-MLN formulates MLN structure adjustment as an asynchronous advantage actor-critic process \citep{{mnih2016asynchronous}}. A state is the current MLN rule set and weight configuration; actions add, delete, replace, or reweight rules; and rewards are based on validation log-likelihood with a complexity penalty. Multiple Actors explore different rule subspaces and asynchronously update a shared Critic and global rule memory. This design is aligned with concept-lattice knowledge organization: different concept boxes provide different local rule contexts, and their discoveries are merged into a single probabilistic knowledge network.

The contributions are threefold. First, we propose a streaming MLN structure-learning formulation where Actor actions directly correspond to MLN rule and weight operations. Second, we design an asynchronous multi-Actor learning mechanism that decouples foreground inference from background rule refresh. Third, we conduct a complete experimental study with 21 baselines, four real datasets, a strong rule-drift stress test, component ablations, Actor-count analysis, and update-frequency analysis.

\section{{Related Work}}
\subsection{{MLN Structure Learning}}
MLNs were introduced by Richardson and Domingos \citep{{richardson2006markov}}. Early structure learners searched clauses and accepted formulas that improved likelihood or pseudo-likelihood \citep{{kok2005learning}}, while later methods improved weight learning and scalability \citep{{lowd2007efficient,jha2016scalable}}. Boosted and beam-search MLN variants remain useful references because they represent the classic batch or rolling-retrain view of MLN structure learning. Recent incremental MLN work, such as affordance learning with MLNs \citep{{potter2024incremental}}, also confirms the value of revisable probabilistic rules.

\subsection{{Streaming and Online Learning}}
Data-stream learning has developed many drift-aware models, including adaptive windowing \citep{{bifet2009adaptive}}, adaptive random forests \citep{{gomes2017adaptive}}, streaming random patches \citep{{gomes2019streaming}}, and the River framework \citep{{montiel2021river}}. These methods are strong online predictors, but their learned structures are not MLN-style weighted logical rules. RT-A3C-MLN targets a different objective: fast probabilistic reasoning with explicit rule additions and deletions.

\subsection{{Recent Rule-Adaptive and Neuro-Symbolic Learning}}
Recent work in neuro-symbolic and rule learning includes differentiable first-order rule learning \citep{{gao2024dforl}}, rule networks \citep{{wei2024rns}}, neuro-symbolic rule lists \citep{{xu2024neurules}}, neural probabilistic logic learning \citep{{sun2026npll}}, and broader surveys of neuro-symbolic architectures \citep{{bhuyan2024neurosymbolic,amador2024profiling,feldstein2024mapping,colelough2025systematic}}. These studies motivate the 2024--2026 rule-adaptive baselines in this paper. We implement their comparable core mechanisms as style/proxy baselines under a unified MLN rule-feature protocol, because the goal is to compare adaptation behavior and latency under the same stream setting.

\subsection{{Tabular Foundation and Deep Models}}
Recent tabular foundation or deep models, including TabPFN v2 \citep{{hollmann2025tabpfn}} and TabM \citep{{gorishniy2025tabm}}, provide strong predictive references. They are included to avoid comparing only against older MLN learners. However, these models do not directly maintain an interpretable MLN rule structure, so they serve as modern predictive baselines rather than rule-adaptation baselines.

\section{{Method}}
\subsection{{MLN Rule State, Actions, and Reward}}
Let $\mathcal{{R}}=\{{r_1,\ldots,r_m\}}$ be candidate MLN rules and $\mathbf{{z}}\in\{{0,1\}}^m$ be the active rule mask. Given rule activations $\phi_i(x)$ and weights $w_i$, prediction is written as
\begin{{equation}}
P(y=1\mid x,\mathcal{{M}})=\sigma\left(b+\sum_i z_i w_i \phi_i(x)\right).
\end{{equation}}
The learning state consists of the current active rule set, recent weights, local validation reward, and global best reward. The action set contains four operations: adding a candidate rule, deleting an active rule, replacing a weak rule, and adjusting a rule weight. The reward is
\begin{{equation}}
R(\mathbf{{z}},\mathbf{{w}})=
\ell_{{base}}-\ell_{{val}}(\mathbf{{z}},\mathbf{{w}})
-\lambda\frac{{\|\mathbf{{z}}\|_0}}{{B}},
\end{{equation}}
where $\ell_{{val}}$ is validation log-loss and $B$ is the active-rule budget.

\subsection{{Asynchronous Advantage Actor-Critic Learning}}
Each Actor uses local rule evidence to propose actions and receives an advantage signal
\begin{{equation}}
A_t=G_t - V(s_t),
\end{{equation}}
where $G_t$ is the $n$-step return and $V(s_t)$ is the Critic value estimate. Positive advantage increases the probability of the corresponding rule operation. Multiple Actors send asynchronous gradients and local best structures to the global network. This prevents the learner from depending on a single greedy path and reduces the risk of local optima in large formula spaces.

\subsection{{Concept-Box Search and Global Merge}}
The candidate rule space is divided into concept boxes derived from the MLN rule representation. Each Actor is assigned one rule subspace and searches inside it. When there are $K$ Actors and $m$ candidate rules, each Actor handles roughly $m/K$ candidates. The global Critic merges local best masks, trims the union to the active-rule budget, and accepts a structure refresh when it improves validation likelihood within a tolerance. This is the intended RT-A3C-MLN mechanism rather than a separate version of the method.

\subsection{{Realtime Inference Path}}
Foreground inference uses the latest accepted global MLN and performs only probability evaluation plus lightweight prior/weight adjustment. Background structure refresh is accounted for separately. This separation lets the model respond to a stream chunk immediately while still adapting its rule structure when labels arrive.

\begin{{algorithm}}[!t]
\caption{{RT-A3C-MLN streaming rule learning}}
\begin{{algorithmic}}[1]
\STATE Initialize candidate MLN rules, weights, Actors, Critic, and global rule memory.
\FOR{{each stream chunk}}
  \STATE Predict using the latest accepted MLN rule structure.
  \STATE Update foreground prior and online rule weights using arriving labels.
  \FOR{{each Actor in parallel}}
    \STATE Explore add/delete/replace/reweight actions in its concept box.
    \STATE Compute reward from validation log-likelihood and complexity penalty.
    \STATE Send advantage update and local best rules to the global network.
  \ENDFOR
  \STATE Asynchronously merge accepted local structures into global MLN memory.
\ENDFOR
\end{{algorithmic}}
\end{{algorithm}}

\section{{Experiments}}
\label{{sec:experiments}}
\subsection{{Experimental Design}}
The experiments are designed to test the central claim: RT-A3C-MLN can form a probabilistic knowledge network that responds in real time while adapting its rules under changing streams. We evaluate four real tabular datasets converted into MLN rule-feature streams: mushroom, adult, bank, and spambase. We also use a strong emerging-rule drift stress test in which the post-drift label rule depends on predicates that are rare or misleading in the initial window. All methods are evaluated under the same chunked stream protocol.

The comparison contains 21 baselines in addition to RT-A3C-MLN: recent rule-adaptive style/proxy baselines, classic rolling MLN learners, fixed-rule online MLNs, stream learners, and tabular foundation/deep references. The metrics are F1, log-loss, blocking cycle time, 10 ms success rate, structure refresh rate, rule additions/deletions, and number of activated emerging rules.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=\textwidth]{{figs/Fig1_Method_Framework.pdf}}
\caption{{RT-A3C-MLN framework. The foreground path performs realtime probabilistic inference, while background Actors explore rule addition, deletion, replacement, and weight adjustment in concept boxes and asynchronously update the global MLN.}}
\label{{fig:framework}}
\end{{figure*}}

{table_tex(main_table, "Main formal comparison with all baselines.", "tab:main")}

\subsection{{Main Comparison}}
Table~\ref{{tab:main}} and Fig.~\ref{{fig:dashboard}} summarize the formal comparison. The main point is not that RT-A3C-MLN always maximizes static F1 on every ordinary stream. Rather, the method is designed for the joint requirement of realtime response and rule-structure adaptation. Fixed-rule online baselines can be fast but do not activate new MLN rules. Rolling MLN baselines can update structure but have much larger foreground retraining costs. Modern foundation/deep baselines can be strong predictors, but they do not provide MLN rule additions, deletions, and emerging-rule activation evidence. RT-A3C-MLN is therefore evaluated by a combined profile: low blocking cycle, competitive post-drift prediction, and explicit structure refresh.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=\textwidth]{{figs/Fig2_All_Method_Dashboard.pdf}}
\caption{{High-density dashboard for all methods. The four panels show realtime blocking cycle, prediction quality, strong-drift probability quality, and rule-structure adaptation evidence.}}
\label{{fig:dashboard}}
\end{{figure*}}

\subsection{{Strong Emerging-Rule Drift}}
The strong drift experiment is the most direct test of the paper's claim. Before the drift point, emerging predicates have low support and can even be misleading. After the drift point, those predicates become the dominant decision rules. Fig.~\ref{{fig:strong}} shows chunk-level F1, blocking cycle, and active emerging-rule counts. A method that only updates weights over a fixed initial structure cannot reliably recover when the required predicates were absent from the initial rule memory. RT-A3C-MLN can refresh its rule structure and activate post-drift predicates, providing direct evidence that MLN rules can be self-adaptively adjusted under different stream inputs.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=\textwidth]{{figs/Fig3_Strong_Drift_Heatmap.pdf}}
\caption{{Strong emerging-rule drift stress test. The heatmap jointly reports recovery trajectory, realtime cost, and whether each method activates newly relevant rules.}}
\label{{fig:strong}}
\end{{figure*}}

{table_tex(ablation_table, "Ablation study of RT-A3C-MLN components.", "tab:ablation")}

\subsection{{Ablation Study}}
Table~\ref{{tab:ablation}} and Fig.~\ref{{fig:ablation}} isolate the contribution of each component. The full model combines multi-Actor structure search, prior-shift correction, and background structure refresh. The single-Actor variant tests whether parallel exploration matters. The no-prior-shift variant removes fast local distribution adjustment. The no-structure-refresh variant freezes the rule set after initial learning, and the online-only variant removes MLN structure search entirely. Degradation in the no-structure or online-only settings supports the claim that fast weight adjustment alone is insufficient when the active logical rules change.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=\textwidth]{{figs/Fig4_Ablation_Rule_Evidence.pdf}}
\caption{{Ablation evidence. The figure shows post-drift predictive quality, realtime cycle, and emerging-rule activation for the full model and reduced variants.}}
\label{{fig:ablation}}
\end{{figure*}}

{table_tex(actor_table, "Effect of Actor count under fixed global rule scope and fixed total exploration budget.", "tab:actor")}

\subsection{{Parallel Actor Count}}
Table~\ref{{tab:actor}} and Fig.~\ref{{fig:actor}} evaluate Actor counts $[1,2,3,4,5,6,8,9,10,11,12]$. The rule scope and total episode budget are fixed, so increasing Actor count does not increase total exploration. Instead, it reduces the local search scope per Actor. The Base T90 metric measures how quickly each setting reaches 90\% of the one-Actor final reward. This experiment directly tests the technical feasibility claim that multiple Actors can explore multiple rule paths in parallel and accelerate convergence to useful structures.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=\textwidth]{{figs/Fig5_Actor_Count_Study.pdf}}
\caption{{Actor-count study. More Actors reduce local rule-scope size and can improve time-to-useful-rule discovery without increasing the total episode budget.}}
\label{{fig:actor}}
\end{{figure*}}

{table_tex(update_table, "Effect of asynchronous structure-refresh frequency.", "tab:update_frequency")}

\subsection{{Update-Frequency Analysis}}
Table~\ref{{tab:update_frequency}} and Fig.~\ref{{fig:update_frequency}} vary the structure refresh interval from 1 to 10 chunks. A shorter interval refreshes rules more aggressively and may activate drift rules sooner, while a longer interval reduces background update cost but reacts more slowly. The foreground cycle remains the key realtime metric because background refresh is asynchronous. This experiment completes the planned evaluation of different update frequencies and clarifies the latency--adaptation tradeoff.

\begin{{figure*}}[!t]
\centering
\includegraphics[width=\textwidth]{{figs/Fig6_Update_Frequency_Study.pdf}}
\caption{{Update-frequency study. The panels compare latency, post-drift quality, and rule-adaptation evidence across refresh intervals.}}
\label{{fig:update_frequency}}
\end{{figure*}}

\section{{Discussion}}
The experiments support three observations. First, RT-A3C-MLN should be judged by realtime adaptive reasoning rather than by static accuracy alone. Second, explicit structure refresh is essential under strong rule drift because fixed-rule online updates cannot add missing post-drift predicates. Third, multi-Actor exploration is useful when the rule scope is large and the total exploration budget is fixed, which matches the intended concept-lattice setting.

The experiments also clarify limitations. The recent rule-adaptive baselines are implemented as style/proxy models under a unified rule-feature protocol, so they are not claimed to be official reproductions. In addition, asynchronous background update time is reported separately from foreground blocking latency; this matches the realtime deployment assumption but should be considered when provisioning compute resources.

\section{{Conclusion}}
This paper proposed RT-A3C-MLN, a real-time Markov logic network rule-learning framework based on asynchronous advantage actor-critic search. The method allows multiple Actors to explore rule additions, deletions, replacements, and weight adjustments in different concept boxes, while a global Critic asynchronously merges useful structures. Experiments with 21 baselines, four real datasets, a strong rule-drift stress test, ablations, Actor-count analysis, and update-frequency analysis show that RT-A3C-MLN provides a practical route for building probabilistic knowledge networks that can respond quickly to data streams and adapt their rule structures when the underlying logical relations change.

\bibliographystyle{{cas-model2-names}}
\bibliography{{rt_a3c_mln_refs}}
\end{{document}}
"""
    tex = textwrap.dedent(tex).strip() + "\n"
    (template_dir / "rt_a3c_mln_kbs_full_draft.tex").write_text(tex, encoding="utf-8")


def make_zip(template_dir):
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in template_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(template_dir.parent))
    return ZIP_PATH


def main():
    template_dir = copy_template()
    copy_figures(template_dir)
    build_bib(template_dir)
    build_tex(template_dir)
    zip_path = make_zip(template_dir)
    print(f"[done] tex: {template_dir / 'rt_a3c_mln_kbs_full_draft.tex'}")
    print(f"[done] zip: {zip_path}")


if __name__ == "__main__":
    main()
