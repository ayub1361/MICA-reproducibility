# MICA: reproducibility material

Code and result tables for the paper

> A. Zeinvand-Lorestani, M. Mohammadrezaei, and M. Kolahkaj, "Preventing Irrecoverability in Sequential Group Recommendation: A Minimally Disruptive Approach to Contract Attainability," submitted to *IEEE Transactions on Knowledge and Data Engineering*.

MICA (Minimal Intervention for Contract Attainability) is a filter that runs on top of an arbitrary base group recommender. Before each daily commitment it solves a day-layered mixed-integer program (P) to check whether every member's entitlement is still attainable over the remaining horizon. It keeps the base recommender's choice when the contract stays attainable and otherwise commits the highest-ranked feasible alternative.

## Contents

| Folder | What it holds |
|---|---|
| `src/vista_gt/` | The method, the 29 compared policies and the day-by-day simulator. `vista_gt` is the internal package name; in raw simulator output the MICA policy is labelled `vista`. |
| `scripts/experiments/` | Shell scripts that build the processed data and run the confirmatory study and the two data replications. |
| `scripts/analysis/` | Recomputation of the Table III statistics and of the intervention-ratio intervals from the per-group outcomes. |
| `scripts/figures/` | Plotting code for Figs. 1, 3, 4, 5 and S1. |
| `results/main/` | One CSV per results table of the paper (Tables II to V), plus the per-group outcomes of the confirmatory run. |
| `results/supplementary/` | One CSV per results table of the Supplementary Material (Tables S3 to S9). |
| `results/figure_data/` | The numbers plotted in each figure. |
| `tests/` | Unit tests, including an exact brute-force check of the MILP on small instances. |

## Where each table and figure comes from

| Paper item | File |
|---|---|
| Table II, decision instances and protocol | `results/main/table2_instances_and_protocol.csv` |
| Table III, confirmatory results | `results/main/table3_confirmatory_results.csv`; per-group outcomes of all 29 policies on the 219 held-out groups in `results/main/table3_per_group_outcomes.csv` |
| Table IV, replications | `results/main/table4_replications.csv` |
| Table V, cost of the guarantee | `results/main/table5_cost_two_dimensions.csv` |
| Table S3, study design | `results/supplementary/tableS3_study_design.csv` |
| Table S4, component ablation | `results/supplementary/tableS4_component_ablation.csv` |
| Table S5, full confirmatory field | `results/supplementary/tableS5_full_confirmatory_field.csv` |
| Table S6, dependence on the margin σ | `results/supplementary/tableS6_margin_dependence.csv` |
| Table S7, paired comparison | `results/supplementary/tableS7_paired_comparison.csv` |
| Table S8, four design dimensions | `results/supplementary/tableS8_design_dimensions.csv` |
| Table S9, history-dependent utility | `results/supplementary/tableS9_history_dependent_utility.csv` |
| Fig. 1 | `results/figure_data/fig1_trajectory.csv`, `scripts/figures/fig1.py` |
| Fig. 3 | `results/figure_data/fig3_day_of_loss.csv`, `scripts/figures/fig3.py` |
| Fig. 4 | `results/figure_data/fig4_policy_field.csv`, `scripts/figures/fig4.py` |
| Fig. 5 | `results/figure_data/fig5_cost_and_scope.csv`, `scripts/figures/fig5.py` |
| Fig. S1 | `results/figure_data/figS1_contract_level.csv`, `scripts/figures/figS1.py` |

The table CSVs reproduce the printed tables; the table notes in the paper explain the symbols (for example, † in Table III marks a McNemar test with no discordant pairs). In `table3_per_group_outcomes.csv`, member identifiers are replaced by salted hashes, which keeps the membership overlap between groups (used by the cluster bootstrap) without redistributing Yelp user identifiers.

## Installation

Python 3.13 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick checks without the datasets

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -t .
python scripts/analysis/table3_statistics.py
python scripts/analysis/mica_ratio_ci.py --csv results/main/table3_per_group_outcomes.csv
python scripts/figures/fig4.py
```

`table3_statistics.py` recomputes the irrecoverability rates, Wilson intervals, intervention rates, base-score loss and exact McNemar p-values of Table III from the per-group file.

## Rerunning the experiments

The datasets are not redistributed. Download them from their providers and accept their terms:

- Yelp Open Dataset, January 2022 release: https://business.yelp.com/data/resources/open-dataset/ . Place the JSON files in `data/raw/`.
- Google Local Reviews (Pennsylvania and Florida review and metadata files), for the second-corpus replication only. Place them in `data/raw/dataset2/`.

Then, from the repository root:

```bash
bash scripts/experiments/prepare_data.sh               # corpus, split, utility model, catalogues, groups, calibration
bash scripts/experiments/run_confirmatory.sh           # Table III, Table S5, Fig. 4
bash scripts/experiments/run_replication_2021_window.sh  # Table IV, 2021 window
bash scripts/experiments/run_replication_google_local.sh # Table IV, Google Local
```

Each simulator run writes one row per group, decision day and policy, together with a provenance file that records the configuration and input hashes. The other studies in the paper use the same entry point, `python -m vista_gt.s5_simulate`, with the settings stated in the paper and the Supplementary Material (see `--help`). The program (P) is solved with the HiGHS solver bundled with SciPy; decision times depend on the machine. Where several anchors are exactly tied, the solver may return a different but equivalent choice, which can move individual cells of the ablation table slightly.

## License

The code is released under the MIT License (see `LICENSE`). The datasets remain subject to their providers' terms.
