# Final paper tables

All numbers measured in one evaluation environment unless a row is explicitly marked as
reported by its authors. Regenerate measured rows from the raw logs with
`scripts/extract_results.py`; figures with `scripts/make_figures.py`.

| file | content | status |
|---|---|---|
| `table1_main.tex` | Main zero-shot MSR-VTT T-VAS, both directions. Groups (a) external models (reported) / (b)–(d) measured here. No `Data` column; provenance by marker (`*` released weights, `†` trained by us, `‡` reimplementation) | final |
| `table2_zeroshot_all.tex` | Zero-shot across 5 benchmarks, both directions | final |
| `table3_finetune.tex` | Fine-tuned MSR-VTT + depth ($k{=}5$), both ft learning rates | final |
| `table4_missingness.tex` | Missing modalities: (a) ITM metric, (b) raw space, 3 seeds | final |
| `table5_missing_transfer.tex` | Missing modalities on the 3 non-selection benchmarks (raw, 3 seeds) | final |
| `table5_calibration.tex` | Semantic calibration (E6) + cardinality fairness (E5) | final |
| `table6_ablations.tex` | Ablations: (a) loss components, (b) adaptation regime, (c) knobs | **(a) pending runs** |
| `table7_env_audit.tex` | Evaluation-environment audit (appendix) | final |
| `table8_reported_and_variants.tex` | Appendix: author-reported GRAM/PMRL/HyperGRAM numbers + baselines equipped with LoRA/masked training (moved out of Table 1) | final |
| `main_table_single.tex` | Alternative single-block Table 1 (superseded by table1) | alt |
| `PAPER_TABLES.md` | Human-readable master with every number | reference |

Figures: `figures/fig_e4_missingness.pdf` (raw curves), `fig_e4_transfer.pdf`
(off-benchmark curves, 3 panels), `fig_e6_calibration.pdf` (slope + $R^2$ panels).

## LaTeX preamble requirements

```latex
\usepackage{booktabs}
\usepackage{pifont}
\newcommand{\cmark}{\ding{51}}
\newcommand{\xmark}{\ding{55}}
```

## Two configurations, reported consistently

Every table reports both SCA operating points rather than picking the better one per
benchmark:

- **SCA** (base learning rate) — best calibration ($R^2$ $-$0.67), best cross-benchmark
  transfer, best V2T.
- **SCA$_{lr}$** (learning rate 1e-4) — best in-domain MSR-VTT (54.9 $\pm$0.15, +2.5 over
  GRAM at 52.4 $\pm$0.21), best robustness curve, but weaker transfer and V2T.

The trade-off is a single hyperparameter, is stated in the captions, and is visible in
Tables 2, 4 and 5.

## Claims the tables support, and their limits

| claim | evidence | limit |
|---|---|---|
| +2.5 R@1 over GRAM at identical budget/env/recipe | Tab. 1, 3 seeds each | T2V direction; V2T is parity |
| The gain is not the adapter regime | Tab. 1: SCA at full-FT 53.4 vs best full-FT baseline 52.5; Tab. 8(c): GRAM+LoRA 53.3, PMRL+LoRA 53.9 vs SCA 54.9 | baselines' LoRA lr not separately tuned; SCA's full-FT arm is at base lr |
| Leads at every missing rate on the reported metric | Tab. 4(a): 54.9/42.2/34.0 vs 52.6/40.5/32.1 | 2 rates only (each is a full reranked pass) |
| 2$\times$ gentler raw-space degradation | Tab. 4(b): drop 7.1 vs 9.5--9.9 | GRAM-LoRA converges to a tie by 90\% |
| Only calibrated score in the family | Tab. 5: slope 0.978 vs 1.8--2.8 and 0.5 | $R^2$ negative for all methods (under-dispersion) |
| Masked training is free, and buys the robustness | Tab. 1 (53.5 vs 53.4) + Tab. 4(b) (7.1 vs 15.4 drop) | -- |
| Adapter regime is a quality choice | Tab. 6: full-FT worse at both lrs, unstable at 1e-4 | -- |
| Arity-invariance pays: $k{=}5$ adds +1.6 with no code change | Tab. 3 last row | single benchmark |
| Mean imputation is degenerate under volume scoring | Prop. 1 + Tab. 4 note | proof + measurement |

Honest negatives that stay in the paper: GRAM finetunes better in-domain (Tab. 3);
GRAM-LoRA has the strongest raw space at low missing rates (Tab. 4b); V2T is parity, not
a win (Tab. 1); HyperGRAM's published full-modality number is the highest in its
environment (Tab. 1, group (b)).
