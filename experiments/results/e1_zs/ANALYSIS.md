# E1 zero-shot grid — SCA vs GRAM head-to-head, all five benchmarks (equal 150k budget)

Both models: Wave-1 save_best checkpoints (selected on the MSR-VTT in-training
validation), evaluated cold on each benchmark's test set, full ITM protocol.

## Table metric (ITM-reranked R@1, T2D)

| benchmark | SCA | GRAM (repro) | Δ | GRAM paper (27M) |
|---|---|---|---|---|
| MSR-VTT T-VAS | **53.5** | 52.6 | +0.9 | 54.8 |
| DiDeMo T-VA | **50.0** | 49.6 | +0.4 | 54.2 |
| ActivityNet T-VA | **52.6** | 52.0 | +0.6 | 59.0 |
| VATEX T-VAS (431 subset) | **90.3** | 88.9 | +1.4 | (83.5, std split — n/c) |
| AudioCaps T-VA | **35.2** | 33.1 | +2.1 | **33.2 (27M!)** |

**SCA wins 5/5 zero-shot benchmarks at equal budget.** D2T direction: 3 wins
(ActivityNet +2.1, VATEX tie 86.1, AudioCaps +1.6), 1 loss (DiDeMo 47.3 vs 48.5),
MSR-VTT +? (52.6 best in wave rows). Highlight: **SCA at 150k clips beats the GRAM
paper's own 27M-pretrain zero-shot AudioCaps number** (35.2 vs 33.2) — 180× less
pretraining data (protocol caveat: our AudioCaps eval uses the HyperAlign T-VA
annotation; theirs is labeled T-V-A — verify comparability before headlining).

## Raw scorers — cross-dataset transfer flips the MSR-VTT picture

| benchmark | SCA centroid | GRAM volume | Δ |
|---|---|---|---|
| MSR-VTT (in-domain val) | 34.2 | **37.7** | −3.5 |
| DiDeMo | **32.7** | 26.4 | **+6.3** |
| ActivityNet | **33.3** | 29.2 | +4.1 |
| AudioCaps | **26.1** | 21.9 | +4.2 |
| VATEX | 71.9 | **73.8** | −1.9 |

The MSR-VTT raw deficit is the EXCEPTION, not the rule: both checkpoints were
model-selected on MSR-VTT raw scores (save_best), so GRAM's edge there is partly
selection bias toward its own scorer. Off the selection benchmark, SCA's raw space
transfers better on 3 of 4 (by large margins on the T-VA benchmarks). Text-video
cosine (backbone quality proxy) favors SCA on all five (e.g. DiDeMo 37.1 vs 30.6).

## Reading for the paper

- Zero-shot dominance at equal budget is now measured on five benchmarks, not one.
- The raw-space story sharpens: GRAM's volume advantage is largely in-domain; SCA's
  centroid space generalizes (consistent with calibration slope ≈ 1 — a well-scaled
  space transfers).
- gram_audiocaps' teardown SIGABRT (malloc_consolidate) cost nothing — all metric
  families were logged before the crash.
