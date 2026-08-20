# HyperGRAM reproduction, v2 (normalised-features reading)

| reading | what the hyperbolic branch consumes | ITM best / final | raw best / final |
|---|---|---|---|
| v1 (method-section reading) | pre-normalisation projections (varying spatial norms) | 35.2 / 35.2 | 18.2 / 18.2 |
| **v2 (experiments-section reading)** | the same L2-normalised features GRAM uses | **51.0 / 37.4** | 36.9 / 19.0 |
| GRAM (our baseline, same env) | — | 52.6 / 51.1 | 37.7 / 32.2 |
| SCA-T1 (ours) | — | 54.9 / 54.4 | 36.6 / 36.1 |
| HyperGRAM as PUBLISHED (their env) | — | 56.6 / — | — |

## Verdict

v2 confirms the diagnosis: the ambiguity in their paper is load-bearing. Feeding the
Lorentzian Gram unnormalised projections (v1) collapses training; using the normalised
features their experiments section describes ("only changes the inner product
computation") recovers a plausible model — **51.0 ITM at its best checkpoint, with a
raw score (36.9) essentially level with GRAM's (37.7)**. So the geometry works; our v1
was the wrong reading.

Two honest caveats, both reported:
1. v2 does not reach their published 56.6 (our env runs ~2 R@1 colder, which accounts
   for part but not all of the difference), and it lands below our GRAM baseline (52.6).
2. v2 is **unstable**: 51.0 → 37.4 best→final (−13.6), the largest decay of any arm in
   the campaign (next worst: PMRL-masked −6.7). The hybrid volume's learnable α mixing
   two differently-scaled volume terms under one temperature is the likely mechanism.

**Reporting policy**: HyperGRAM's PUBLISHED numbers remain what our comparison tables
cite (Table 1, group (b)). The v2 reproduction is reported in the appendix as a
best-effort attempt with the ambiguity documented — no claim is made that their method
underperforms, since no code is released and a third reading may exist.
