# Wave 8 — the finetuning learning rate, given symmetrically to both methods

| method | regime | ft lr | T2V | V2T |
|---|---|---|---|---|
| GRAM | full-FT | 2e-5 | **61.2** | **60.4** |
| GRAM | full-FT | 1e-4 | 58.8 (−2.4) | 58.2 (−2.2) |
| SCA | full-FT | 2e-5 | **59.4** | **60.9** |
| SCA | full-FT | 1e-4 | 56.2 (−3.2) | 56.9 (−4.0) |

**Result: 1e-4 is worse for both.** The pretraining lr finding (where 1e-4 gained SCA
+1.4) does not transfer to finetuning — unsurprising in hindsight, since finetuning runs
4 epochs on 9k clips rather than 1 epoch on 150k, so the same step size is far more
aggressive per unit of data. SCA degrades slightly more than GRAM (−3.2 vs −2.4).

**Consequence for Table 3**: each method is reported at its own best configuration —
GRAM 61.2, SCA 59.4 — and the −1.8 gap stands. This is now the *tuned* comparison, not
an artifact of an inherited hyperparameter: both methods received both learning rates and
both preferred 2e-5.

The finetune gap is therefore a genuine finding, and its cause is identified: adapter
capacity was the binding constraint (LoRA 56.6 → full-FT 59.4, +2.8), the initialisation
was not (v2a from a better zero-shot start gained nothing), and the learning rate is not
(this wave). What remains is that GRAM's volume objective extracts more from in-domain
finetuning on the benchmark it is strongest on — reported as such.
