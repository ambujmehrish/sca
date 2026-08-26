"""Caption boilerplate that must read identically in every table that needs it.

Imported by the table generators. Nothing here is computed -- these are disclosure sentences
whose only job is to be stated the same way everywhere, so a reviewer comparing two captions
never finds two different accounts of the same protocol fact.

WHY THE VATEX NOTE EXISTS. Our VATEX gallery is 431 test clips, inherited from GRAM's
release: their paper reports "a large part of the dataset is now unavailable online due to
removed or private videos thus we use only a portion of the original dataset composed of
14491 samples", which is their 14,060 train + 431 test (GRAM ICLR 2025, Table 5). We use
their annotation file verbatim -- all 91 VATEX config references in this repo resolve to the
same descs_ret_test_431.json, across the main, transfer, subset, qweight, itm and full
missing-modality grids for every method -- so every row we measure is scored on an identical
gallery and the comparisons between them are exact.

The disclosure is needed because 431 candidates is NOT the ~1,500-clip protocol common in the
VATEX retrieval literature, and R@1 rises as the gallery shrinks. That is why every method
reads near 90 on VATEX in our tables. Without the note, a reader who knows VATEX would take
those absolute numbers as a claim against the published VATEX state of the art, which they
are not: they are a claim against the baselines in the same rows, on the same clips.
"""

# Full form: for tables where VATEX absolute recall is a headline column.
VATEX_NOTE_FULL = (
    "Following the released baselines~\\cite{cicchetti2025gramian}, our VATEX evaluation uses "
    "the 431 test clips that remain downloadable (14{,}060 train / 431 test of the 14{,}491 "
    "recoverable samples), not the $\\sim$1{,}500-clip protocol common in the VATEX "
    "literature. Every row measured here is scored on that identical gallery with the "
    "identical annotation file, so the comparisons within this table are exact; absolute "
    "VATEX recalls are not comparable to numbers reported on the full test set.")

# Short form: for space-constrained captions where VATEX is one column among several.
VATEX_NOTE = (
    "VATEX uses the 431 downloadable test clips of~\\cite{cicchetti2025gramian}, not the "
    "$\\sim$1{,}500-clip protocol; all rows share that gallery, so they compare exactly to "
    "each other but not to VATEX numbers reported on the full test set.")

# Split provenance, for the setup section and any table that states its splits.
SPLITS_NOTE = (
    "Splits follow~\\cite{cicchetti2025gramian} Table~5: MSR-VTT 9{,}000/1{,}000 (the 9k + "
    "1k-A protocol), DiDeMo 8{,}394/1{,}065/1{,}003, ActivityNet 10{,}009/4{,}917, VATEX "
    "14{,}060/431, AudioCaps 700 test.")
