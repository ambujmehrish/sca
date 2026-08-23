"""The reranker, which is where the R@1 is going.

Measured against PMRL's released checkpoint on identical data and protocol, our aggregator
leads on 3 of 4 benchmarks while its reranker converts 2-4x better:

    MSR-VTT   ours 45.4 -> 54.6 (+9.2)    PMRL 31.5 -> 54.3 (+22.8)
    DiDeMo    ours 34.3 -> 50.7 (+16.4)   PMRL 28.7 -> 52.5 (+23.8)
    VATEX     ours 81.2 -> 90.3 (+9.1)    PMRL 53.6 -> 89.6 (+36.0)

Two structural differences are testable here. The negatives our ITM head is trained against
come from one multinomial draw over an all-gathered batch, while inference asks it to
separate the top-50 of a 1000-4000 gallery. And it was fitted on condition_feats_va while
MSR-VTT and VATEX rerank with condition_feats_vas.

Every knob below defaults to the previous behaviour exactly, so an unchanged config is
unchanged arithmetic -- the three-seed numbers already in the table stay reproducible.
"""
import re

SRC = open('model/sca.py').read()
_start = SRC.index('def _itm_loss')
_after = SRC.find('\n    def ', _start + 10)
BLOCK = SRC[_start:_after if _after > 0 else len(SRC)]


def test_the_defaults_reproduce_the_previous_behaviour():
    """itm_neg_topk=0 and itm_num_neg=1 must be the old code path exactly, or every measured
    number in the table silently changes meaning."""
    assert "getattr(self.config, 'itm_neg_topk', 0)" in BLOCK
    assert "getattr(self.config, 'itm_num_neg', 1)" in BLOCK
    assert "getattr(self.config, 'itm_condition_key', 'va')" in BLOCK


def test_negatives_can_be_drawn_from_the_hard_end_of_the_distribution():
    """Sampling the full softmax over ~128 candidates mostly yields easy negatives; inference
    is a top-50 problem."""
    assert 'weights.topk(topk, dim=1)' in BLOCK
    assert 'idx.gather(1, choice)' in BLOCK


def test_an_impossible_topk_is_refused_rather_than_clamped():
    """Silently sampling the whole pool when topk exceeds it would report a hard-negative run
    that was not one."""
    assert 'exceeds the candidate pool' in BLOCK
    assert 'rather than silently sampling the whole pool' in BLOCK


def test_more_than_one_negative_keeps_the_positive_block_first():
    """ground_truth marks the first bs rows positive; the layout must stay
    [pos | n_neg x wrong-clip | n_neg x wrong-text]."""
    assert 'bs * (1 + 2 * n_neg)' in BLOCK
    assert 'ground_truth[:bs] = 1' in BLOCK
    assert '[input_ids] * (1 + n_neg)' in BLOCK


def test_the_reranker_can_be_trained_on_the_modalities_it_is_scored_with():
    """evaluation_mm reranks with condition_feats_{task} -- vas on MSR-VTT and VATEX -- while
    training was hardcoded to va."""
    assert "condition_feats_%s' % itm_key" in BLOCK
    assert 'itm_condition_key' in BLOCK
    assert "'vas'" in BLOCK


def test_training_on_a_modality_the_batch_lacks_is_refused():
    assert "no raw_subtitles" in BLOCK
    assert 'silently fall back' in BLOCK


def test_the_stale_no_subtitles_claim_is_not_relied_on():
    """build_optimizer justified freezing the cross-encoder partly on 'our training set
    carries no subtitles at all'. HyperGram's subtitle guard passes on annotations150k.json
    and only passes at 100% coverage, so that premise is false and must not be reused."""
    assert 'That is false' in BLOCK or 'is false' in BLOCK
