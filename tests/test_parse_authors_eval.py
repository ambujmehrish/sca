"""Reading numbers out of the released repositories' own eval logs.

Both forks print through the same VAST-family logger, so one parser serves the PMRL
released-checkpoint row and the HyperGRAM authors'-code row. The property that matters is
which of the two retrieval blocks becomes the row: ret_itm (after reranking, what every other
row of our table reports) versus ret_itc (the aggregator before it). On VATEX those differ by
36 points, so picking the wrong one silently would not look like an error.
"""
import importlib.util
import json
import subprocess

SAMPLE = """08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--vatex_ret_vatex_ret_cosine_TV========
08/23 - INFO - __main__ -   {'forward_r1': 81.2, 'forward_recall': '81.2/97.9/99.3'}
08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--vatex_ret_vatex_ret_cosine_TA========
08/23 - INFO - __main__ -   {'forward_r1': 20.4, 'forward_recall': '20.4/45.9/58.0'}
08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--vatex_ret_vatex_ret_ret_itc_tvas========
08/23 - INFO - __main__ -   {'video_r1': 53.6, 'video_recall': '53.6/85.2/92.6'}
08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--vatex_ret_vatex_ret_ret_itm_tvas========
08/23 - INFO - __main__ -   {'video_r1': 89.6, 'video_recall': '89.6/97.9/98.8', 'txt_r1': 87.0}
"""


def _mod():
    spec = importlib.util.spec_from_file_location('m', 'scripts/parse_authors_eval.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_both_retrieval_blocks_are_extracted_and_distinguished(tmp_path):
    log = tmp_path / 'run.log'
    log.write_text(SAMPLE)
    got = _mod().parse(str(log))
    assert got[('vatex', 'ret_itm_tvas')]['video_r1'] == 89.6
    assert got[('vatex', 'ret_itc_tvas')]['video_r1'] == 53.6
    assert got[('vatex', 'cosine_TV')]['forward_r1'] == 81.2


def test_the_reported_number_is_the_reranked_one(tmp_path):
    """ret_itm, not ret_itc. Every other row of our table is the reranked figure."""
    log = tmp_path / 'run.log'
    log.write_text(SAMPLE)
    r = subprocess.run(['python3', 'scripts/parse_authors_eval.py', str(log)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    reported = [l for l in r.stdout.splitlines() if 'REPORTED' in l][0]
    assert '89.6' in reported and '53.6' not in reported


def test_text_to_video_is_the_reported_direction(tmp_path):
    log = tmp_path / 'run.log'
    log.write_text(SAMPLE)
    r = subprocess.run(['python3', 'scripts/parse_authors_eval.py', str(log)],
                       capture_output=True, text=True)
    assert 'not the reported direction' in r.stdout, \
        'V->T is also printed and must be marked, or it gets copied into the table'


def test_a_log_without_evaluation_says_so_rather_than_reporting_nothing(tmp_path):
    log = tmp_path / 'run.log'
    log.write_text('the job died before evaluation\n')
    r = subprocess.run(['python3', 'scripts/parse_authors_eval.py', str(log)],
                       capture_output=True, text=True)
    assert 'NO EVALUATION BLOCKS' in r.stderr
    assert not r.stdout.strip(), 'an empty result must not print a table of nothing'


def test_the_aggregation_gain_is_measured_against_the_best_single_pathway(tmp_path):
    log = tmp_path / 'run.log'
    log.write_text(SAMPLE)
    r = subprocess.run(['python3', 'scripts/parse_authors_eval.py', str(log)],
                       capture_output=True, text=True)
    assert '-27.6' in r.stdout, 'aggregator 53.6 against cosine_TV 81.2'


def test_json_output_is_machine_readable_for_the_table_builder(tmp_path):
    log = tmp_path / 'run.log'
    log.write_text(SAMPLE)
    r = subprocess.run(['python3', 'scripts/parse_authors_eval.py', '--json', str(log)],
                       capture_output=True, text=True)
    data = json.loads(r.stdout)
    assert data['vatex']['ret_itm_tvas']['video_r1'] == 89.6


HG_SAMPLE = """08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--msrvtt_ret_ret_area_forward========
08/23 - INFO - __main__ -   {'volume_T2D_r1': 39.1, 'volume_T2D_recall': '39.1/63.8/74.2'}
08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--msrvtt_ret_ret_area_backard========
08/23 - INFO - __main__ -   {'forward_r1': 33.2, 'forward_recall': '33.2/62.1/71.4'}
08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--msrvtt_ret_ret_itm_area========
08/23 - INFO - __main__ -   {'volume_ITM_T2D_r1': 54.0, 'volume_ITM_T2D_recall': '54.0/74.7/82.0', 'volume_ITM_D2T_r1': 51.9}
08/23 - INFO - __main__ -   ==== evaluation--ret%tvas--msrvtt_ret_cosine_TV========
08/23 - INFO - __main__ -   {'forward_r1': 42.5, 'forward_recall': '42.5/70.1/80.5'}
"""


def test_hypergrams_key_dialect_is_read_not_reported_as_nan(tmp_path):
    """HyperGram prints volume_ITM_T2D_r1 where PMRL prints video_r1, and calls the
    aggregator ret_area_forward rather than ret_itc_*. Reading only one dialect made a
    COMPLETED HyperGram cell print 'T->V R@1 nan' -- a found result reported as missing,
    which reads like a failed run and nearly got treated as one."""
    log = tmp_path / 'run.log'
    log.write_text(HG_SAMPLE)
    r = subprocess.run(['python3', 'scripts/parse_authors_eval.py', str(log)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    reported = [l for l in r.stdout.splitlines() if 'REPORTED' in l][0]
    assert '54.0' in reported and 'nan' not in reported
    agg = [l for l in r.stdout.splitlines() if 'aggregator' in l][0]
    assert '39.1' in agg, 'ret_area_forward is their name for the pre-rerank aggregator'
    assert '-3.4' in r.stdout, 'aggregation gain against cosine_TV 42.5'


def test_the_backward_direction_is_not_mistaken_for_the_aggregator(tmp_path):
    """ret_area_backard (their spelling) is the reverse direction at 33.2; treating it as the
    aggregator would understate the (negative) aggregation gain by a point."""
    log = tmp_path / 'run.log'
    log.write_text(HG_SAMPLE)
    r = subprocess.run(['python3', 'scripts/parse_authors_eval.py', str(log)],
                       capture_output=True, text=True)
    assert '33.2' not in [l for l in r.stdout.splitlines() if 'aggregator' in l][0]
