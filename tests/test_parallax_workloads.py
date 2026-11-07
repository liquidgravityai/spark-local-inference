"""Protect repeatable traffic histories and honest failure accounting without a GPU."""
from copy import deepcopy
import json
import sys
import threading

import pytest

from scripts.parallax import workloads as w


@pytest.mark.parametrize('family', w.WORKLOADS)
def test_requests_are_repeatable_and_do_not_share_mutable_histories(family):
    first = w.requests(family, 'test-model', copies=2, repeats=2)
    assert first == w.requests(family, 'test-model', copies=2, repeats=2)
    assert len({case['id'] for case in first}) == len(first)
    assert all(case['request']['reasoning_effort'] == 'none' for case in first)
    first[0]['request']['messages'][-1]['content'] = 'changed'
    assert first != w.requests(family, 'test-model', copies=2, repeats=2)
    assert all(case['request']['messages'][-1]['content'] != 'changed' for case in first[1:])


def test_agent_histories_have_matched_tools_and_no_future_result():
    cases = w.requests('agentic', 'test-model')
    assert [case['stage'] for case in cases] == [0, 1, 2]
    for index, case in enumerate(cases):
        calls, results = set(), set()
        for message in case['request']['messages']:
            calls.update(call['id'] for call in message.get('tool_calls', []))
            if message['role'] == 'tool':
                assert message['tool_call_id'] in calls
                results.add(message['tool_call_id'])
        assert results == calls == set(['source', 'failure', 'success'][:index+1])
        assert case['request']['tool_choice'] == 'none'


def test_workload_shift_preserves_explicit_block_order():
    cases = w.requests('shift', 'test-model')
    assert [{c['category'] for c in cases if c['stage'] == stage} for stage in range(4)] == [
        {'code'}, {'prose'}, {'chat'}, {'code'}]


def test_measure_waits_for_each_stage_and_keeps_failed_requests_visible():
    cases = w.requests('chat', 'test-model', copies=2)
    original = deepcopy(cases)
    lock = threading.Lock()
    finished = []

    def send(body):
        stage = next(case['stage'] for case in cases if case['request'] == body)
        with lock:
            assert len(finished) >= stage * 2
            finished.append(stage)
        if stage == 1:
            raise RuntimeError('fixture transport failure')
        return {'usage': {'completion_tokens': 17}, 'choices': [{'finish_reason': 'stop'}]}

    result = w.measure(cases, send, 2)
    assert cases == original
    assert not result['passed']
    assert [g['completion_tokens'] for g in result['groups']] == [34, 0, 34]
    assert result['groups'][1]['tokens_per_second'] is None
    assert all('fixture transport failure' in r['error'] for r in result['groups'][1]['requests'])
    assert [r['id'] for g in result['groups'] for r in g['requests']] == [c['id'] for c in cases]


@pytest.mark.parametrize('response', [
    {}, {'usage': {'completion_tokens': 0}, 'choices': [{}]},
    {'usage': {'completion_tokens': True}, 'choices': [{}]},
    {'usage': {'completion_tokens': 5}, 'choices': []},
])
def test_missing_usage_or_choices_never_produce_a_successful_rate(response):
    result = w.measure(w.requests('code', 'test-model'), lambda _: response, 1)
    assert not result['passed']
    assert result['groups'][0]['tokens_per_second'] is None


def test_idle_guard_requires_all_gauges_and_zero_values():
    idle = '\n'.join(name + ' 0' for name in w.IDLE_GAUGES)
    assert w.idle_metrics(idle)
    assert not w.idle_metrics(idle.replace('flare_pending_requests 0', 'flare_pending_requests 1'))
    assert not w.idle_metrics(idle.split('\n', 1)[1])


def test_export_never_contacts_server_or_overwrites_results(tmp_path, monkeypatch):
    output = tmp_path/'traffic'
    monkeypatch.setattr(sys, 'argv', ['workloads', '--workload', 'agentic', '--model', 'test-model', '--output', str(output)])
    monkeypatch.setattr(w, 'http_json', lambda *args: pytest.fail('export contacted a server'))
    w.main()
    saved = json.loads((output/'requests.json').read_text())
    assert saved['synthetic'] and saved['version'] == 1
    assert saved['cases'] == w.requests('agentic', 'test-model')
    with pytest.raises(FileExistsError):
        w.main()


def test_empty_experiment_is_not_a_pass():
    with pytest.raises(ValueError):
        w.measure([], lambda _: {}, 1)
