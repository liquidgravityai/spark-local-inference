"""Authored HTTP traffic for Parallax experiments; no serving or predictor code.

Histories and tool results are fixed fixtures. Generated responses are recorded,
never executed or inserted into later requests. This is not an agent quality benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen

WORKLOADS = ('code', 'agentic', 'prose', 'chat', 'long_context', 'shift')
IDLE_GAUGES = ('flare_active_requests', 'flare_pending_requests', 'flare_retiring_request_slots',
               'flare_http_inflight_requests', 'flare_frontend_active_workers')


def definitions(workload, records=256):
    if workload not in WORKLOADS or not 1 <= records <= 4096:
        raise ValueError('unknown workload or records outside 1..4096')
    cases = []

    def add(name, category, messages, stage=0, **fields):
        cases.append(dict(id=name, category=category, stage=stage, request=dict(messages=deepcopy(messages), **fields)))

    def user(text):
        return {'role': 'user', 'content': text}

    if workload == 'code':
        for name, prompt in [
            ('lru', 'Implement a bounded Python LRU cache with get, put, eviction, and pytest examples. Explain its complexity.'),
            ('parser', 'Write a Rust parser for key=value configuration lines, including whitespace, comments, duplicate keys and useful errors. Include tests.'),
            ('sql', 'Design PostgreSQL tables for orders and line items. Write a query for monthly revenue including months with no orders. Explain indexes and edge cases.'),
        ]:
            add(name, workload, [user(prompt)])
    elif workload == 'agentic':
        tools = [{'type': 'function', 'function': {'name': 'read_fixture',
            'description': 'Read a recorded source file or test result.',
            'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']}}}]
        history = [user('Fix validators.py so validate_workers rejects bool while accepting integers from 1 through 64. Preserve the existing range errors.')]

        def tool_result(identifier, path, text):
            history.extend([
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier, 'type': 'function',
                    'function': {'name': 'read_fixture', 'arguments': json.dumps({'path': path})}}]},
                {'role': 'tool', 'tool_call_id': identifier, 'content': text},
            ])

        tool_result('source', 'validators.py', 'def validate_workers(value):\n    if not isinstance(value, int):\n        raise TypeError("workers must be an integer")\n    if not 1 <= value <= 64:\n        raise ValueError("workers out of range")\n    return value\n')
        add('inspect', workload, history + [user('Explain the defect and propose a minimal patch. Do not call tools.')], 0, tools=tools, tool_choice='none')
        history.append({'role': 'assistant', 'content': 'The range check remains unchanged. Add an explicit bool rejection before accepting int values.'})
        tool_result('failure', 'pytest.txt', 'FAILED test_reject_bool: validate_workers(True) returned True. Boundary tests for 1 and 64 passed.')
        add('patch', workload, history + [user('Return the complete corrected function and parameterized tests. Do not call tools.')], 1, tools=tools, tool_choice='none')
        history.append({'role': 'assistant', 'content': 'The patch uses isinstance(value, bool) or not isinstance(value, int) for type rejection, then the original range check.'})
        tool_result('success', 'pytest.txt', '8 passed: True, False, string and float rejection; 0 and 65 range rejection; 1 and 64 acceptance. This is an authored fixture, not an executed test.')
        add('review', workload, history + [user('Review the patch and test coverage. State what the recorded result does and does not establish. Do not call tools.')], 2, tools=tools, tool_choice='none')
    elif workload == 'prose':
        for name, prompt in [
            ('exposition', 'Write a clear essay explaining how a public library can preserve local history while respecting privacy. Include concrete fictional examples and practical tradeoffs.'),
            ('fiction', 'Write a scene set in a lighthouse during a power failure. Two siblings disagree about selling it. Use dialogue, sensory detail and an unresolved ending.'),
            ('editing', 'Rewrite this fictional notice into a warm, concise announcement, then explain the edits: "Pursuant to the decision of the committee, utilization of the communal garden will be suspended for the purpose of repairs from Monday to Thursday. We apologize for any inconvenience occasioned thereby."'),
        ]:
            add(name, workload, [user(prompt)])
    elif workload == 'chat':
        history = [user('Help plan a fictional neighborhood science evening for 20 people with a 200-credit budget. Suggest activities that work indoors.')]
        add('plan', workload, history, 0)
        history += [{'role': 'assistant', 'content': 'Use three stations: paper bridges, shadow experiments and water filtration demonstrations. Reserve 50 credits for materials at each station and 50 for refreshments.'},
                    user('Ten attendees are children under eight, and we cannot use water. Revise the activities while preserving the budget.')]
        add('revise', workload, history, 1)
        history += [{'role': 'assistant', 'content': 'Replace filtration with paper spinners. Assign adult helpers, avoid small detachable parts and retain the four 50-credit allocations.'},
                    user('Now only two volunteers are available. Give a realistic schedule, explain the compromises, and remember the restrictions.')]
        add('constraints', workload, history, 2)
    elif workload == 'long_context':
        catalog = '\n'.join(f'Service svc-{i:04d}: owner team-{i % 17:02d}; region zone-{i % 5}; replicas {2+i%7}; timeout_ms {700+i%13*50}; retry_limit {i%4}. Revision {1000+i}.' for i in range(records))
        wanted = sorted({0, records//2, records-1})
        add('retrieve', workload, [user(f'Use only this fictional service catalog:\n{catalog}\nReturn the owner, replicas and timeout for services {wanted}. Cite each exact service ID and explain how you checked the records.')])
    else:
        for stage, family in enumerate(('code', 'prose', 'chat', 'code')):
            for case in definitions(family, records):
                case['id'] = f'{stage}-{case["id"]}'
                case['stage'] = stage
                cases.append(case)
    return cases


def requests(workload, model, *, copies=1, repeats=1, max_tokens=512, records=256):
    if not model or not 1 <= copies <= 32 or not 1 <= repeats <= 10 or not 16 <= max_tokens <= 32768:
        raise ValueError('model required; copies 1..32, repeats 1..10, max_tokens 16..32768')
    result = []
    base = definitions(workload, records)
    stages = 1 + max(case['stage'] for case in base)
    for repeat in range(repeats):
        for case in base:
            for copy in range(copies):
                entry = deepcopy(case)
                entry['id'] = f'{workload}-{case["id"]}-r{repeat}-c{copy}'
                entry['stage'] += repeat * stages
                body = entry['request']
                body.update(model=model, max_tokens=max_tokens, seed=7, temperature=1.0, top_p=.95, top_k=-1, reasoning_effort='none', stream=False)
                body['messages'].insert(0, {'role': 'system', 'content': f'Synthetic traffic session {workload}-r{repeat}-c{copy}. Answer the request directly. Tool results and prior assistant messages are authored fixtures.'})
                result.append(entry)
    return result


def http_json(url, body=None):
    request = Request(url, data=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=300) as response:
        return json.load(response)


def idle_metrics(text):
    values = dict(line.split() for line in text.splitlines()
                  if line and not line.startswith('#') and line.split()[0] in IDLE_GAUGES)
    return set(values) == set(IDLE_GAUGES) and all(float(v) == 0 for v in values.values())


def wait_idle(url):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        with urlopen(url + '/metrics', timeout=5) as response:
            metrics = response.read().decode()
        if idle_metrics(metrics):
            return metrics
        time.sleep(.1)
    raise TimeoutError('server did not drain within 30 seconds; server left running')


def measure(cases, send, concurrency):
    if not cases or not 1 <= concurrency <= 32:
        raise ValueError('nonempty cases and concurrency 1..32 required')

    def one(case):
        started = time.monotonic_ns()
        try:
            response = send(case['request'])
            usage = response['usage']
            if type(usage['completion_tokens']) is not int or usage['completion_tokens'] <= 0 or not response['choices']:
                raise ValueError('missing completion usage or choices')
            return dict(id=case['id'], latency_ns=time.monotonic_ns()-started, response=response, error=None)
        except Exception as error:
            return dict(id=case['id'], latency_ns=time.monotonic_ns()-started, error=f'{type(error).__name__}: {error}')

    groups = []
    for stage in sorted({c['stage'] for c in cases}):
        selected = [c for c in cases if c['stage'] == stage]
        started = time.monotonic_ns()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows = list(pool.map(one, selected))
        elapsed = time.monotonic_ns()-started
        passed = all(row['error'] is None for row in rows)
        tokens = sum(row['response']['usage']['completion_tokens'] for row in rows if row['error'] is None)
        groups.append(dict(stage=stage, passed=passed, elapsed_ns=elapsed, completion_tokens=tokens,
            tokens_per_second=tokens*1e9/elapsed if passed else None,
            median_request_ms=statistics.median(row['latency_ns'] for row in rows)/1e6, requests=rows))
    return dict(passed=all(g['passed'] for g in groups), groups=groups,
        scope='Authored synthetic traffic, fixed histories, client concurrency bound. Wall time includes prefill and HTTP; not TTFT or ITL. Natural EOS and length stops retained; no quality score or fixed-output-length guarantee.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--workload', choices=WORKLOADS, default='code')
    parser.add_argument('--model')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--base-url', help='Optional existing Flare server; omitted means export requests only.')
    parser.add_argument('--copies', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--records', type=int, default=256)
    args = parser.parse_args()
    if args.list:
        print('\n'.join(WORKLOADS)); return
    if args.output is None or not args.model or not 1 <= args.concurrency <= 32:
        parser.error('--output, --model and concurrency 1..32 are required')
    cases = requests(args.workload, args.model, copies=args.copies, repeats=args.repeats, max_tokens=args.max_tokens, records=args.records)
    args.output.mkdir(parents=True)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    save('requests.json', dict(version=1, synthetic=True, workload=args.workload, concurrency=args.concurrency, cases=cases))
    if not args.base_url:
        print(f'Exported {len(cases)} requests; no server contacted.'); return
    url = args.base_url.rstrip('/')
    save('model.json', http_json(url+'/v1/models'))
    (args.output/'metrics-before.txt').write_text(wait_idle(url))
    save('parallax-before.json', http_json(url+'/v1/parallax/predictions'))
    result = measure(cases, lambda body: http_json(url+'/v1/chat/completions', body), args.concurrency)
    save('results.json', result)
    (args.output/'metrics-after.txt').write_text(wait_idle(url))
    save('parallax-after.json', http_json(url+'/v1/parallax/predictions'))
    print(json.dumps({'passed': result['passed'], 'groups': len(result['groups']), 'output': str(args.output)}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
