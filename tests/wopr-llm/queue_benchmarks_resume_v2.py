#!/usr/bin/env python3
"""Parent-owned serial CLI benchmark queue; dry-run unless --execute is supplied."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

PLAN_SHA = '611646e745c83cd85c39d8a56ca12975bb81e358709f0c9cc6267fef551ae86a'
RUNNER_SHA = 'b86829c56259b6cb821a0749e812465127c1c8ea49f2b4cff9faf5827e6076b0'
TIMING = re.compile(r'^\[prompt (\d+) tokens in ([0-9.]+)s; (\d+) generated in ([0-9.]+)s \(([0-9.]+) tok/s\)\]$', re.M)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            available = next(int(x.split()[1]) * 1024 for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:'))
            if available < 8 << 30:
                raise RuntimeError('host reserve below 8 GiB during hashing')
            h.update(block)
    return h.hexdigest()


def timing(stderr):
    matches = TIMING.findall(stderr)
    if len(matches) != 1:
        raise ValueError('expected exactly one CLI timing line')
    n, pp, generated, tg, rate = matches[0]
    n, generated = int(n), int(generated)
    pp, tg, rate = float(pp), float(tg), float(rate)
    if not (0 < n <= 480 and 1 <= generated <= 32 and
            all(math.isfinite(x) and x > 0 for x in (pp, tg, rate))):
        raise ValueError('invalid counts or timing')
    # CLI prints rounded seconds and separately rounds the unrounded ratio.
    if not generated / (tg + .005) - .005 <= rate <= generated / max(tg - .005, 1e-9) + .005:
        raise ValueError('printed generation rate inconsistent with rounded duration')
    return {'prompt_tokens': n, 'prompt_seconds_rounded': pp,
            'prompt_tokens_per_second_approx': n / pp,
            'generated_tokens': generated, 'generation_seconds_rounded': tg,
            'generation_tokens_per_second': rate, 'time_to_first_token_seconds': None}


def command(a, case):
    return [sys.executable, str(a.runner), '--out', str(a.out / case['id'] / 'watch'),
            '--lock', str(a.lock), '--input', str(a.runtime), '--input', case['model'],
            '--timeout', '300', '--precheck-mode', 'kfd-compute', '--amd-device', str(a.amd_device),
            '--', str(a.binary), *case['argv'][1:]]


def summarize(watch, case, plan):
    report = json.loads((watch / 'report.json').read_text())
    if not (report.get('pass') is True and report.get('exit_code') == 0 and report.get('group_verified_empty') is True):
        raise ValueError('wrapper did not finish successfully and drain the group')
    if report['executable_sha256'] != plan['cli_sha256'] or case['model_sha256'] not in report['input_sha256'].values() or plan['runtime_sha256'] not in report['input_sha256'].values():
        raise ValueError('executed input identities differ')
    rows = [json.loads(line) for line in (watch / 'samples.jsonl').read_text().splitlines()]
    if not rows or len(rows) != report['samples']:
        raise ValueError('missing memory samples')
    peak = max(x['group_drm_bytes'] for x in rows)
    available = min([report['available_before_bytes']] + [x['available_bytes'] for x in rows])
    if not (0 < peak <= 4026531840 and peak == report['max_drm_allocated_bytes'] and
            available >= 8589934592 and available == report['minimum_available_bytes']):
        raise ValueError('memory bound or summary mismatch')
    answer = (watch / 'stdout.log').read_text()
    if not answer.strip():
        raise ValueError('empty generated output')
    result = timing((watch / 'stderr.log').read_text())
    result.update(answer=answer, sampled_gpu_driver_bytes=peak,
                  sampled_gpu_driver_decimal_GB=peak / 1e9,
                  minimum_host_available_bytes=available,
                  numeric_pass=None, tracked_cleanup=None,
                  artifacts={name: sha(watch / name) for name in ['report.json', 'stdout.log', 'stderr.log', 'samples.jsonl']})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['runner', 'binary', 'runtime', 'plan', 'out', 'lock', 'amd-device']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if any(not getattr(a, name).is_absolute() for name in ['runner', 'binary', 'runtime', 'plan', 'out', 'lock', 'amd_device']):
        p.error('all paths must be absolute')
    if sha(a.plan) != PLAN_SHA:
        raise ValueError('unexpected benchmark plan')
    plan = json.loads(a.plan.read_text())
    jobs = [(case, command(a, case)) for case in plan['cases']]
    if not a.execute:
        print(json.dumps({'jobs': [cmd for _, cmd in jobs], 'environment': plan['environment']}, indent=2))
        return 0
    for path, expected in [(a.runner, RUNNER_SHA), (a.binary, plan['cli_sha256']), (a.runtime, plan['runtime_sha256'])]:
        if sha(path) != expected:
            raise ValueError('unexpected executable/runtime identity: ' + str(path))
    a.out.mkdir(exist_ok=False)
    env = dict(os.environ, **plan['environment'], NS_LIB_FROM_DISK='1')
    env['NEXTROCM_LIB'] = str(a.runtime)
    for key in plan['unset_environment']:
        env.pop(key, None)
    active, interrupted, records, errors = None, [], [], []
    def interrupt(sig, frame):
        interrupted.append(sig)
        if active is not None and active.poll() is None:
            try:
                active.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass
    old = {sig: signal.signal(sig, interrupt) for sig in [signal.SIGTERM, signal.SIGINT]}
    try:
        for case, cmd in jobs:
            if interrupted:
                raise InterruptedError('queue interrupted')
            if sha(Path(case['model'])) != case['model_sha256']:
                raise ValueError('model differs from original validated asset')
            if interrupted:
                raise InterruptedError('queue interrupted before launch')
            job = a.out / case['id']
            job.mkdir()
            record = {'id': case['id'], 'command': cmd, 'status': 'INCOMPLETE'}
            records.append(record)
            with (job / 'wrapper.log').open('x') as log:
                active = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                if interrupted:
                    active.send_signal(signal.SIGTERM)
                # Never kill the reviewed wrapper while it drains its native child.
                code = active.wait()
                active = None
            if interrupted or code:
                raise RuntimeError(f'wrapper stopped: exit={code}, signals={interrupted}')
            record['metrics'] = summarize(job / 'watch', case, plan)
            record['status'] = 'BENCHMARK_COMPLETE'
            print(json.dumps(record), flush=True)
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    finally:
        # Unexpected queue exceptions must still let the wrapper drain.
        if active is not None:
            if active.poll() is None:
                active.send_signal(signal.SIGTERM)
            active.wait()
        for path, expected in [(a.plan, PLAN_SHA), (a.runner, RUNNER_SHA), (a.binary, plan['cli_sha256']), (a.runtime, plan['runtime_sha256'])]:
            try:
                if sha(path) != expected:
                    errors.append('identity changed: ' + str(path))
            except Exception as exc:
                errors.append(str(exc))
        complete = not errors and not interrupted and len(records) == 2 and all(r['status'] == 'BENCHMARK_COMPLETE' for r in records)
        with (a.out / 'queue.json').open('x') as stream:
            json.dump({'pass': complete, 'jobs': records, 'errors': errors, 'received_signals': interrupted,
                       'environment': {k: env[k] for k in plan['environment']},
                       'plan_sha256': PLAN_SHA,
                       'scope': 'One short greedy CLI run per model, not a quality test or steady-state benchmark. No separate TTFT measurement.'}, stream, indent=2)
            stream.write('\n')
        for sig, handler in old.items():
            signal.signal(sig, handler)
    return 0 if complete else 1


if __name__ == '__main__':
    raise SystemExit(main())
