#!/usr/bin/env python3
"""Offline affected-engine planner and pinned per-device regression comparator."""
import argparse
import fnmatch
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

def validate(data):
    if data.get('version') != 1 or not isinstance(data.get('engines'), dict):
        raise ValueError('invalid gate schema version/engines')
    policy = data['policy']
    if not 0 <= policy['speed_regression_fraction'] < 1 or policy['default_runs'] < 2:
        raise ValueError('invalid regression policy')
    for name, engine in data['engines'].items():
        if len(name.split('/')) != 2 or not engine.get('paths'):
            raise ValueError('engine requires kind/name and source paths')
        if engine.get('runs', policy['default_runs']) < 2:
            raise ValueError('at least two repetitions required')
        for card, baseline in engine.get('cards', {}).items():
            d = baseline['device']
            for key in ('gfx', 'wave_size', 'total_bytes', 'units', 'driver', 'compiler'):
                if key not in d:
                    raise ValueError('missing device provenance: ' + key)
            if not card or not baseline['commit'] or not baseline['runtime_sha256'] or not baseline['exact']:
                raise ValueError('missing baseline provenance/output')
            if not baseline.get('numeric_gate_pass', False):
                raise ValueError('baseline requires a separate numeric gate')
            if not isinstance(baseline.get('metrics'), dict) or not baseline['metrics']:
                raise ValueError('baseline requires nonempty performance metrics')
            for metric in baseline['metrics'].values():
                if metric['direction'] not in ('higher', 'lower') or not positive(metric['value']):
                    raise ValueError('invalid baseline metric')

def positive(v):
    return isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and v > 0

def affected(data, paths):
    names, unmapped = set(), []
    for path in paths:
        matches = {n for n, e in data['engines'].items() if any(path.startswith(p) for p in e['paths'])}
        for prefix, patterns in data['shared'].items():
            if path.startswith(prefix):
                matches.update(n for n in data['engines'] if any(fnmatch.fnmatchcase(n, pat) for pat in patterns))
        names.update(matches)
        if not matches:
            unmapped.append(path)
    return {'engines': sorted(names), 'unmapped_paths': unmapped,
            'status': 'PLAN_READY' if names and not unmapped else 'UNVALIDATED',
            'scope': 'Variants conservatively gate every recorded device of their engine; narrowing requires verified device-match evidence.'}

def compare(data, selected, results):
    if selected['status'] != 'PLAN_READY' or not selected['engines']:
        raise ValueError('unvalidated or empty affected-engine coverage')
    rows = []
    for name in selected['engines']:
        e = data['engines'][name]
        if not e.get('cards'):
            raise ValueError('engine has no hardware baselines: ' + name)
        for card, b in e['cards'].items():
            r = results[name][card]
            if r['device'] != b['device'] or not r.get('numeric_gate_pass'):
                raise ValueError('device mismatch or missing independent numeric gate')
            if not r.get('runtime_sha256') or not r.get('executable_sha256') or not r.get('commit'):
                raise ValueError('missing candidate provenance')
            if selected.get('head') and r['commit'] != selected['head']:
                raise ValueError('candidate receipt is not planned head')
            runs = r['runs']
            if len(runs) != e.get('runs', data['policy']['default_runs']):
                raise ValueError('wrong repeat count')
            for run in runs:
                if run['exact'] != b['exact'] or run['exit_code'] != 0 or not run['group_empty']:
                    raise ValueError('output/exit/cleanup gate failed')
                if not run.get('memory_gate_pass') or not run.get('idle_gate_pass'):
                    raise ValueError('missing memory/idle gate')
            metrics = {}
            for key, metric in b['metrics'].items():
                values = [run['metrics'][key] for run in runs]
                if not all(positive(v) for v in values):
                    raise ValueError('nonpositive/nonfinite timing')
                higher = metric['direction'] == 'higher'
                best = max(values) if higher else min(values)
                tolerance = data['policy']['speed_regression_fraction']
                passed = best >= metric['value'] * (1-tolerance) if higher else best <= metric['value'] * (1+tolerance)
                metrics[key] = {'baseline': metric['value'], 'candidate_best': best, 'pass': passed}
            rows.append({'engine': name, 'card': card, 'metrics': metrics, 'pass': all(m['pass'] for m in metrics.values())})
    return {'status': 'PASS' if all(r['pass'] for r in rows) else 'FAIL', 'rows': rows,
            'scope': 'Exact outputs plus declared per-device best-of-repeats performance; independent numerical, memory, idle and process-cleanup gates are required inputs.'}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baselines', type=Path, default=ROOT/'tooling/gate-baselines.json')
    sub = ap.add_subparsers(dest='action', required=True)
    p = sub.add_parser('plan'); p.add_argument('base'); p.add_argument('head')
    p = sub.add_parser('compare'); p.add_argument('plan', type=Path); p.add_argument('results', type=Path)
    p = sub.add_parser('run', help='execute one explicit pinned job under the reviewed AMD watchdog')
    p.add_argument('job', type=Path); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lock', type=Path, required=True); p.add_argument('--amd-device', type=Path, required=True)
    a = ap.parse_args(); raw = a.baselines.read_bytes(); data = json.loads(raw); validate(data)
    if a.action == 'run':
        job_raw = a.job.read_bytes(); job = json.loads(job_raw)
        command = job['argv']
        if not command or not all(isinstance(v, str) for v in command) or not Path(command[0]).is_absolute():
            raise ValueError('job requires direct absolute argv, not a shell command')
        exe_path = str(Path(command[0]).resolve(strict=True))
        paths = {}
        for path, expected in job['inputs_sha256'].items():
            resolved = str(Path(path).resolve(strict=True))
            if resolved in paths and paths[resolved] != expected:
                raise ValueError('conflicting duplicate input identity')
            paths[resolved] = expected
        if exe_path in paths and paths[exe_path] != job['executable_sha256']:
            raise ValueError('input identity conflicts with executable identity')
        paths[exe_path] = job['executable_sha256']
        runtime = str(Path(job['runtime']).resolve(strict=True))
        if runtime not in paths:
            raise ValueError('runtime must be included in pinned inputs')
        with Path(command[0]).open('rb') as f:
            if f.read(4) != b'\x7fELF': raise ValueError('direct native ELF required')
        for path, expected in paths.items():
            h = hashlib.sha256()
            with Path(path).open('rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
            if h.hexdigest() != expected: raise ValueError('job input identity mismatch: ' + path)
        device = a.amd_device.resolve(strict=True)
        if not device.is_relative_to(Path('/sys')) or (device/'vendor').read_text().strip().lower() != '0x1002':
            raise ValueError('AMD sysfs device required')
        total = int((device/'mem_info_vram_total').read_text().strip())
        cap = job['device_cap_bytes']
        if not isinstance(cap, int) or not 0 < cap <= total:
            raise ValueError('job cap exceeds reported device total')
        spec = importlib.util.spec_from_file_location('gpu_watchdog', ROOT/'tooling/gpu/run_job.py')
        watchdog = importlib.util.module_from_spec(spec); spec.loader.exec_module(watchdog)
        previous = os.environ.get('NEXTROCM_LIB')
        try:
            os.environ['NEXTROCM_LIB'] = runtime
            code = watchdog.run(a.out, command, job['timeout_seconds'], job.get('host_reserve_bytes', 8*1024**3),
                                a.lock, list(paths), 'kfd-compute', device, cap, paths)
        finally:
            if previous is None: os.environ.pop('NEXTROCM_LIB', None)
            else: os.environ['NEXTROCM_LIB'] = previous
        (a.out/'gate-job.json').write_text(json.dumps({'job_sha256':hashlib.sha256(job_raw).hexdigest(),
            'reported_device_total_bytes':total, 'device':str(device), 'numeric_validation_pass':False,
            'scope':'Watchdog execution only; model-specific numeric/output/metric extraction and gate comparison remain mandatory.'},indent=2)+'\n')
        raise SystemExit(code)
    if a.action == 'plan':
        base = subprocess.check_output(['git','rev-parse','--verify',a.base+'^{commit}'],cwd=ROOT,text=True).strip()
        head = subprocess.check_output(['git','rev-parse','--verify',a.head+'^{commit}'],cwd=ROOT,text=True).strip()
        paths = subprocess.check_output(['git','diff','--no-renames','--name-only','-z',base,head],cwd=ROOT).decode().split('\0')
        report = affected(data, [p for p in paths if p]); report.update(base=base, head=head)
    else:
        selected = json.loads(a.plan.read_text())
        if selected.get('baselines_sha256') != hashlib.sha256(raw).hexdigest():
            raise ValueError('baseline changed since plan')
        report = compare(data, selected, json.loads(a.results.read_text()))
    report['baselines_sha256'] = hashlib.sha256(raw).hexdigest()
    print(json.dumps(report, indent=2))
    if report['status'] in ('FAIL','UNVALIDATED'): raise SystemExit(1)

if __name__ == '__main__':
    main()
