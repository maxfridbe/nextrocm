#!/usr/bin/env python3
"""Offline-only audit of a WOPR typed all-family event capture; no GPU or subprocesses."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def expected_records(tensors):
    expected = []
    schedule = [(l, 0) for l in range(6)]
    schedule += [(l, p) for p in (1, 2, 3) for l in range(6, 18)]
    schedule += [(l, 0) for l in range(18, 24)]

    def matrix(name, family):
        t = tensors[name]
        shape = t['shape_outer_first']
        require(len(shape) == 2, 'not a matrix: ' + name)
        expected.append(((t['type_id'], *shape, 0), family))

    for l, p in schedule:
        if p > 1:
            for role in ('attn', 'k', 'v'):
                for part in ('down', 'gain', 'forget'):
                    matrix(f'blk.{l}.loop_{role}_{part}.weight', 'controllers')
        for role in ('attn_q', 'attn_k', 'attn_v', 'attn_rel'):
            matrix(f'blk.{l}.{role}.weight', 'attention_projections')
        if (l + 1) % 6 == 0:
            for role in ('attn_k_conv_proj', 'attn_v_conv_proj'):
                matrix(f'blk.{l}.{role}.weight', 'attention_projections')
        for role in ('attn_sink_proj', 'attn_gate', 'attn_output'):
            matrix(f'blk.{l}.{role}.weight', 'attention_projections')
        if p > 1:
            for part in ('down', 'gain', 'forget'):
                matrix(f'blk.{l}.loop_mlp_{part}.weight', 'controllers')
        matrix(f'blk.{l}.ffn_gate_inp.weight', 'router_projection')
        for _ in range(8):
            expected.append(((40 if l < 20 else 13, 2304, 2304, int(l < 20)), 'expert_gate_up'))
            expected.append(((40 if l < 20 else 8, 2304, 1152, int(l < 20)), 'expert_down'))
    matrix('output.weight', 'output_projection')
    require(len(expected) == 1457, 'wrong independently derived record count')
    return expected, schedule


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'baseline', 'inventory', 'window', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--runtime-sha', required=True)
    p.add_argument('--exe-sha', required=True)
    a = p.parse_args()
    pins = {str(x): sha(x) for x in (a.capture/'report.json', a.capture/'all-profile.json', a.baseline/'report.json', a.inventory, a.window/'report.json', a.window/'samples.jsonl')}
    require(pins[str(a.baseline/'report.json')] == '8657467641a8652bbce0a9f8b11738a8336ace6d00a89f066cd33b00443d0785', 'baseline authority changed')
    require(pins[str(a.inventory)] == '9654f242cec0265d797ae217975dfd400e22654c7113f554db2ead563aee6413', 'GGUF inventory authority changed')
    b = json.loads((a.baseline/'report.json').read_text())
    c = json.loads((a.capture/'report.json').read_text())
    q = json.loads((a.capture/'all-profile.json').read_text())
    w = json.loads((a.window/'report.json').read_text())
    ts = json.loads(a.inventory.read_text())
    require(len(ts) == len({t['name'] for t in ts}) == 734, 'tensor inventory')
    expected, schedule = expected_records({t['name']: t for t in ts})
    paths = {}
    for token in range(2):
        for ordinal, (layer, _) in enumerate(schedule):
            paths[f'token-{token:03}/visit-{ordinal:02}-physical-{layer:02}.f32'] = 2304
        paths[f'token-{token:03}/logits.f32'] = 200058
    require(b['status'] == c['status'] == 'DIAGNOSTIC_COMPLETE', 'capture incomplete')
    require(b['context'] == 2 and b['input_ids'] == [0, 1] and c['input_ids'] == [0] and not b['generated_ids'] and not c['generated_ids'], 'workload mismatch')
    require(b['config_sha256'] == c['config_sha256'] and b['model_bytes'] == c['model_bytes'], 'model/config mismatch')
    require(b['runtime_sha256'] == '25a45d71d718cd92c2e2fec67dc4ffe08ab7f393bb3c8338e9e6becc0a877598', 'baseline runtime')
    require(c['runtime_sha256'] == a.runtime_sha and c['executable_sha256'] == a.exe_sha and c['identities_unchanged'], 'capture identities')
    require(set(b['files']) == set(paths), 'baseline schedule mismatch')
    token0 = {name for name in paths if name.startswith('token-000/')}
    require(set(c['files']) == token0, 'profile schedule mismatch')
    for name, count in paths.items():
        v = b['files'][name]
        raw = (a.baseline/name).read_bytes()
        require(len(raw) == count*4 and v['shape'] == [count] and hashlib.sha256(raw).hexdigest() == v['sha256'], 'baseline payload: '+name)
        require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)), 'nonfinite baseline')
        if name in token0:
            actual = (a.capture/name).read_bytes()
            require(actual == raw and hashlib.sha256(actual).hexdigest() == c['files'][name]['sha256'], 'profile bytes: '+name)
    require(q == c['all_profile'] and q['status'] == 'PROFILE_AND49_BYTE_PARITY_PASS', 'profile reports differ')
    require(q['record_count'] == q['expected_count'] == len(q['records']) == 3581, 'event count')
    groups = {}; families = {}; total = 0.
    counts=[1457,1,434,144,384,48,1,192,48,48,48,144,48,96,16,88,384]
    names=['matvec','decode','RMS','clip','SiTU','gate','softcap','shortconv','attention_logits','attention_probabilities','attention_weighted','depthmix','router','loop_control','scale','add','axpy']
    observed=[0]*17; mi=0
    for r in q['records']:
        f=r['family'];require(1<=f<=17 and r['reserved']==0,'family/reserved')
        observed[f-1]+=1
        if f==1:
            require(tuple(r[k] for k in ('type','rows','cols','aux'))==expected[mi][0],'matrix order');mi+=1
        elif f==2:require(tuple(r[k] for k in ('type','rows','cols','aux'))==(8,1,2304,0),'decode')
        else:
            require(r['type']==0 and r['rows']>0 and r['cols']>0,'nonquant metadata')
            if f==8:require(r['aux'] in (0,1) and (r['rows'],r['cols']) in ((8,1024),(1,2304)),'conv')
            elif f==12:require(1<=r['rows']<=5 and r['cols']==2304 and 1<=r['aux']<=4,'depth')
            else:require(r['aux']==0,'unused aux')
            if f in (9,10,11):require((r['rows'],r['cols'])==(24,1),'token0 attention')
        family=names[f-1];key=tuple(r[k] for k in ('family','type','rows','cols','aux'))
        ms=r['gpu_ms'];require(math.isfinite(ms) and ms>=0,'duration');total+=ms
        for g in (groups.setdefault(key,{'count':0,'gpu_ms':0.}),families.setdefault(family,{'count':0,'gpu_ms':0.})):
            g['count']+=1;g['gpu_ms']+=ms
    require(observed==counts==q['expected_family_counts']==q['observed_family_counts'] and mi==1457,'family/subsequence coverage')
    require(total == q['summed_gpu_ms'] and len(q['groups']) == len(groups), 'event total/groups')
    seen = set()
    for g in q['groups']:
        key = tuple(g[k] for k in ('family', 'type', 'rows', 'cols', 'aux'))
        require(key not in seen and groups[key] == {k:g[k] for k in ('count', 'gpu_ms')}, 'group aggregation')
        seen.add(key)
    require(c['allocation_after_drop'][0] == 0 and c['allocation_after_drop'][1] < 4026531840, 'allocation/drop gate')
    require(w['status'] == 'PASS' and w['exit_code'] == 0 and w['group_verified_empty'] and not w['cleanup_signals'], 'window failure')
    require(a.runtime_sha in w['input_sha256'].values() and w['executable_sha256'] == a.exe_sha, 'watchdog identities')
    samples = [json.loads(x) for x in (a.window/'samples.jsonl').read_text().splitlines()]
    require(len(samples) == w['samples'] and max(s['group_drm_bytes'] for s in samples) == w['max_drm_allocated_bytes'] < 4026531840, 'raw memory samples')
    wall = c['token_seconds']; require(len(wall) == 1 and math.isfinite(wall[0]) and wall[0] > 0, 'token wall')
    for g in families.values():g['percent_of_all_compute_events'] = 100*g['gpu_ms']/total if total else 0.
    for path, pin in pins.items():require(sha(Path(path)) == pin, 'audit input changed')
    result = {'status':'INDEPENDENT_WOPR_PROFILE_PASS', 'baseline_files':98, 'profile_full_byte_equal_outputs':49, 'ordered_records':3581, 'compute_event_ms':total, 'families':families, 'diagnostic_token_ms':wall[0]*1000, 'event_to_instrumented_wall_ratio':total/(wall[0]*1000), 'tracked_after_drop':c['allocation_after_drop'], 'sampled_drm_bytes':w['max_drm_allocated_bytes'], 'runtime_sha256':a.runtime_sha, 'executable_sha256':a.exe_sha, 'pins':pins, 'scope':'All explicit compute kernel event intervals, not utilization or throughput. Residual includes transfers, host reading/enqueue/synchronization, per-visit readbacks/hashes and observer overhead; overlapping CPU/GPU time is not additive. Runtime identity bound to watchdog before/after, not local binary rehash.'}
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
