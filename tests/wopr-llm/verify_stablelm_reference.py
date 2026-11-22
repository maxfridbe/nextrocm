#!/usr/bin/env python3
"""Read-only verification of the retained corrected Zephyr CPU oracle, no inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

MODEL_SHA = '74b2613b6e89d904a2ea38d56d233e4d2ca2fe663844b2e7aa90e769d359061b'
MODEL_BYTES = 1708595648
IDS = [29,93,4537,49651,187,1276,310,253,5347,273,6181,32,0,187,29,93,515,5567,49651,187]

def require(ok, why):
    if not ok:
        raise ValueError(why)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def unique(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, 'duplicate JSON key')
        d[k] = v
    return d

def verify(root, model, manifest_sha):
    root, model = root.resolve(strict=True), model.resolve(strict=True)
    manifest = root / 'manifest.json'
    require(manifest.stat().st_size <= 1024*1024, 'manifest exceeds bound')
    require(sha(manifest) == manifest_sha, 'external manifest pin mismatch')
    m = json.loads(manifest.read_text(), object_pairs_hook=unique)
    require(m['sha256'] == MODEL_SHA and m['bytes'] == MODEL_BYTES, 'reference model identity')
    require(model.stat().st_size == MODEL_BYTES and sha(model) == MODEL_SHA, 'actual model identity')
    require(m['hf_revision'] == 'fe1fd5e36ccb8cb2cb1dc225ab6d2962692b9837', 'HF revision')
    require(m['source_sha256'] == '01e7a09c37aff4ad41d9fa6e3b7ee85ab2584ccfaad5f0e3110647c4352c7ec0', 'original HF source hash')
    require(m['input_ids'] == IDS, 'requires corrected20-ID fixture; old23-ID artifact rejected')
    names = {'inp_embd', 'result_norm', 'result_output'}
    names |= {f'{n}-{i}' for i in (0,15,31) for n in ('attn_norm','ffn_inp','ffn_norm','l_out','Kcur','Vcur')}
    expected = {(mode, n) for mode in ('fp32','fp16kv') for n in names}
    require(len(m['taps']) == 42, 'requires all42 original taps')
    seen, pins, total = set(), {}, 0
    for row in m['taps']:
        key = (row['mode'], row['name'])
        require(key in expected and key not in seen, 'unexpected or duplicate tap')
        seen.add(key)
        filename = f'{key[0]}-{key[1]}.f32'
        require(row['file'] == filename, 'tap filename/path contract')
        path = root / filename
        require(not path.is_symlink() and path.resolve(strict=True).parent == root, 'tap escapes directory')
        shape = [1,50304] if key[1] == 'result_output' else [1,20,2560]
        require(row['shape'] == shape, 'exact tap shape')
        elements = math.prod(shape)
        require(path.stat().st_size == elements*4, 'exact tap byte size')
        require(sha(path) == row['sha256'], 'tap SHA mismatch')
        count = 0
        best_value, best_index = -math.inf, None
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                for value, in struct.iter_unpack('<f', chunk):
                    require(math.isfinite(value), 'nonfinite reference')
                    if value > best_value:
                        best_value, best_index = value, count
                    count += 1
        require(count == elements, 'tap element count')
        if key[1] == 'result_output':
            require(best_index == 510, 'actual original full-logit argmax')
        total += count
        pins[filename] = row['sha256']
    require(seen == expected, 'missing tap')
    for mode in ('fp32','fp16kv'):
        require(m['runs'][mode]['expected_greedy_tokens'] == [510], 'original greedy token')
    for filename, pin in pins.items():
        require(sha(root/filename) == pin, 'tap changed during verification')
    require(sha(manifest) == manifest_sha, 'manifest changed during verification')
    require(sha(model) == MODEL_SHA, 'model changed during verification')
    return {'status':'REFERENCE_VERIFIED', 'numeric_gpu_pass':None,
            'manifest_sha256':manifest_sha, 'model_sha256':MODEL_SHA,
            'tap_sha256':pins, 'finite_values':total, 'input_ids':IDS,
            'fp16kv_expected_checks':401, 'expected_greedy':510,
            'scope':'retained original reference integrity; no CPU model or GPU execution'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    with a.report.open('x') as f:
        try:
            result = verify(a.reference, a.model, a.manifest_sha256)
        except Exception as e:
            json.dump({'status':'FAIL','error':str(e),'numeric_gpu_pass':None}, f, indent=2)
            raise
        json.dump(result, f, indent=2)
        f.write('\n')

if __name__ == '__main__':
    main()
