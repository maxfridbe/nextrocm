#!/usr/bin/env python3
"""Verify existing pinned original CPU tap fixtures; never performs inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

# Manifests independently audited with every original local array. These are
# original CPU artifacts, not manifests regenerated from the native candidate.
CONTRACTS = {
 'qwen3moe': ('4e56c630c1e7210e61b699bd36a60c7df7fb797d87994724043a8b055169e43c',
              'fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad',18556689568,966,9,9707),
 'qwen35': ('7a75daf715559c72013d8a2710e47bb8c23d7a6349aa28c222b14a9a2ced563f',
            '00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4',2740937888,838,19,760),
 'qwen36': ('4a88a818c4a3f29ffc07d133d02e0cd73705f35b23d4b45212ea35e4b5b2f116',
            'ae2fb73ac0da875640269f1e65e9c7fb415b066c6d544c3eef9adb0d03f04792',21166758336,966,17,8160),
}

def require(ok, why):
    if not ok:
        raise ValueError(why)

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def unique(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, 'duplicate JSON key')
        d[k] = v
    return d

def verify(family, root, model):
    manifest_pin, model_pin, model_bytes, taps, positions, greedy = CONTRACTS[family]
    root, model = root.resolve(strict=True), model.resolve(strict=True)
    mf = root/'manifest.json'
    require(mf.stat().st_size < 4*1024*1024, 'manifest size bound')
    require(sha(mf) == manifest_pin, 'original manifest identity')
    m = json.loads(mf.read_text(), object_pairs_hook=unique)
    require(m['model_sha256'] == model_pin, 'manifest model pin')
    require(model.stat().st_size == model_bytes and sha(model) == model_pin, 'actual full model pin')
    require(len(m['input_ids']) == positions and len(m['taps']) == taps, 'original inventory geometry')
    paths, keys, top, total = {}, set(), {}, 0
    for row in m['taps']:
        mode, name, filename = row['mode'], row['name'], row['file']
        key = (mode, name)
        require(mode in ('fp32','fp16kv') and key not in keys, 'tap mode or duplicate')
        require(Path(filename).name == filename and filename not in paths, 'tap path or duplicate')
        keys.add(key)
        p = root/filename
        require(not p.is_symlink() and p.resolve(strict=True).parent == root, 'tap containment')
        shape = row['shape']
        require(isinstance(shape,list) and shape and all(type(n) is int and n > 0 for n in shape), 'shape dimensions')
        elements = math.prod(shape)
        require(elements <= 64*1024*1024 and p.stat().st_size == elements*4, 'tap exact bytes/bound')
        h, count, maximum, argmax = hashlib.sha256(), 0, -math.inf, None
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
                for value, in struct.iter_unpack('<f', chunk):
                    require(math.isfinite(value), 'nonfinite tap')
                    if name == 'result_output' and value > maximum:
                        maximum, argmax = value, count
                    count += 1
        require(count == elements and h.hexdigest() == row['sha256'], 'tap hash/elements')
        if name == 'result_output':
            require(argmax == greedy == m['runs'][mode]['top10'][0]['id'], 'actual original logit argmax')
            top[mode] = argmax
        paths[filename] = row['sha256']
        total += count
    require(set(top) == {'fp32','fp16kv'}, 'both logit modes required')
    for filename, pin in paths.items():
        require(sha(root/filename) == pin, 'tap changed during verification')
    require(sha(mf) == manifest_pin and sha(model) == model_pin, 'manifest/model changed')
    return {'status':'REFERENCE_VERIFIED','family':family,'numeric_gpu_pass':None,
            'manifest_sha256':manifest_pin,'model_sha256':model_pin,'input_ids':m['input_ids'],
            'tap_count':len(paths),'finite_values':total,'argmax':top,'tap_sha256':paths,
            'scope':'original CPU artifact integrity/full model identity; no inference executed'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('family', choices=CONTRACTS)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a = p.parse_args()
    with a.report.open('x') as f:
        try:
            result = verify(a.family,a.reference,a.model)
        except Exception as e:
            json.dump({'status':'FAIL','error':str(e),'numeric_gpu_pass':None},f,indent=2)
            raise
        json.dump(result,f,indent=2)
        f.write('\n')

if __name__ == '__main__':
    main()
