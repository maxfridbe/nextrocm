#!/usr/bin/env python3
"""Explicit HIP translation units, preserved arithmetic flags, isolated outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
KINDS = ('llm', 'image', 'audio', 'video', 'model3d')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source(name):
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError('missing or escaping source: ' + name)
    return p

def plan(manifest, target, out, hipcc):
    profile = manifest['profiles'][target]
    commands, objects = [], {k: [] for k in ('shared',) + KINDS}
    seen = set()
    for u in manifest['units']:
        name = u['source']; source(name)
        if name in seen or u['kind'] not in objects:
            raise ValueError('duplicate source or unknown kind: ' + name)
        seen.add(name)
        obj = out / 'objects' / (name.replace('/', '_') + '.o')
        flags = [f.replace('{wave_size}', str(profile['wave_size'])) for f in u['flags']]
        commands.append([hipcc, *flags, '--offload-arch=' + target, '-x', 'hip', '-c', str(source(name)), '-o', str(obj)])
        objects[u['kind']].append(str(obj))
    # Shared ABI and utility kernels stay together. All old exports also remain
    # available through the runtime compatibility library.
    for kind in KINDS:
        if objects[kind]:
            commands.append([hipcc, '--offload-arch=' + target, '-shared', *objects['shared'], *objects[kind], '-o', str(out / ('libnextrocm-' + kind + '.so'))])
    commands.append([hipcc, '--offload-arch=' + target, '-shared', *[o for v in objects.values() for o in v], '-o', str(out / 'libnextrocm-runtime.so')])
    for silo in manifest['silos']:
        arch = silo['architecture']
        if not re.fullmatch('[a-z0-9-]+', arch):
            raise ValueError('invalid architecture name')
        base_ops = set(silo['exports'])
        for variant in [None, *silo.get('variants', [])]:
            spec = silo if variant is None else variant
            suffix = ''
            if variant is not None:
                if not re.fullmatch('[a-z0-9-]+', variant['name']):
                    raise ValueError('invalid variant name')
                if not set(variant['exports']) or not set(variant['exports']) <= base_ops:
                    raise ValueError('variant must export a nonempty subset of base ops')
                source(variant['manifest'])
                suffix = '@' + variant['name']
            objs = []
            for i, u in enumerate(spec['units']):
                obj = out / 'objects' / f'silo-{arch}{suffix}-{i}.o'
                flags = [f.replace('{wave_size}', str(profile['wave_size'])) for f in u['flags']]
                commands.append([hipcc, *flags, '--offload-arch=' + target, '-x', 'hip', '-c', str(source(u['source'])), '-o', str(obj)])
                objs.append(str(obj))
            if not objs:
                raise ValueError('empty silo')
            commands.append([hipcc, '--offload-arch=' + target, '-shared', *objs, '-Wl,-Bsymbolic', '-Wl,--version-script=' + str(out / 'silo' / f'{arch}{suffix}.map'), '-o', str(out / 'silo' / f'libnextrocm-{arch}{suffix}.so')])
    return commands

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--target', required=True, choices=('gfx803', 'gfx1035'))
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--hipcc', default=os.environ.get('HIPCC', 'hipcc'))
    ap.add_argument('--execute', action='store_true', help='compile only; default prints plan')
    a = ap.parse_args()
    if not a.out.is_absolute() or a.out.exists():
        ap.error('--out must be a new absolute directory')
    m = json.loads((ROOT / 'kernels/build-manifest.json').read_text())
    commands = plan(m, a.target, a.out, a.hipcc)
    if not a.execute:
        print(json.dumps({'target': a.target, 'scope': m['profiles'][a.target]['scope'], 'commands': commands}, indent=2)); return
    # Hash all kernel/include sources, including transitive dependencies. Fresh
    # objects avoid timestamp-only cache invalidation and removed-object reuse.
    paths = sorted(p for p in (ROOT / 'kernels').rglob('*') if p.is_file() and p.suffix in ('.hip', '.h', '.hpp', '.cpp', '.inc', '.json', '.py', '.sh', '.yml'))
    before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    a.out.mkdir(); (a.out / 'objects').mkdir(); (a.out / 'silo').mkdir()
    for silo in m['silos']:
        for spec in [silo, *silo.get('variants', [])]:
            suffix = '' if spec is silo else '@' + spec['name']
            names = spec['exports']
            if not names or any(not re.fullmatch('[A-Za-z_][A-Za-z_0-9]*', n) for n in names) or len(set(names)) != len(names):
                raise ValueError('invalid or duplicate exported operation')
            (a.out / 'silo' / f"{silo['architecture']}{suffix}.map").write_text('{ global: ' + '; '.join(names) + '; local: *; };\n')
    (a.out / 'source-before.json').write_text(json.dumps(before, indent=2) + '\n')
    (a.out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
    with (a.out / 'build.log').open('w') as log:
        subprocess.run([a.hipcc, '--version'], stdout=log, stderr=log, check=True)
        for cmd in commands:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log, check=True)
    for silo in m['silos']:
        for spec in [silo, *silo.get('variants', [])]:
            suffix = '' if spec is silo else '@' + spec['name']
            library = a.out / 'silo' / f"libnextrocm-{silo['architecture']}{suffix}.so"
            output = subprocess.check_output(['nm', '-D', '--defined-only', '--format=posix', str(library)], text=True)
            actual = {line.split()[0] for line in output.splitlines() if line.strip()}
            if actual != set(spec['exports']):
                raise RuntimeError('actual silo exports differ from declaration: ' + str(library))
            library.with_suffix('.exports.json').write_text(json.dumps(sorted(actual), indent=2) + '\n')
        for v in silo.get('variants', []):
            (a.out / 'silo' / f"libnextrocm-{silo['architecture']}@{v['name']}.yml").write_bytes(source(v['manifest']).read_bytes())
    after = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    (a.out / 'source-after.json').write_text(json.dumps(after, indent=2) + '\n')
    if after != before:
        raise RuntimeError('source changed during build')
    binaries = {str(p.relative_to(a.out)): digest(p) for p in sorted(a.out.rglob('*.so'))}
    (a.out / 'libraries.json').write_text(json.dumps({'target': a.target, 'libraries': binaries, 'numeric_validation_pass': False}, indent=2) + '\n')

if __name__ == '__main__':
    main()
