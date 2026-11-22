#!/usr/bin/env python3
"""Parent-queued 30-second probe: context open/memory/close, no tensor allocation.

Context streams/events and driver initialization may consume memory. This is
an instantaneous free-memory observation, not model/request fit validation.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import time

REQUIRED_FREE = 4237469188


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def classify(total, free):
    if total <= 0 or free < 0 or free > total:
        raise ValueError('unknown or invalid GPU memory telemetry')
    return {
        'total_bytes': total, 'free_bytes': free,
        'qwen_required_free_bytes': REQUIRED_FREE,
        'free_minus_required_bytes': free - REQUIRED_FREE,
        'qwen_load_admission': 'FREE_BOUND_SATISFIED' if free >= REQUIRED_FREE else 'BLOCKED_CAPACITY',
        'model_fit_pass': None, 'numeric_pass': None,
    }


def measure(lib, device):
    def bind(name, args, result=C.c_int):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, result
        return fn
    error = bind('ns_last_error', [], C.c_char_p)
    op = bind('ns_gpu_open', [C.c_int, C.POINTER(C.c_void_p)])
    close = bind('ns_gpu_close', [C.c_void_p], None)
    memory = bind('ns_gpu_memory', [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_int64)])
    gpu = C.c_void_p()
    def check(rc):
        if rc:
            raise RuntimeError((error() or b'unknown native error').decode(errors='replace'))
    try:
        check(op(device, C.byref(gpu)))
        if not gpu.value:
            raise RuntimeError('successful open returned null GPU')
        total, free = C.c_uint64(), C.c_int64(-1)
        check(memory(gpu, C.byref(total), C.byref(free)))
        result = classify(total.value, free.value)
        result['observed_monotonic_seconds'] = time.monotonic()
        return result
    finally:
        if gpu.value:
            close(gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--runtime-sha256', required=True)
    parser.add_argument('--source-pins', type=Path, required=True)
    parser.add_argument('--source-pins-sha256', required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0 or any(len(s) != 64 or any(c not in '0123456789abcdef' for c in s)
                              for s in [args.runtime_sha256, args.source_pins_sha256]):
        parser.error('nonnegative device and lowercase SHA256 pins required')
    # Exclusive output; never overwrite a prior measurement.
    with args.report.open('x') as stream:
        result = {'status': 'INCOMPLETE', 'device': args.device,
                  'scope': __doc__, 'tensor_allocations_requested': 0,
                  'allocator_drop_pass': None}
        errors = []
        runtime = args.runtime.resolve()
        script = Path(__file__).resolve()
        script_before = digest(script)
        started = time.monotonic()
        try:
            if digest(runtime) != args.runtime_sha256:
                raise ValueError('runtime SHA256 mismatch')
            if digest(args.source_pins) != args.source_pins_sha256:
                raise ValueError('source-pins SHA256 mismatch')
            result['source_pins'] = json.loads(args.source_pins.read_text())
            result.update(measure(C.CDLL(str(runtime)), args.device))
        except Exception as exc:
            errors.append(f'{type(exc).__name__}: {exc}')
        finally:
            for name, path, expected in [('runtime', runtime, args.runtime_sha256),
                                          ('script', script, script_before),
                                          ('source_pins', args.source_pins, args.source_pins_sha256)]:
                try:
                    actual = digest(path)
                    result[name + '_sha256_after'] = actual
                    if actual != expected:
                        errors.append(name + ' changed or pin mismatch')
                except Exception as exc:
                    errors.append(name + ': ' + str(exc))
            result.update(status='FAIL' if errors else 'MEASUREMENT_COMPLETE', errors=errors,
                          runtime_sha256_before=args.runtime_sha256,
                          script_sha256_before=script_before,
                          elapsed_seconds=time.monotonic() - started)
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result), flush=True)
        return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
