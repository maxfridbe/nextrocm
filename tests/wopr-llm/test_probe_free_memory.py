import ctypes as C
from types import SimpleNamespace
import unittest
import probe_free_memory as probe


class Function:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class ProbeTests(unittest.TestCase):
    def test_threshold_is_inclusive_but_not_fit_claim(self):
        self.assertEqual(probe.classify(1 << 32, probe.REQUIRED_FREE)['qwen_load_admission'], 'FREE_BOUND_SATISFIED')
        result = probe.classify(1 << 32, probe.REQUIRED_FREE - 1)
        self.assertEqual(result['qwen_load_admission'], 'BLOCKED_CAPACITY')
        self.assertIsNone(result['model_fit_pass'])

    def test_unknown_and_invalid_telemetry_rejected(self):
        for total, free in [(0, 0), (1 << 32, -1), (10, 11)]:
            with self.assertRaises(ValueError):
                probe.classify(total, free)

    def test_exact_abi_and_close_on_memory_error(self):
        closed = []
        def op(device, out):
            C.cast(out, C.POINTER(C.c_void_p))[0] = 123
            return 0
        lib = SimpleNamespace(ns_last_error=Function(lambda: b'memory failed'),
                              ns_gpu_open=Function(op),
                              ns_gpu_close=Function(lambda gpu: closed.append(gpu.value)),
                              ns_gpu_memory=Function(lambda *args: 1))
        with self.assertRaisesRegex(RuntimeError, 'memory failed'):
            probe.measure(lib, 0)
        self.assertEqual(closed, [123])
        self.assertEqual(lib.ns_gpu_memory.argtypes, [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_int64)])

    def test_unknown_memory_closes_context(self):
        closed = []
        def op(device, out):
            C.cast(out, C.POINTER(C.c_void_p))[0] = 456
            return 0
        def memory(gpu, total, free):
            C.cast(total, C.POINTER(C.c_uint64))[0] = 1 << 32
            return 0  # Deliberately leaves initialized sentinel -1 unchanged.
        lib = SimpleNamespace(ns_last_error=Function(lambda: b''), ns_gpu_open=Function(op),
                              ns_gpu_close=Function(lambda gpu: closed.append(gpu.value)),
                              ns_gpu_memory=Function(memory))
        with self.assertRaises(ValueError):
            probe.measure(lib, 0)
        self.assertEqual(closed, [456])


if __name__ == '__main__':
    unittest.main()
