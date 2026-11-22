import unittest
from pathlib import Path
from types import SimpleNamespace
import queue_benchmarks_resume_v2 as q


class Tests(unittest.TestCase):
    def test_cli_rounding_and_no_invented_ttft(self):
        r = q.timing('[prompt 24 tokens in 2.00s; 32 generated in 16.00s (2.00 tok/s)]\n')
        self.assertEqual(r['prompt_tokens_per_second_approx'], 12)
        self.assertEqual(r['generation_tokens_per_second'], 2)
        self.assertIsNone(r['time_to_first_token_seconds'])

    def test_missing_duplicate_invalid_and_inconsistent_timings(self):
        valid = '[prompt 24 tokens in 2.00s; 32 generated in 16.00s (2.00 tok/s)]'
        for text in ['', valid + '\n' + valid, valid.replace('32 generated', '33 generated'),
                     valid.replace('2.00 tok/s', '3.00 tok/s'), valid.replace('in 2.00s', 'in 0.00s')]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                q.timing(text)

    def test_direct_cli_and_model_pin(self):
        a = SimpleNamespace(runner=Path('/runner'), out=Path('/out'), lock=Path('/lock'),
                            runtime=Path('/runtime'), binary=Path('/cli'), amd_device=Path('/sys/device'))
        case = {'id': 'test', 'model': '/model', 'argv': ['$CLI', 'llm', 'generate', '/model']}
        c = q.command(a, case)
        self.assertEqual(c[c.index('--')+1:], ['/cli', 'llm', 'generate', '/model'])
        self.assertIn('/model', c[:c.index('--')])
        self.assertEqual(c[c.index('--timeout')+1], '300')


if __name__ == '__main__':
    unittest.main()
