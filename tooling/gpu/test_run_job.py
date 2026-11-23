import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run_job.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class RunnerTests(unittest.TestCase):
    def test_changed_locked_prelaunch_pin_never_launches(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(r,'file_sha256',return_value='changed'), patch.object(r.subprocess,'Popen') as launch:
                rc=r.run(root/'out',[sys.executable,'-c','pass'],3,r.RESERVE,root/'lock',
                         device_cap=4026531840,expected_pins={str(Path(sys.executable).resolve()):'authorized'})
            self.assertEqual(rc,1);launch.assert_not_called()
            report=json.loads((root/'out/report.json').read_text())
            self.assertIn('locked prelaunch identities',report['error'])

    def invoke(self, code, drm=1024, timeout=3, reserve=None):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            sample={'drm_allocated_bytes':drm,'drm_clients':int(drm>0)}
            with patch.object(r,'precheck_amd_clients',return_value={'clients':[]}), patch.object(r,'memory_available',return_value=16*1024**3 if reserve is None else reserve), patch.object(r,'strict_sample',return_value=sample):
                rc=r.run(root/'out',[sys.executable,'-c',code],timeout,r.RESERVE,root/'lock',device_cap=4026531840)
            return rc,json.loads((root/'out/report.json').read_text())
    def test_precheck_permission_context(self):
        from types import SimpleNamespace
        proc=Path('/proc/12345')
        def entries(path):
            if path==Path('/proc'): return iter([proc])
            raise PermissionError(13,'Permission denied',str(path))
        with patch.object(r.Path,'iterdir',entries), patch.object(r.Path,'stat',return_value=SimpleNamespace(st_uid=r.os.getuid())):
            with self.assertRaisesRegex(RuntimeError,r'pid=12345 path=/proc/12345/fdinfo errno=13; exclusivity unverified'):
                r.precheck_amd_clients()
    def test_kfd_mode_requires_device(self):
        with self.assertRaisesRegex(ValueError,'requires --amd-device'):r.precheck_amd_clients('kfd-compute')
    def test_kfd_context_rejected(self):
        with patch.object(r,'kfd_inventory',return_value=['42']):
            with self.assertRaisesRegex(RuntimeError,'existing KFD'):r.precheck_amd_clients('kfd-compute',Path('/sys/devices/mock'))
    def test_kfd_busy_rejected(self):
        with patch.object(r,'kfd_inventory',return_value=[]), patch.object(r,'idle_amd_sample',return_value={'gpu_busy_percent':1}):
            with self.assertRaisesRegex(RuntimeError,'not idle'):r.precheck_amd_clients('kfd-compute',Path('/sys/devices/mock'))
    def test_kfd_visibility_scope(self):
        from types import SimpleNamespace
        proc=Path('/proc/12345')
        def entries(path):
            if path==Path('/proc'):return iter([proc])
            raise PermissionError(13,'denied',str(path))
        with patch.object(r.Path,'iterdir',entries), patch.object(r.Path,'stat',return_value=SimpleNamespace(st_uid=r.os.getuid())), patch.object(r,'unknown_identity',return_value={'pid':12345,'certified_no_gpu_client':False}), patch.object(r,'kfd_inventory',return_value=[]), patch.object(r,'idle_amd_sample',return_value={'gpu_busy_percent':0}), patch.object(r.time,'sleep'):
            result=r.precheck_amd_clients('kfd-compute',Path('/sys/devices/mock'))
        self.assertEqual(len(result['idle_samples']),3);self.assertFalse(result['graphics_exclusivity_verified']);self.assertEqual(result['inaccessible_pids'][0]['pid'],12345)
    def test_kfd_missing_visibility_fails(self):
        with patch.object(r,'kfd_inventory',side_effect=PermissionError(13,'denied')):
            with self.assertRaises(PermissionError):r.precheck_amd_clients('kfd-compute',Path('/sys/devices/mock'))
    def test_device_vendor_and_busy_validation(self):
        with patch.object(r.Path,'resolve',return_value=Path('/sys/devices/mock')),patch.object(r.Path,'read_text',return_value='0x8086'):
            with self.assertRaisesRegex(ValueError,'not AMD'):r.idle_amd_sample(Path('/sys/devices/mock'))
        with patch.object(r.Path,'resolve',return_value=Path('/sys/devices/mock')),patch.object(r.Path,'read_text',side_effect=['0x1002','invalid']):
            with self.assertRaisesRegex(RuntimeError,'invalid GPU busy'):r.idle_amd_sample(Path('/sys/devices/mock'))
    def test_telemetry_validation(self):
        good={'drm-client-id':'3','drm-memory-vram':'10 KiB','drm-memory-gtt':'2 kB'}
        self.assertEqual(r.amd_allocated(good),12288)
        for value in ('','-1 KiB','3 MB','nan KiB'):
            with self.assertRaises(RuntimeError): r.amd_allocated(dict(good,**{'drm-memory-vram':value}))
        with self.assertRaises(RuntimeError): r.amd_allocated({'drm-client-id':'3'})
    def test_stream_hash(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data';data=b'a'*(2*1024*1024+3);p.write_bytes(data)
            with patch.object(r,'memory_available',return_value=16*1024**3):
                self.assertEqual(r.file_sha256(p),hashlib.sha256(data).hexdigest())
            with patch.object(r,'memory_available',return_value=0):
                with self.assertRaises(RuntimeError):r.file_sha256(p)
    def test_success_no_numeric_claim(self):
        rc,x=self.invoke('pass');self.assertEqual(rc,0);self.assertTrue(x['group_verified_empty']);self.assertIsNone(x['numeric_pass'])
    def test_nonzero(self):
        rc,x=self.invoke('raise SystemExit(3)');self.assertEqual(rc,1);self.assertTrue(x['group_verified_empty'])
    def test_no_drm(self):
        rc,x=self.invoke('pass',drm=0);self.assertEqual(rc,1);self.assertIn('no positive DRM',x['error'])
    def test_cap(self):
        rc,x=self.invoke('import time;time.sleep(10)',drm=4026531840+1);self.assertEqual(rc,1);self.assertTrue(x['group_verified_empty'])
    def test_parent_sample_not_discarded(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=iter([{'drm_allocated_bytes':4026531840+1,'drm_clients':1}, {'drm_allocated_bytes':0,'drm_clients':0}])
            with patch.object(r,'precheck_amd_clients',return_value={'clients':[]}), patch.object(r,'memory_available',return_value=16*1024**3), patch.object(r,'strict_sample',side_effect=lambda pid: next(rows)):
                rc=r.run(root/'out',[sys.executable,'-c','import time;time.sleep(10)'],3,r.RESERVE,root/'lock',device_cap=4026531840)
            result=json.loads((root/'out/report.json').read_text())
            self.assertEqual(rc,1);self.assertIn('DRM cap',result['error']);self.assertTrue(result['group_verified_empty'])
    def test_host_reserve(self):
        rc,x=self.invoke('pass',reserve=1);self.assertEqual(rc,1);self.assertNotIn('pid',x)
    def test_timeout(self):
        rc,x=self.invoke('import time;time.sleep(10)',timeout=1);self.assertEqual(rc,1);self.assertTrue(x['timed_out']);self.assertTrue(x['group_verified_empty'])
    def test_surviving_child(self):
        rc,x=self.invoke('import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(10)"])');self.assertEqual(rc,1);self.assertIn('live owned children',x['error']);self.assertTrue(x['group_verified_empty'])

if __name__=='__main__': unittest.main()
