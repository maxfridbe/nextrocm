#!/usr/bin/env python3
"""CPU-only tests; no HIP loading, GPU, network, model files or services."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
build = module('architecture_build', ROOT/'kernels/build.py')
gate = module('architecture_gate', ROOT/'tooling/gate.py')

def fixture():
    device = dict(gfx='gfx803', wave_size=64, total_bytes=4<<30, units=8, driver='fixture', compiler='fixture')
    baseline = dict(device=device, commit='base', runtime_sha256='baseline-fixture', exact='output-fixture', numeric_gate_pass=True,
                    metrics={'seconds': {'direction':'lower', 'value':10.0}})
    data = dict(version=1, policy=dict(speed_regression_fraction=.03, default_runs=2),
                shared={'kernels/ns/':['*']}, engines={'llm/example':dict(paths=['llm/example/','kernels/llm/example/'],cards={'fixture':baseline})})
    run = dict(exact='output-fixture', exit_code=0, group_empty=True, memory_gate_pass=True, idle_gate_pass=True, metrics={'seconds':10.1})
    result = {'llm/example':{'fixture':dict(device=device, commit='head',runtime_sha256='candidate-fixture',executable_sha256='elf-fixture',numeric_gate_pass=True,runs=[copy.deepcopy(run),copy.deepcopy(run)])}}
    return data, result

class Architecture(unittest.TestCase):
    def test_actual_props_function_with_mock_hip(self):
        # Compile the exact implementation, not a rewritten model of its guards.
        text=(ROOT/'kernels/ns/ns.hip').read_text()
        body=text[text.index('int ns_gpu_props('):text.index('// Legacy compatibility query:')]
        prefix=r'''
#include <cstdint>
#include <cstring>
#include <cassert>
#include <string>
struct hipDeviceProp_t { char gcnArchName[64]; size_t totalGlobalMem; int warpSize; int integrated; };
static int calls=0, mode=0;
int ns_fail(const std::string&) { return -1; }
int hipGetDeviceProperties(hipDeviceProp_t* p,int i) {
 ++calls; if(i>1 || mode==1)return -1;
 std::strcpy(p->gcnArchName,i==0?"gfx803":"gfx1035:xnack-");
 p->totalGlobalMem=uint64_t(4)<<30;p->warpSize=i==0?64:32;p->integrated=i;
 if(mode==2)p->warpSize=0;
 return 0;
}
#define NS_CU(x) do { if((x)!=0)return -1; } while(0)
'''
        suffix=r'''
int main() {
 uint64_t total=7;int wave=9,integrated=11;char gfx[64];std::memset(gfx,42,sizeof gfx);
 assert(ns_gpu_props(-1,&total,&wave,&integrated,gfx,sizeof gfx)==-1 && calls==0);
 assert(ns_gpu_props(0,nullptr,&wave,&integrated,gfx,sizeof gfx)==-1 && calls==0);
 assert(ns_gpu_props(0,&total,&wave,&integrated,gfx,0)==-1 && calls==0);
 for(int m=0;m<3;++m){mode=m;assert(ns_gpu_props(0,&total,&wave,&integrated,gfx,2)==-1);
  assert(total==7&&wave==9&&integrated==11);for(char c:gfx)assert(c==42);}
 mode=0;assert(ns_gpu_props(0,&total,&wave,&integrated,gfx,sizeof gfx)==0);
 assert(total==(uint64_t(4)<<30)&&wave==64&&integrated==0&&std::strcmp(gfx,"gfx803")==0);
 assert(ns_gpu_props(1,&total,&wave,&integrated,gfx,sizeof gfx)==0);
 assert(wave==32&&integrated==1&&std::strcmp(gfx,"gfx1035:xnack-")==0);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            cpp=Path(tmp)/'props.cpp';exe=Path(tmp)/'props'
            cpp.write_text(prefix+body+suffix)
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-Wall','-Wextra','-Werror',str(cpp),'-o',str(exe)],check=True,timeout=30)
            subprocess.run([str(exe)],check=True,timeout=5)

    def test_manifest_sources_and_flags(self):
        m=json.loads((ROOT/'kernels/build-manifest.json').read_text())
        self.assertEqual(len(m['units']),len({u['source'] for u in m['units']}))
        for target,wave in [('gfx803',64),('gfx1035',32)]:
            commands=build.plan(m,target,Path('/not-created-build-output'),'hipcc')
            self.assertTrue(any('libnextrocm-runtime.so' in c[-1] for c in commands))
            low=[c for c in commands if str(ROOT/'kernels/audio/lowvram/ops.hip') in c]
            self.assertEqual(len(low),1);self.assertIn('-DNR_LOWVRAM_WAVE_SIZE='+str(wave),low[0])
            self.assertFalse(any('.sm' in x for c in commands for x in c))
            quant = 'quant_batch32.hip' if target == 'gfx803' else 'quant_wave32_dispatch.hip'
            strict=[c for c in commands if str(ROOT/'kernels/llm/inkling2'/quant) in c]
            self.assertIn('-ffp-contract=off',strict[0]); self.assertIn('-fno-unsafe-math-optimizations',strict[0])

    def test_all_silo_closures_and_target_exports(self):
        m=json.loads((ROOT/'kernels/build-manifest.json').read_text())
        self.assertEqual(len(m['units']),48)
        self.assertEqual(len(m['silos']),18)
        for target in ('gfx803','gfx1035'):
            resolved=build.selected_manifest(m,target)
            units={u['source']:u['flags'] for u in resolved['units']}
            covered=set()
            for silo in resolved['silos']:
                self.assertTrue(silo['exports'])
                self.assertEqual(len(silo['exports']),len(set(silo['exports'])))
                for u in silo['units']:
                    self.assertEqual(u['flags'],units[u['source']])
                    covered.add(u['source'])
            self.assertEqual(covered,{u['source'] for u in resolved['units'] if u['kind']!='shared'})
            commands=build.plan(m,target,Path('/unused'),'hipcc')
            compiles=[c for c in commands if '-c' in c]
            self.assertEqual(len(compiles),49) # reuse identical objects, not duplicate compilation
            exports=next(s['exports'] for s in resolved['silos'] if s['architecture']=='inkling2')
            self.assertEqual('ni2_quant_matvec_batch32' in exports,target=='gfx803')
            self.assertEqual('ni2_depthmix_baseline' in exports,target=='gfx803')
            sources=[c[c.index('-c')+1] for c in compiles]
            self.assertEqual(any(x.endswith('/mix_depth_parallel.hip') for x in sources),target=='gfx803')
        bad=copy.deepcopy(m);bad['silos'].append(bad['silos'][0])
        with self.assertRaises(ValueError):build.plan(bad,'gfx803',Path('/unused'),'hipcc')
        bad=copy.deepcopy(m);bad['silos'][0]['units']*=2
        with self.assertRaises(ValueError):build.plan(bad,'gfx803',Path('/unused'),'hipcc')

    def test_duplicate_source_rejected(self):
        m=json.loads((ROOT/'kernels/build-manifest.json').read_text());m['units'].append(m['units'][0])
        with self.assertRaises(ValueError):build.plan(m,'gfx803',Path('/unused'),'hipcc')

    def test_source_escape_rejected(self):
        with self.assertRaises(ValueError):build.source('/etc/passwd')

    def test_empty_baselines_are_unvalidated(self):
        d=json.loads((ROOT/'tooling/gate-baselines.json').read_text());gate.validate(d)
        p=gate.affected(d,['kernels/ns/ns.h']);self.assertEqual(p['status'],'UNVALIDATED')
        with self.assertRaises(ValueError):gate.compare(d,p,{})

    def test_empty_metrics_rejected(self):
        d,_=fixture();d['engines']['llm/example']['cards']['fixture']['metrics']={}
        with self.assertRaises(ValueError):gate.validate(d)

    def test_cross_family_rename_keeps_both_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            def git(*args):
                return subprocess.check_output(['git',*args],cwd=tmp,stderr=subprocess.DEVNULL,text=True)
            git('init');git('config','user.email','fixture@example.invalid');git('config','user.name','Fixture')
            p=Path(tmp);(p/'llm/a').mkdir(parents=True);(p/'image/b').mkdir(parents=True)
            (p/'llm/a/op.cpp').write_text('unchanged operator fixture\n')
            git('add','.');git('commit','-m','base');base=git('rev-parse','HEAD').strip()
            (p/'llm/a/op.cpp').rename(p/'image/b/op.cpp');git('add','-A');git('commit','-m','move')
            paths=git('diff','--no-renames','--name-only',base,'HEAD').splitlines()
            self.assertEqual(set(paths),{'llm/a/op.cpp','image/b/op.cpp'})

    def test_shared_change_and_unknown_paths(self):
        d,_=fixture();self.assertEqual(gate.affected(d,['kernels/ns/ns.h'])['engines'],['llm/example'])
        self.assertEqual(gate.affected(d,['unknown.rs'])['status'],'UNVALIDATED')

    def test_regression_and_exact_failures(self):
        d,r=fixture();gate.validate(d);p=gate.affected(d,['llm/example/lib.rs'])
        self.assertEqual(gate.compare(d,p,r)['status'],'PASS')
        r['llm/example']['fixture']['runs'][0]['exact']='wrong'
        with self.assertRaises(ValueError):gate.compare(d,p,r)
        d,r=fixture()
        for run in r['llm/example']['fixture']['runs']:run['metrics']['seconds']=10.31
        self.assertEqual(gate.compare(d,p,r)['status'],'FAIL')

    def test_nonfinite_missing_numeric_device_and_cleanup(self):
        for mutation in ('nan','numeric','device','cleanup','repeat'):
            d,r=fixture();p=gate.affected(d,['llm/example/lib.rs']);x=r['llm/example']['fixture']
            if mutation=='nan':x['runs'][0]['metrics']['seconds']=float('nan')
            if mutation=='numeric':x['numeric_gate_pass']=False
            if mutation=='device':x['device']=dict(x['device'],wave_size=32)
            if mutation=='cleanup':x['runs'][0]['group_empty']=False
            if mutation=='repeat':x['runs'].pop()
            with self.assertRaises(ValueError,msg=mutation):gate.compare(d,p,r)

if __name__=='__main__':unittest.main()
