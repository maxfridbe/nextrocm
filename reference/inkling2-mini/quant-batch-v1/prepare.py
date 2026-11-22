#!/usr/bin/env python3
"""CPU only. Four distinct vectors/full-row original decode and F64 for all66 cases."""
import argparse,hashlib,json,subprocess,pathlib,shutil
P=pathlib.Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=argparse.ArgumentParser()
 for n in ['fixtures','oracle','out']:a.add_argument('--'+n,type=P,required=True)
 a.add_argument('--manifest-sha',required=True);a=a.parse_args()
 assert sha(a.fixtures/'manifest.json')==a.manifest_sha
 m=json.loads((a.fixtures/'manifest.json').read_text());assert len(m['cases'])==66
 assert [c['name'] for c in m['cases']]==[f'actual-{i:02}' for i in range(26)]+[f'tail-{i:02}' for i in range(40)]
 for n,v in m['files'].items():assert P(n).name==n and sha(a.fixtures/n)==v['sha256']
 exe=sha(a.oracle);a.out.mkdir(exist_ok=False);cases=[]
 for c in m['cases']:
  name=c['name'];prefix=a.out/name
  cmd=[str(a.oracle.resolve()),str(c['type']),str(c['rows']),str(c['cols']),str((a.fixtures/c['weight']).resolve()),str((a.fixtures/c['scale']).resolve()) if c['scale'] else '-',str(prefix.resolve())]
  with (a.out/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
  for suf,size in [('x.f32',c['cols']*4),('wave.f32',c['rows']*4),('scalar.f32',c['rows']*4),('ref.f64',c['rows']*8),('bound.f64',c['rows']*8)]:
   data=P(str(prefix)+'.'+suf).read_bytes();assert len(data)==4*size and data[:size]==(a.fixtures/(name+'.'+suf)).read_bytes(),(name,suf,'originalvector0')
  for key in ['weight','scale']:
   if c[key]:shutil.copyfile(a.fixtures/c[key],a.out/c[key])
  if name.startswith('tail-'):shutil.copyfile(a.fixtures/(name+'.decoded.f32'),a.out/(name+'.decoded.f32'))
  cases.append(c);print('CPU_CASE_PASS',name,flush=True)
 assert sha(a.oracle)==exe and sha(a.fixtures/'manifest.json')==a.manifest_sha
 for n,v in m['files'].items():assert sha(a.fixtures/n)==v['sha256']
 out={'status':'FOUR_VECTOR_FULL_ROW_CPU_PASS','cases':cases,'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.out.iterdir()) if p.is_file()},'parent_manifest_sha256':a.manifest_sha,'oracle_sha256':exe,'builder_sha256':sha(P(__file__)),'scope':'All66originalmatrices,4distinctvectors,fullF64allrows andoriginaldecodebits; vector0 byteexactpriorCPUfixtures, no modelclaim.'}
 (a.out/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
