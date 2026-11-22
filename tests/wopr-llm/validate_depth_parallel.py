#!/usr/bin/env python3
"""Parent-run bounded native depth/router/control tests against frozen CPU fixtures."""
import argparse,json,struct,time,os,ctypes as C,math,statistics
from pathlib import Path
import gpu_validation as V

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--runtime-sha',required=True);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert V.sha(a.runtime)==a.runtime_sha and V.sha(a.fixtures/'manifest.json')==a.manifest_sha
 m=json.loads((a.fixtures/'manifest.json').read_text());assert m['status']=='CPU_MIX_FIXTURES_PASS' and len(m['cases'])==36 and m['gate']['abs']==m['gate']['rel']==2e-5
 for n,r in m['files'].items():
  assert Path(n).name==n and V.sha(a.fixtures/n)==r['sha256'];shape,_=V.load(a.fixtures/n);assert shape==r['shape']
 a.out.mkdir(exist_ok=False);os.environ['NS_PROFILE']='gpu';g=V.GPU(a.runtime);fn={'depth':g.bind('ni2_depthmix',[V.P,V.P,V.P,V.I,V.I,V.I,V.P]),'router':g.bind('ni2_router',[V.P,V.P,V.P,V.P,V.P]),'control':g.bind('ni2_loop_control',[V.P,V.P,V.P,V.P,V.P,V.I,V.P]),'scale':g.bind('ni2_scale',[V.P,V.P,V.I,__import__('ctypes').c_float,V.P]),'add':g.bind('ni2_add',[V.P,V.P,V.P,V.I,V.P]),'axpy':g.bind('ni2_axpy',[V.P,V.P,V.P,V.I,V.P])};results=[];negative=0;start=time.monotonic();report={'status':'INCOMPLETE','runtime_sha256':a.runtime_sha,'manifest_sha256':a.manifest_sha,'validator_sha256':V.sha(__file__),'helper_sha256':V.sha(V.__file__),'cases':results}
 try:
  assert g.stats()[0]==0
  baseline=g.bind('ni2_depthmix_baseline',[V.P,V.P,V.P,V.I,V.I,V.I,V.P])
  stamp=g.bind('ns_stamp',[V.P,C.POINTER(V.L)])
  elapsed=g.bind('ns_elapsed',[V.P,V.L,V.L,C.POINTER(C.c_double)])
  depth_bits=0;synthetic=[]
  def timed(f,args):
   begin=V.L();end=V.L();value=C.c_double()
   g.ok(stamp(g.g,C.byref(begin)));g.ok(f(*args));g.ok(stamp(g.g,C.byref(end)));g.sync()
   g.ok(elapsed(g.g,begin,end,C.byref(value)));assert math.isfinite(value.value) and value.value>=0
   return value.value

  for c in m['cases']:
   op=c['op'];params=c['parameters'];expected=V.load(a.fixtures/c['outputs']['expected'])[1]
   for inplace in ((False,True) if op in ('control','scale','add') else (False,)):
    bs={k:V.Buffer(g,V.load(a.fixtures/n)[1]) for k,n in c['inputs'].items()};ids=V.Buffer(g,[37.]*8) if op=='router' else None
    if inplace:out=bs[{'control':'current','scale':'x','add':'a'}[op]]
    elif op=='axpy':out=bs['initial']
    else:out=V.Buffer(g,[37.]*len(expected))
    def call():
     d={k:b.p for k,b in bs.items()};f=fn[op]
     if op=='depth':return f(g.g,d['states'],d['queries'],params['count'],params['nq'],params['width'],out.p)
     if op=='router':return f(g.g,d['logits'],d['bias'],ids.p,out.p)
     if op=='control':return f(g.g,d['current'],d['message'],d['gain'],d['forget'],params['n'],out.p)
     if op=='scale':return f(g.g,d['x'],params['n'],params['factor'],out.p)
     if op=='add':return f(g.g,d['a'],d['b'],params['n'],out.p)
     return f(g.g,d['x'],d['weight'],params['n'],out.p)
    g.ok(call());actual=out.read();metrics=V.compare(actual,expected);assert metrics['passed'],c['name']
    if op=='depth':
     baseline_out=V.Buffer(g,[37.]*len(expected))
     g.ok(baseline(g.g,bs['states'].p,bs['queries'].p,params['count'],params['nq'],params['width'],baseline_out.p))
     assert baseline_out.read().tobytes()==actual.tobytes(),'original depth full bits'
     depth_bits+=1
    if op in ('add','scale','axpy'):assert actual.tobytes()==expected.tobytes(),'separate RN exact bits'
    native_ids=None
    if ids:
     native_ids=ids.read().tobytes();refids=V.load(a.fixtures/c['outputs']['ids'])[1];assert list(struct.unpack('<8i',native_ids))==[int(x) for x in refids],c['name']+' routing IDs';(a.out/(c['name']+'.ids')).write_bytes(native_ids)
    for b in bs.values():
     if b is not out:b.unchanged()
    name=c['name']+('-inplace' if inplace else '-separate');(a.out/(name+'.f32')).write_bytes(actual.tobytes());results.append({'name':name,'op':op,**metrics})
    for b in bs.values():b.reset()
    if all(out is not b for b in bs.values()):out.reset()
    if ids:ids.reset()
    g.ok(call());assert out.read().tobytes()==actual.tobytes()
    if ids:assert ids.read().tobytes()==native_ids
    for b in bs.values():
     if b is not out:b.unchanged()
    g.clear()
  # Every supported state/query count, reduction tails and actual width.
  for width in (1,3,16,127,128,129,255,256,257,2304,8191,8192):
   for count in range(1,6):
    for nq in range(1,5):
     states=V.Buffer(g,[((i*17+count*7)%131-65)/32 for i in range(count*width)])
     queries=V.Buffer(g,[((i*13+nq*3)%127-63)/64 for i in range(nq*width)])
     old=V.Buffer(g,[37.]*(nq*width));new=V.Buffer(g,[37.]*(nq*width))
     aa=(g.g,states.p,queries.p,count,nq,width,old.p);bb=aa[:-1]+(new.p,)
     g.ok(baseline(*aa));g.ok(fn['depth'](*bb));gold=old.read().tobytes();actual=new.read()
     assert all(math.isfinite(v) for v in actual) and actual.tobytes()==gold,'synthetic exact depth'
     old_ns=[];new_ns=[]
     if width==2304:
      for rep in range(5):
       for candidate in ((False,True) if rep%2==0 else (True,False)):
        (new_ns if candidate else old_ns).append(timed(fn['depth'] if candidate else baseline,bb if candidate else aa))
        assert (new if candidate else old).read().tobytes()==gold,'timed full bits'
     states.unchanged();queries.unchanged()
     synthetic.append({'width':width,'count':count,'nq':nq,'full_bits':True,'elements':nq*width,'baseline_ns':old_ns,'candidate_ns':new_ns,'baseline_median_ns':statistics.median(old_ns) if old_ns else None,'candidate_median_ns':statistics.median(new_ns) if new_ns else None})
     g.clear()
  report.update(depth_original_full_bit_cases=depth_bits,synthetic_depth_cases=synthetic)
  assert depth_bits==8 and len(synthetic)==240
  x=V.Buffer(g,[.25]*40960);y=V.Buffer(g,[.5]*40960);z=V.Buffer(g,[37.]*40960);w=V.Buffer(g,[.75]*40960)
  def reject(label,f):
   nonlocal negative
   before=[b.read().tobytes() for b in g.buffers];assert f()!=0,label;g.sync();assert [b.read().tobytes() for b in g.buffers]==before;negative+=1
  args=[g.g,x.p,y.p,2,1,16,z.p]
  for i,v in [(0,None),(1,None),(2,None),(3,0),(3,6),(4,0),(4,5),(5,0),(5,8193),(6,None),(6,x.p),(6,y.p),(1,x.p+1),(1,(1<<64)-4)]:
   b=args.copy();b[i]=v;reject('depth',lambda b=b:fn['depth'](*b))
  args=[g.g,x.p,y.p,z.p,w.p]
  for i,v in [(0,None),(1,None),(2,None),(3,None),(4,None),(3,x.p),(4,y.p),(4,z.p),(3,z.p+1),(1,(1<<64)-4)]:
   b=args.copy();b[i]=v;reject('router',lambda b=b:fn['router'](*b))
  # Isolate vector guard/alias conditions for each API.
  vectors=[('control',[g.g,x.p,y.p,w.p,w.p,16,z.p],5,6),('scale',[g.g,x.p,16,.5,z.p],2,4),('add',[g.g,x.p,y.p,16,z.p],3,4),('axpy',[g.g,x.p,y.p,16,z.p],3,4)]
  for op,args,ni,oi in vectors:
   for i,v in [(0,None),(1,None),(ni,0),(ni,-1),(ni,16777217),(oi,None),(oi,x.p+4),(1,x.p+1),(1,(1<<64)-4)]:
    b=args.copy();b[i]=v;reject(op,lambda b=b,op=op:fn[op](*b))
  for v in (float('inf'),float('nan')):reject('scale scalar',lambda v=v:fn['scale'](g.g,x.p,16,v,z.p))
  reject('axpy weight output alias',lambda:fn['axpy'](g.g,x.p,z.p,16,z.p))
  # Valid recovery calls after all rejected operations.
  g.ok(fn['depth'](g.g,x.p,y.p,2,1,16,z.p));g.ok(fn['router'](g.g,x.p,y.p,z.p,w.p));g.sync();g.clear();live,peak=g.stats();assert live==0 and 0<peak<64*1024*1024
  assert V.sha(a.runtime)==a.runtime_sha and V.sha(a.fixtures/'manifest.json')==a.manifest_sha
  for n,r in m['files'].items():assert V.sha(a.fixtures/n)==r['sha256']
  report.update(status='PASS',negative_cases=negative,after_drop=live,tracked_peak=peak,wall_seconds=time.monotonic()-start,scope='unchanged36 CPU fixture gates/54numeric+63reject,8 depth fixtures exact original bits plus240 synthetic depth exact bits,20 actual-width shapes five alternating timing pairs; no full graph',outputs={p.name:V.sha(p) for p in a.out.iterdir() if p.is_file()})
 except BaseException as e:report['error']=repr(e);raise
 finally:
  try:g.finish()
  finally:(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print('PASS',len(results),negative)
if __name__=='__main__':main()
