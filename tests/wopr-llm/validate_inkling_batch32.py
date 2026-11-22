#!/usr/bin/env python3
"""Parent-only batch32 versus frozen typed32 operator proof/timing; full F64 every output, no sampling."""
import argparse,ctypes as C,hashlib,json,math,os,statistics,struct,time
from pathlib import Path
P=C.c_void_p;I=C.c_int;L=C.c_int64;U=C.c_uint64;Z=C.c_size_t
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def nums(b,fmt):return [v[0] for v in struct.iter_unpack(fmt,b)]
def main():
 p=argparse.ArgumentParser()
 for n in ('runtime','fixtures','out'):p.add_argument('--'+n,type=Path,required=True)
 for n in ('runtime-sha','manifest-sha'):p.add_argument('--'+n,required=True)
 a=p.parse_args();assert sha(a.runtime)==a.runtime_sha and sha(a.fixtures/'manifest.json')==a.manifest_sha
 m=json.loads((a.fixtures/'manifest.json').read_text());assert m['status']=='FOUR_VECTOR_FULL_ROW_CPU_PASS' and len(m['cases'])==66
 assert [c['name'] for c in m['cases']]==[f'actual-{i:02}' for i in range(26)]+[f'tail-{i:02}' for i in range(40)]
 for n,v in m['files'].items():assert Path(n).name==n and sha(a.fixtures/n)==v['sha256'] and (a.fixtures/n).stat().st_size==v['bytes']
 a.out.mkdir(exist_ok=False);os.environ['NS_PROFILE']='gpu';lib=C.CDLL(str(a.runtime.resolve()))
 def bind(name,args,result=I):f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
 error=bind('ns_last_error',[],C.c_char_p)
 def ok(rc):
  if rc:raise RuntimeError((error() or b'GPU failure').decode())
 op=bind('ns_gpu_open',[I,C.POINTER(P)]);close=bind('ns_gpu_close',[P],None)
 alloc=bind('ns_alloc',[P,Z,C.POINTER(P)]);free=bind('ns_free',[P,P]);to=bind('ns_copy_to',[P,P,P,Z]);frm=bind('ns_copy_from',[P,P,P,Z]);sync=bind('ns_sync',[P]);stats=bind('nr_allocation_stats',[P,C.POINTER(U),C.POINTER(U)])
 stamp=bind('ns_stamp',[P,C.POINTER(L)]);elapsed=bind('ns_elapsed',[P,L,L,C.POINTER(C.c_double)])
 args=[P,P,I,I,I,L,P,L,P,P,L]
 scalar_fn=bind('ni2_quant_matvec_scalar_baseline',args);baseline=bind('ni2_quant_matvec_typed32',args);candidate=bind('ni2_quant_matvec_batch32',[P,P,I,I,I,L,P,L,L,P,P,L,L,I]);decode=bind('ni2_quant_decode',[P,P,I,I,I,L,P,L])
 g=P();buffers=[];results=[];negative=0;started=time.monotonic();report={'status':'INCOMPLETE','runtime_sha256':a.runtime_sha,'manifest_sha256':a.manifest_sha,'script_sha256':sha(Path(__file__)),'cases':results,'scope':'B1..4 every66matrices with4distinctCPUvectors: fullwavebits/fullF64 and scalarcontrols. Timingpairedoperatorinterval: BsequentialPythontypedcalls includeshostsubmissiongaps versus1batch, notpurekernelcost; no modeladmission'}
 class Buffer:
  def __init__(self,b):
   self.initial=bytes(b);self.raw=b'\xa5'*64+self.initial+b'\x5a'*64;self.ptr=P();ok(alloc(g,len(self.raw),C.byref(self.ptr)));buffers.append(self);self.reset()
  @property
  def p(self):return self.ptr.value+64
  def reset(self):
   h=C.create_string_buffer(self.raw);ok(to(g,self.ptr,C.cast(h,P),len(self.raw)));ok(sync(g))
  def read(self):
   h=C.create_string_buffer(len(self.raw));ok(frm(g,C.cast(h,P),self.ptr,len(self.raw)));ok(sync(g));b=h.raw
   assert b[:64]==self.raw[:64] and b[-64:]==self.raw[-64:],'redzone'
   return b[64:-64]
 def cleanup():
  while buffers:
   b=buffers.pop();ok(free(g,b.ptr))
 def timed(fn,aa):
  first,last=L(),L();ok(stamp(g,C.byref(first)));ok(fn(*aa));ok(stamp(g,C.byref(last)));ok(sync(g));ns=C.c_double();ok(elapsed(g,first,last,C.byref(ns)))
  assert math.isfinite(ns.value) and ns.value>=0;return ns.value/1e6
 def timed_many(fn,calls):
  first,last=L(),L();ok(stamp(g,C.byref(first)))
  for aa in calls:ok(fn(*aa))
  ok(stamp(g,C.byref(last)));ok(sync(g));ns=C.c_double();ok(elapsed(g,first,last,C.byref(ns)))
  assert math.isfinite(ns.value) and ns.value>=0;return ns.value/1e6
 try:
  ok(op(0,C.byref(g)))
  initial_live,initial_peak=U(),U();ok(stats(g,C.byref(initial_live),C.byref(initial_peak)))
  assert initial_live.value==0,'validator requires empty device allocation tracker'
  report['initial_live']=initial_live.value
  for c in m['cases']:
   name=c['name'];rows=c['rows'];cols=c['cols'];prefix=a.fixtures/name
   weight=(a.fixtures/c['weight']).read_bytes();assert len(weight)%rows==0;rb=len(weight)//rows
   small=name.startswith('tail-');ws=rb+3 if small else rb;xs=2 if small else 1;ys=3 if small else 1
   wr=b''.join(weight[r*rb:(r+1)*rb]+b'\x33'*3 for r in range(rows)) if small else weight
   allx=Path(str(prefix)+'.x.f32').read_bytes();allwave=Path(str(prefix)+'.wave.f32').read_bytes();allscalar=Path(str(prefix)+'.scalar.f32').read_bytes()
   refs=nums(Path(str(prefix)+'.ref.f64').read_bytes(),'<d');bounds=nums(Path(str(prefix)+'.bound.f64').read_bytes(),'<d')
   assert len(allx)==4*cols*4 and len(allwave)==len(allscalar)==4*rows*4 and len(refs)==len(bounds)==4*rows
   for batch in (1,2,3,4):
    xbs=(cols-1)*xs+1+(7 if small else 0);ybs=(rows-1)*ys+1+(11 if small else 0)
    xp=bytearray(struct.pack('<f',19)*(batch*xbs));want=bytearray(struct.pack('<f',37)*(batch*ybs));old=bytearray(want)
    for token in range(batch):
     for k in range(cols):xp[(token*xbs+k*xs)*4:(token*xbs+k*xs+1)*4]=allx[(token*cols+k)*4:(token*cols+k+1)*4]
     for r in range(rows):
      offset=(token*ybs+r*ys)*4;want[offset:offset+4]=allwave[(token*rows+r)*4:(token*rows+r+1)*4];old[offset:offset+4]=allscalar[(token*rows+r)*4:(token*rows+r+1)*4]
    w=Buffer(wr);x=Buffer(xp);base=Buffer(struct.pack('<f',37)*(batch*ybs));out=Buffer(base.initial);scalar_out=Buffer(base.initial)
    scale=Buffer((a.fixtures/c['scale']).read_bytes()) if c['scale'] else None
    aa=[g,w.p,c['type'],rows,cols,ws,x.p,xs,xbs,scale.p if scale else None,out.p,ys,ybs,batch]
    calls=[[g,w.p,c['type'],rows,cols,ws,x.p+t*xbs*4,xs,scale.p if scale else None,base.p+t*ybs*4,ys] for t in range(batch)]
    scalls=[z.copy() for z in calls]
    for t,z in enumerate(scalls):z[9]=scalar_out.p+t*ybs*4;ok(scalar_fn(*z))
    for z in calls:ok(baseline(*z))
    ok(candidate(*aa));assert scalar_out.read()==old and base.read()==want and out.read()==want,(name,batch,'allsourcebits')
    values=nums(allwave[:batch*rows*4],'<f');ref=refs[:batch*rows];bound=bounds[:batch*rows];errors=[abs(float(v)-r) for v,r in zip(values,ref)]
    assert all(math.isfinite(v) for v in values+ref+bound+errors) and all(b>=0 and e<=b for e,b in zip(errors,bound)),(name,batch,'fullF64')
    if small and batch==1:
     d=Buffer(bytes(rows*cols*4));ok(decode(g,w.p,c['type'],rows,cols,ws,d.p,cols));assert d.read()==Path(str(prefix)+'.decoded.f32').read_bytes()
    pairs=[]
    for rep in range(3):
     if rep%2==0:bt=timed_many(baseline,calls);ct=timed(candidate,aa)
     else:ct=timed(candidate,aa);bt=timed_many(baseline,calls)
     assert base.read()==want and out.read()==want,(name,batch,'repeatbits')
     pairs.append({'baseline_ms':bt,'candidate_ms':ct})
    assert w.read()==wr and x.read()==xp
    if scale:assert scale.read()==scale.initial
    raw=out.read();(a.out/(name+f'-B{batch}.f32')).write_bytes(raw)
    results.append({'name':name,'batch':batch,'rows':rows,'cols':cols,'type':c['type'],'input_stride':xs,'input_batch_stride':xbs,'output_stride':ys,'output_batch_stride':ybs,'all_rows_f64':True,'f64_outputs':batch*rows,'max_abs_error':max(errors),'source_wave_bits':True,'scalar_bits':True,'typed_baseline_bits':True,'pairs':pairs,'baseline_median_ms':statistics.median(v['baseline_ms'] for v in pairs),'candidate_median_ms':statistics.median(v['candidate_ms'] for v in pairs)})
    print('CASE_PASS',name,batch,results[-1]['baseline_median_ms'],results[-1]['candidate_median_ms'],flush=True);cleanup()
  w=Buffer(bytes(512));x=Buffer(struct.pack('<f',1)*512);y=Buffer(struct.pack('<f',37)*512);s=Buffer(struct.pack('<f',.75))
  aa=[g,w.p,40,3,64,36,x.p,1,64,s.p,y.p,1,3,4]
  bad=[(0,None),(1,None),(2,39),(3,0),(3,201025),(4,63),(5,35),(6,None),(6,x.p+1),(7,0),(7,1000000001),(8,0),(8,63),(8,1000000001),(9,s.p+1),(9,y.p),(10,None),(10,y.p+1),(10,w.p),(10,x.p),(11,0),(11,1000000001),(12,0),(12,2),(12,1000000001),(13,0),(13,5),(13,-1),(1,(1<<64)-16),(6,(1<<64)-16),(10,(1<<64)-16)]
  for at,value in bad:
   before=[b.read() for b in buffers];z=aa.copy();z[at]=value;assert candidate(*z)!=0;ok(sync(g));assert [b.read() for b in buffers]==before;negative+=1
  # Combined cross-batch span aliases and oversized combined extents.
  for change in [{10:x.p+64*4},{10:w.p+36},{9:y.p+3*4},{8:400000000},{12:400000000}]:
   before=[b.read() for b in buffers];z=aa.copy()
   for at,value in change.items():z[at]=value
   assert candidate(*z)!=0;ok(sync(g));assert [b.read() for b in buffers]==before;negative+=1
  ok(candidate(*aa));ok(sync(g))
  recovery=bytearray(y.initial)
  for token in range(4):
   for row in range(3):recovery[(token*3+row)*4:(token*3+row+1)*4]=struct.pack('<f',0.0)
  assert y.read()==recovery and w.read()==w.initial and x.read()==x.initial and s.read()==s.initial,'valid recovery full bytes/guards'
  cleanup();live,peak=U(),U();ok(stats(g,C.byref(live),C.byref(peak)));assert live.value==0 and peak.value<768*1024**2
  assert len(results)==264 and sha(a.runtime)==a.runtime_sha and sha(a.fixtures/'manifest.json')==a.manifest_sha
  for n,v in m['files'].items():assert sha(a.fixtures/n)==v['sha256']
  report.update(status='PASS',negative_cases=negative,tracked_peak=peak.value,after_drop=live.value,wall_seconds=time.monotonic()-started,outputs={p.name:sha(p) for p in a.out.glob('*.f32')})
 except BaseException as exc:report['error']=repr(exc);raise
 finally:
  try:cleanup()
  except BaseException as exc:report.update(status='INCOMPLETE',cleanup_error=repr(exc))
  if g.value:close(g)
  (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 assert report['status']=='PASS';print('BATCH32_OPERATOR_PASS',len(results),negative,flush=True)
if __name__=='__main__':main()
