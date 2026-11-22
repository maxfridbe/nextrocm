#!/usr/bin/env python3
"""Parent-only typed32 versus frozen logical32 operator proof/timing; full F64 every output, no sampling."""
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
 m=json.loads((a.fixtures/'manifest.json').read_text());assert m['status']=='FULL_ROW_CPU_FIXTURES_PASS' and len(m['cases'])==66
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
 scalar_fn=bind('ni2_quant_matvec_scalar_baseline',args);baseline=bind('ni2_quant_matvec_logical32',args);candidate=bind('ni2_quant_matvec',args);decode=bind('ni2_quant_decode',[P,P,I,I,I,L,P,L])
 g=P();buffers=[];results=[];negative=0;started=time.monotonic();report={'status':'INCOMPLETE','runtime_sha256':a.runtime_sha,'manifest_sha256':a.manifest_sha,'script_sha256':sha(Path(__file__)),'cases':results,'scope':'typed32 and frozen logical32 full wave-order source bits; separate scalar bits; full F64 all outputs; paired timing baseline is logical32, not scalar; no full model admission'}
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
   xr=Path(str(prefix)+'.x.f32').read_bytes();assert len(xr)==cols*4
   xp=b''.join(xr[i:i+4]+struct.pack('<f',19) for i in range(0,len(xr),4)) if small else xr
   w=Buffer(wr);x=Buffer(xp);base=Buffer(struct.pack('<f',37)*(rows*ys));out=Buffer(base.initial);scalar_out=Buffer(base.initial)
   scale=Buffer((a.fixtures/c['scale']).read_bytes()) if c['scale'] else None
   aa=[g,w.p,c['type'],rows,cols,ws,x.p,xs,scale.p if scale else None,out.p,ys]
   ba=aa.copy();ba[9]=base.p;sa=aa.copy();sa[9]=scalar_out.p
   expected=Path(str(prefix)+'.wave.f32').read_bytes();scalar=Path(str(prefix)+'.scalar.f32').read_bytes()
   assert len(expected)==len(scalar)==rows*4
   def full_padded(raw):return b''.join(raw[i:i+4]+struct.pack('<f',37)*(ys-1) for i in range(0,len(raw),4))
   want=full_padded(expected);old=full_padded(scalar)
   ok(scalar_fn(*sa));ok(baseline(*ba));ok(candidate(*aa));assert scalar_out.read()==old and base.read()==want and out.read()==want,(name,'source-order full bits')
   ref=nums(Path(str(prefix)+'.ref.f64').read_bytes(),'<d');bounds=nums(Path(str(prefix)+'.bound.f64').read_bytes(),'<d');values=nums(expected,'<f')
   assert len(ref)==len(bounds)==len(values)==rows
   errors=[abs(float(v)-r) for v,r in zip(values,ref)]
   assert all(math.isfinite(v) for v in ref+bounds+values+errors) and all(b>=0 and e<=b for e,b in zip(errors,bounds)),(name,'full F64')
   if small:
    d=Buffer(bytes(rows*cols*4));ok(decode(g,w.p,c['type'],rows,cols,ws,d.p,cols))
    assert d.read()==Path(str(prefix)+'.decoded.f32').read_bytes(),(name,'full decode bits')
   pairs=[]
   for i in range(5):
    if i%2==0:bt=timed(baseline,ba);ct=timed(candidate,aa)
    else:ct=timed(candidate,aa);bt=timed(baseline,ba)
    ok(scalar_fn(*sa));assert scalar_out.read()==old and base.read()==want and out.read()==want,(name,'repeat bits');pairs.append({'baseline_ms':bt,'candidate_ms':ct})
   assert w.read()==wr and x.read()==xp
   if scale:assert scale.read()==scale.initial
   (a.out/(name+'.f32')).write_bytes(expected)
   results.append({'name':name,'rows':rows,'cols':cols,'type':c['type'],'all_rows_f64':True,'f64_outputs':rows,'max_abs_error':max(errors),'source_wave_bits':True,'baseline_scalar_bits':True,'baseline_logical32_bits':True,'timing_baseline':'frozen logical32','decode_bits':small,'pairs':pairs,'baseline_median_ms':statistics.median(p['baseline_ms'] for p in pairs),'candidate_median_ms':statistics.median(p['candidate_ms'] for p in pairs)})
   print('CASE_PASS',name,results[-1]['baseline_median_ms'],results[-1]['candidate_median_ms'],flush=True);cleanup()
  w=Buffer(bytes(256));x=Buffer(struct.pack('<f',1)*128);y=Buffer(struct.pack('<f',37)*128);s=Buffer(struct.pack('<f',.75))
  aa=[g,w.p,40,1,64,36,x.p,1,s.p,y.p,1]
  bad=[(0,None),(1,None),(2,39),(3,0),(3,201025),(4,63),(5,35),(6,None),(6,x.p+1),(7,0),(8,s.p+1),(8,y.p),(9,None),(9,y.p+1),(9,w.p),(9,x.p),(10,0),(1,(1<<64)-16),(7,1000000001)]
  for at,value in bad:
   before=[b.read() for b in buffers];b=aa.copy();b[at]=value;assert candidate(*b)!=0;ok(sync(g));assert [b.read() for b in buffers]==before;negative+=1
  ok(candidate(*aa));ok(sync(g));cleanup();live,peak=U(),U();ok(stats(g,C.byref(live),C.byref(peak)));assert live.value==0 and peak.value<768*1024**2
  assert sha(a.runtime)==a.runtime_sha and sha(a.fixtures/'manifest.json')==a.manifest_sha
  for n,v in m['files'].items():assert sha(a.fixtures/n)==v['sha256']
  report.update(status='PASS',negative_cases=negative,tracked_peak=peak.value,after_drop=live.value,wall_seconds=time.monotonic()-started,outputs={p.name:sha(p) for p in a.out.glob('*.f32')})
 except BaseException as exc:report['error']=repr(exc);raise
 finally:
  try:cleanup()
  except BaseException as exc:report.update(status='INCOMPLETE',cleanup_error=repr(exc))
  if g.value:close(g)
  (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 assert report['status']=='PASS';print('TYPED32_OPERATOR_PASS',len(results),negative,flush=True)
if __name__=='__main__':main()
