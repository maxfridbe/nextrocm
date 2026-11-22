"""Standard-library utilities for standalone diagnostic validators (no production use)."""
import array,ast,ctypes as C,hashlib,math,struct,sys
from pathlib import Path
P=C.c_void_p;I=C.c_int;L=C.c_int64;Z=C.c_size_t;U=C.c_uint64

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bytes_f32(x):return array.array('f',x).tobytes()
def load(p):
 b=Path(p).read_bytes();assert b[:6]==b'\x93NUMPY' and b[6] in (1,2,3);start=10 if b[6]==1 else 12;n=int.from_bytes(b[8:start],'little');h=ast.literal_eval(b[start:start+n].decode());assert h['descr']=='<f4' and not h['fortran_order'];raw=b[start+n:];assert len(raw)==math.prod(h['shape'])*4
 a=array.array('f');a.frombytes(raw);assert all(math.isfinite(x) for x in a);return list(h['shape']),a

def compare(actual,reference):
 assert len(actual)==len(reference) and all(math.isfinite(x) for x in actual)
 d=[abs(x-y) for x,y in zip(actual,reference)];passed=all(x<=2e-5+2e-5*abs(y) for x,y in zip(d,reference));den=math.fsum(x*x for x in reference);l2=math.sqrt(math.fsum(x*x for x in d)/(den if den else 1))
 return {'passed':passed,'elements':len(actual),'max_abs':max(d),'relative_l2':l2}
class GPU:
 def __init__(self,path):
  assert sys.byteorder=='little' and array.array('f').itemsize==4
  self.lib=C.CDLL(str(Path(path).resolve()));self.buffers=[];self.g=P()
  self.error=self.bind('ns_last_error',[],C.c_char_p);self.open=self.bind('ns_gpu_open',[I,C.POINTER(P)]);self.close=self.bind('ns_gpu_close',[P],None);self.alloc=self.bind('ns_alloc',[P,Z,C.POINTER(P)]);self.free=self.bind('ns_free',[P,P]);self.to=self.bind('ns_copy_to',[P,P,P,Z]);self.frm=self.bind('ns_copy_from',[P,P,P,Z]);self.sync_=self.bind('ns_sync',[P]);self.stats_=self.bind('nr_allocation_stats',[P,C.POINTER(U),C.POINTER(U)]);self.ok(self.open(0,C.byref(self.g)))
 def bind(self,name,args,result=I):
  f=getattr(self.lib,name);f.argtypes=args;f.restype=result;return f
 def ok(self,r):
  if r:raise RuntimeError((self.error() or b'GPU error').decode())
 def sync(self):self.ok(self.sync_(self.g))
 def stats(self):
  self.sync();live,peak=U(),U();self.ok(self.stats_(self.g,C.byref(live),C.byref(peak)));return live.value,peak.value
 def clear(self):
  while self.buffers:
   b=self.buffers.pop();self.ok(self.free(self.g,b.base));b.base=P()
 def finish(self):
  try:self.clear()
  finally:self.close(self.g)
class Buffer:
 def __init__(self,g,x):
  self.g=g;self.initial=array.array('f',x);self.raw=array.array('f',[-918.25]*16)+self.initial+array.array('f',[-918.25]*16);self.base=P();g.ok(g.alloc(g.g,len(self.raw)*4,C.byref(self.base)));g.buffers.append(self);self.reset()
 @property
 def p(self):return self.base.value+64
 def reset(self):self.g.ok(self.g.to(self.g.g,self.base,C.addressof(C.c_char.from_buffer(self.raw)),len(self.raw)*4));self.g.sync()
 def read(self):
  a=array.array('f',[0.])*len(self.raw);self.g.ok(self.g.frm(self.g.g,C.addressof(C.c_char.from_buffer(a)),self.base,len(a)*4));self.g.sync();assert a[:16].tobytes()==self.raw[:16].tobytes() and a[-16:].tobytes()==self.raw[-16:].tobytes(),'redzone';return a[16:-16]
 def unchanged(self):assert self.read().tobytes()==self.initial.tobytes(),'input mutation'
