//! One profiled velocity and same-input VAE; frozen diagnostic outputs, not numerical admission.
use nextrocm_engine_core::{api,DevBuf,Gpu};
use nextrocm_gguf::Gguf;
use nextrocm_image_qwenimage21::{denoiser::Denoiser,ffi::{self,PackedKernel,AttentionKernel},profiler,vae::Vae,sched};
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},process::Command,time::Instant,io::Write};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
fn hash(p: &Path) -> Result<String> {
    let o = Command::new("sha256sum").arg("--").arg(p).output()?;
    if !o.status.success() {
        return Err("hash command failed".into());
    }
    let s = String::from_utf8(o.stdout)?;
    let h = s.split_whitespace().next().ok_or("missing hash")?;
    if h.len() != 64 || !h.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid hash".into());
    }
    Ok(h.into())
}
fn npy(p: &Path) -> Result<(Vec<usize>, Vec<f32>)> {
    let b = fs::read(p)?;
    if b.len() < 12 || &b[..6] != b"\x93NUMPY" {
        return Err("bad NPY".into());
    }
    let (base, len) = match b[6] {
        1 => (10, u16::from_le_bytes([b[8], b[9]]) as usize),
        2 => (12, u32::from_le_bytes(b[8..12].try_into()?) as usize),
        _ => return Err("NPY version".into()),
    };
    let h = std::str::from_utf8(b.get(base..base + len).ok_or("NPY header")?)?;
    if !h.contains("'<f4'") || !h.contains("'fortran_order': False") {
        return Err("NPY dtype/layout".into());
    }
    let shape = h
        .split("'shape':")
        .nth(1)
        .and_then(|x| x.split('(').nth(1))
        .and_then(|x| x.split(')').next())
        .ok_or("NPY shape")?
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let count = shape
        .iter()
        .try_fold(1usize, |n, &v| n.checked_mul(v))
        .ok_or("NPY overflow")?;
    if count > 16_000_000 || b.len() != base + len + count * 4 {
        return Err("NPY length".into());
    }
    let v: Vec<_> = b[base + len..]
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    if v.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite reference".into());
    }
    Ok((shape, v))
}
const FIXTURES:&str=r#"{"/tmp/qi21-render1024-batched-result-v1/report.json": "2488aab2f6c647ed7fa8dd7433297f9952b5f4e4509e17f2962f90a3a8852bca", "/tmp/qi21-render1024-batched-result-v1/identities-before.json": "4b328d25edeed56e36c4b4cf443667207d61928ea1c509001f6b73c04f893069", "/tmp/qi21-render1024-batched-result-v1/identities-after.json": "4b328d25edeed56e36c4b4cf443667207d61928ea1c509001f6b73c04f893069", "/tmp/qi21-render1024-batched-result-v1/te/embeds.f32": "1bc016991ab32ec19d4037047d538c4c07e03d3ed7a8a2dfc63371db2896d448", "/tmp/qi21-render1024-batched-result-v1/dit/v0.f32": "f2dcfca32a1b350071d044dac1bb45841b87fe5dc48c3e0630c1283a6cf35e5c", "/tmp/qi21-render1024-batched-result-v1/dit/latents_7.f32": "47389c4ede78839fcd1de509860f6aafbe3acc374b76cbdfcab440fa3b3f5c15", "/tmp/qi21-render1024-batched-result-v1/vae/image_raw.f32": "cde2c1033932c82c898e414b4a8c186f3eaf312133f402bc0815f2a6c9e243e1", "/tmp/qi21-inputs-1024-v1/manifest.json": "7439a8ebc47cad9e0f34c71b9e40e8bc6a3e372547b65af82874e1b6f28a1c4b", "/tmp/qi21-inputs-1024-v1/noise.npy": "09d60b585c68665d5c538a08d4bbbc29cc1cb11d33e33932c9f5c9e7f010b08c", "/tmp/qi21-inputs-1024-v1/timesteps.npy": "733902ffb944c9bbadbc4ee45b1ddd87d721d3b884bb55af5ccdcca796a7b210", "/tmp/qi21-inputs-1024-v1/sigmas.npy": "11cfe7e03f8d949603b2771405b979c3ecf1c61a61cbb41c8470320c6b625ece"}"#;

fn raw(p:&Path,n:usize)->Result<Vec<f32>> {let b=fs::read(p)?;if b.len()!=n*4{return Err("raw shape".into())}let x:Vec<_>=b.chunks_exact(4).map(|v|f32::from_le_bytes(v.try_into().unwrap())).collect();if x.iter().any(|v|!v.is_finite()){return Err("raw finite".into())}Ok(x)}
fn save_exact(p:&Path,x:&[f32],expected:&Path)->Result<()> {if x.iter().any(|v|!v.is_finite()){return Err("output finite".into())}let b:Vec<_>=x.iter().flat_map(|v|v.to_le_bytes()).collect();fs::write(p,&b)?;if b!=fs::read(expected)?{return Err(format!("full output differs {}",expected.display()).into())}Ok(())}
fn coverage(p:&Value,labels:&[&str])->Result<()> {let ops=p["operations"].as_array().ok_or("missing operations")?;let count:u64=ops.iter().map(|x|x["count"].as_u64().unwrap_or(0)).sum();if count==0||p["operation_count"]!=count{return Err("empty or inconsistent profile operation count".into())}for label in labels {if !ops.iter().any(|x|x["operation"]==*label && x["count"].as_u64().unwrap_or(0)>0){return Err(format!("missing GPU category {label}").into())}}Ok(())}
fn main()->Result<()> {
 let args:Vec<_>=std::env::args().skip(1).collect();if args.len()!=3{return Err("profile_1024 ASSET_DIR NEW_OUTPUT GPU; NS_PROFILE=gpu NEXTROCM_LIB=absolute frozen DSO".into())}
 let assets=PathBuf::from(&args[0]);let out=PathBuf::from(&args[1]);if out.exists(){return Err("new output required".into())}
 let runtime=PathBuf::from(std::env::var_os("NEXTROCM_LIB").ok_or("missing runtime")?);if !runtime.is_absolute()||hash(&runtime)?!="0ac4860f02713e3c0d75ccb5a1e6228bbc4493bcf15156d433c14e819d6d0c52"||std::env::var("NS_PROFILE").as_deref()!=Ok("gpu"){return Err("frozen runtime / GPU events required".into())}
 let mut ids:BTreeMap<PathBuf,String>=serde_json::from_str(FIXTURES)?;for(p,h)in &ids{if hash(p)?!=*h{return Err(format!("fixture pin {}",p.display()).into())}}
 let gguf=assets.join("qwen_image_2.1_turbo_Q4_K_M.gguf");let vaefile=assets.join("qwen_image_2.1_vae_bf16.safetensors");
 for(p,h)in [(&gguf,"0f5a11ebbaf3110a16120311454eaac35db13798e0102339e4f2b8fd5be6ba82"),(&vaefile,"bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9")]{if hash(p)?!=h{return Err("asset pin".into())}ids.insert(p.clone(),h.into());}
 for p in [runtime,fs::canonicalize("/proc/self/exe")?]{ids.insert(p.clone(),hash(&p)?);}
 let root=Path::new("/tmp/qi21-render1024-batched-result-v1");let fixture=Path::new("/tmp/qi21-inputs-1024-v1");
 let embeds=raw(&root.join("te/embeds.f32"),18*4096)?;let latents=raw(&root.join("dit/latents_7.f32"),4096*64)?;
 let(shape,noise)=npy(&fixture.join("noise.npy"))?;if shape!=[4096,64]{return Err("noise shape".into())}let(shape,times)=npy(&fixture.join("timesteps.npy"))?;if shape!=[8]{return Err("time shape".into())}let(shape,sigmas)=npy(&fixture.join("sigmas.npy"))?;if shape!=[9]||sigmas!=sched::turbo_sigmas(){return Err("official schedule".into())}
 let gpu=Gpu::open(args[2].parse()?)?;type Stats=unsafe extern "C" fn(*mut std::ffi::c_void,*mut u64,*mut u64)->i32;let sym=api()?.symbol("nr_allocation_stats");if sym.is_null(){return Err("stats ABI".into())}
 // SAFETY: exact frozen allocation-statistics ABI; loaded runtime remains live.
 let stats:Stats=unsafe{std::mem::transmute(sym)};
 let memory=||->Result<(u64,u64)>{gpu.sync()?;let(mut l,mut p)=(0,0);ffi::check(unsafe{stats(gpu.raw(),&mut l,&mut p)},"stats")?;if p>=7_500_000_000{return Err("tracked cap".into())}Ok((l,p))};
 if memory()?.0!=0{return Err("initial allocations".into())}fs::create_dir(&out)?;fs::write(out.join("identities-before.json"),serde_json::to_vec_pretty(&ids)?)?;
 let result=(||->Result<Value>{
  let gg=Gguf::open(&gguf)?;let t=Instant::now();let dit=Denoiser::load_with_attention(&gg,&gpu,4096,18,PackedKernel::Register64,0,1024*1024*1024,AttentionKernel::Batched1024)?;gpu.sync()?;let load=t.elapsed().as_secs_f64();if memory()?.0!=dit.plan().resident{return Err("DiT resident plan".into())}
  let t=Instant::now();let pre=dit.prefix(&embeds,false)?;gpu.sync()?;let prefix=t.elapsed().as_secs_f64();let x=DevBuf::from_f32(&gpu,&noise)?;
  profiler::begin(&gpu)?;let t=Instant::now();let v=dit.velocity(&pre,&x,(64,64),times[0],None)?;gpu.sync()?;let seconds=t.elapsed().as_secs_f64();let profile=profiler::finish()?;
  if profile["operation_count"]!=424||profile["dropped_spans"]!=0{return Err("DiT span coverage".into())}
  coverage(&profile,&["block.qkv","block.out","block.gate_up","block.down","block.attention","block.norm1","block.norm2"])?;
  for (labels,want) in [(&["block.norm1","block.norm2","block.qkv","block.prefix_copy","block.rms_q","block.rms_k","block.attention","block.out","block.gate1","block.gate2","block.gate_up","block.swiglu","block.down"][..],32u64),(&["aux.timestep1","aux.timestep2","aux.modulation","aux.norm_scale","aux.img_in_f32","output.norm","output.to_float","aux.proj_out"][..],1u64)] {for label in labels {let count:u64=profile["operations"].as_array().ok_or("operations")?.iter().filter(|x|x["operation"]==*label).map(|x|x["count"].as_u64().unwrap_or(0)).sum();if count!=want{return Err(format!("DiT coverage {label} {count} != {want}").into())}}}

  let t=Instant::now();let actual=v.to_f32()?;let readback=t.elapsed().as_secs_f64();save_exact(&out.join("v0.f32"),&actual,&root.join("dit/v0.f32"))?;
  let mem=memory()?;if mem.1>dit.plan().tracked{return Err("DiT aggregate plan".into())}fs::write(out.join("dit-operators.json"),serde_json::to_vec_pretty(&profile)?)?;
  drop((v,x,pre,dit,gg));if memory()?.0!=0{return Err("DiT drop leak".into())}
  let dit_report=json!({"load_seconds":load,"prefix_seconds":prefix,"velocity_seconds":seconds,"readback_seconds":readback,"full_output_bitwise":true,"allocations":mem});fs::write(out.join("dit-report.json"),serde_json::to_vec_pretty(&dit_report)?)?;println!("DiT velocity {seconds}s complete bitwise");std::io::stdout().flush()?;
  let t=Instant::now();let vae=Vae::load(&vaefile,&gpu,7_500_000_000)?;gpu.sync()?;let load=t.elapsed().as_secs_f64();profiler::begin(&gpu)?;let t=Instant::now();let(image,w,h)=vae.decode_f32(&latents,64,64)?;gpu.sync()?;let decode=t.elapsed().as_secs_f64();let profile=profiler::finish()?;
  if (w,h)!=(1024,1024)||image.len()!=1024*1024*4||profile["dropped_spans"]!=0{return Err("VAE shape/profile".into())}coverage(&profile,&["vae.conv","vae.linear","vae.rms","vae.residual","vae.silu","vae.upsample","vae.upsample_shortcut","vae.softmax","vae.capture_cast","vae.latent_cast","vae.rgba"])?;
  let host=profile["host_operations"].as_array().ok_or("missing host readback categories")?;for label in ["vae.raw_readback","vae.rgba_readback"]{if !host.iter().any(|x|x["operation"]==label&&x["count"].as_u64().unwrap_or(0)>0){return Err("missing VAE readback coverage".into())}}
  save_exact(&out.join("image_raw.f32"),&image,&root.join("vae/image_raw.f32"))?;fs::write(out.join("vae-operators.json"),serde_json::to_vec_pretty(&profile)?)?;
  drop(vae);let after=memory()?;if after.0!=0{return Err("VAE drop leak".into())}
  Ok(json!({"status":"PASS","scope":"profiling reproduces frozen diagnostic outputs; original1024 numerical FAIL unchanged; no quality admission","dit":dit_report,"vae":{"load_seconds":load,"decode_including_readback_seconds":decode,"full_output_bitwise":true},"after_drop":after,"features":["operator-profile"]}))
 })();if result.is_err(){profiler::abort();}let after:BTreeMap<_,_>=ids.keys().map(|p|Ok((p.clone(),hash(p)?))).collect::<Result<_>>()?;let unchanged=after==ids;fs::write(out.join("identities-after.json"),serde_json::to_vec_pretty(&after)?)?;
 let report=match &result{Ok(v)if unchanged=>v.clone(),_=>json!({"status":"FAIL","error":result.as_ref().err().map(ToString::to_string),"identities_unchanged":unchanged,"after_unwind":memory().ok()})};fs::write(out.join("report.json"),serde_json::to_vec_pretty(&report)?)?;result?;if !unchanged{return Err("identity changed".into())}Ok(())
}
