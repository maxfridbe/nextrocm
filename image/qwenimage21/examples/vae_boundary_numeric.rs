//! Matched raw decoder and clamped public-output gates; original F32 fixture preserved.
use nextrocm_engine_core::{api,Gpu};
use nextrocm_image_qwenimage21::vae::Vae;
use serde_json::{json,Value};
use std::{collections::BTreeMap,fs,path::{Path,PathBuf},process::Command,time::Instant};
type Result<T>=std::result::Result<T,Box<dyn std::error::Error>>;
fn hash(p:&Path)->Result<String>{let o=Command::new("sha256sum").arg("--").arg(p).output()?;if !o.status.success(){return Err("hash failed".into());}let s=String::from_utf8(o.stdout)?;let h=s.split_whitespace().next().ok_or("missing hash")?;if h.len()!=64||!h.bytes().all(|b|b.is_ascii_hexdigit()){return Err("invalid hash".into());}Ok(h.into())}
fn read(p:&Path,n:usize)->Result<Vec<f32>>{let b=fs::read(p)?;if b.len()!=n*4{return Err("fixture byte count".into());}let v:Vec<_>=b.chunks_exact(4).map(|b|f32::from_le_bytes([b[0],b[1],b[2],b[3]])).collect();if v.iter().any(|v|!v.is_finite()){return Err("nonfinite fixture".into());}Ok(v)}
fn main()->Result<()>{
 let args:Vec<_>=std::env::args().skip(1).collect();if args.len()!=5{return Err("usage: vae_boundary_numeric CHECKPOINT FIXTURE_DIR NEW_OUTPUT GPU_INDEX DEVICE_ALLOWANCE_BYTES; NEXTROCM_LIB absolute required".into());}
 let model=fs::canonicalize(&args[0])?;let fixture=fs::canonicalize(&args[1])?;let output=PathBuf::from(&args[2]);let allowance:usize=args[4].parse()?;
 let runtime=PathBuf::from(std::env::var_os("NEXTROCM_LIB").ok_or("NEXTROCM_LIB required")?);if !runtime.is_absolute()||output.exists()||allowance>=8_000_000_000{return Err("runtime/output/strict allowance invalid".into());}
 let executable=fs::canonicalize("/proc/self/exe")?;let manifest_path=fixture.join("manifest.json");let m:Value=serde_json::from_slice(&fs::read(&manifest_path)?)?;
 if m["schema"]!=2 || m["boundary_contract"]!="raw decoder.conv_out plus original decode clamp[-1,1]" || m["cpu_raw_clamp_bitwise_public"]!=true||m["primary_oracle"]!=true||m["precision"]!="original-f32"||m["stage"]!="vae"||m["relative_l2_max"].as_f64()!=Some(0.001){return Err("invalid frozen VAE fixture".into());}
 let h=usize::try_from(m["height"].as_u64().ok_or("height")?)?;let w=usize::try_from(m["width"].as_u64().ok_or("width")?)?;
 nextrocm_image_qwenimage21::vae::validate_dimensions(h,w)?;
 let mut identities=BTreeMap::new();for p in [&model,&runtime,&executable,&manifest_path]{identities.insert(p.clone(),hash(p)?);}
 if identities[&model]!="bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"||m["checkpoint_sha256"]!=identities[&model]{return Err("wrong VAE checkpoint".into());}
 for name in ["config.json","vae_config.json"]{let p=model.parent().ok_or("model parent")?.join(name);if p.exists(){identities.insert(p.clone(),hash(&p)?);}}
 let prov=fixture.join("original-provenance.json");let prov_hash=hash(&prov)?;if m["original_provenance_sha256"]!=prov_hash{return Err("provenance hash".into());}identities.insert(prov,prov_hash);
 let mut arrays=BTreeMap::new();for(name,shape)in[("latents",vec![h*w/256,64]),("expected",vec![h,w,4]),("expected_raw",vec![h,w,4])]{let meta=&m["files"][name];let file=format!("{name}.f32");if meta["file"]!=file||meta["shape"]!=json!(shape){return Err("fixture layout".into());}let p=fixture.join(file);let sha=hash(&p)?;if meta["sha256"]!=sha{return Err("fixture hash".into());}identities.insert(p.clone(),sha);arrays.insert(name,read(&p,shape.iter().product())?);}
 let raw_trace=fixture.join("raw-trace-manifest.json");let rtsha=hash(&raw_trace)?;if m["raw_trace_manifest_sha256"]!=rtsha{return Err("raw trace manifest identity".into());}identities.insert(raw_trace,rtsha);
 if !arrays["expected_raw"].iter().zip(&arrays["expected"]).all(|(&x,&y)|x.clamp(-1.0,1.0).to_bits()==y.to_bits()){return Err("CPU raw clamp differs from original public output".into());}
 fs::create_dir(&output)?;fs::write(output.join("identities-before.json"),serde_json::to_vec_pretty(&identities)?)?;
 let gpu=Gpu::open(args[3].parse()?)?;
 type Stats=unsafe extern "C" fn(*mut std::ffi::c_void,*mut u64,*mut u64)->i32;
 let p=api()?.symbol("nr_allocation_stats");if p.is_null(){return Err("allocation stats absent".into());}
 // SAFETY: exact private ns allocation-stats ABI; runtime remains loaded.
 let stats:Stats=unsafe{std::mem::transmute(p)};
 let stat=||->Result<(u64,u64)>{let(mut live,mut peak)=(0,0);
 // SAFETY: live GPU and valid local out pointers.
 if unsafe{stats(gpu.raw(),&mut live,&mut peak)}!=0{return Err("allocation stats failed".into());}Ok((live,peak))};
 if stat()?.0!=0{return Err("VAE must start as a separate empty device stage".into());}
 let begin=Instant::now();let vae=Vae::load(&model,&gpu,allowance)?;let load=begin.elapsed().as_secs_f64();let resident=vae.resident_bytes;
 let begin=Instant::now();let(got,ow,oh)=vae.decode_f32(&arrays["latents"],h/16,w/16)?;gpu.sync()?;let seconds=begin.elapsed().as_secs_f64();
 if (ow,oh)!=(w,h)||got.len()!=h*w*4||got.iter().any(|v|!v.is_finite()){return Err("native shape/nonfinite output".into());}
 let bytes:Vec<_>=got.iter().flat_map(|v|v.to_le_bytes()).collect();fs::write(output.join("native-rgba-raw.f32"),bytes)?;
 let public:Vec<f32>=got.iter().map(|v|v.clamp(-1.0,1.0)).collect();
 fs::write(output.join("native-rgba-public.f32"),public.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
 let expected=&arrays["expected"];let(mut diff2,mut ref2,mut actual2,mut dot,mut max)=(0f64,0f64,0f64,0f64,0f64);
 for(&a,&b)in public.iter().zip(expected){let(a,b)=(f64::from(a),f64::from(b));let d=a-b;diff2+=d*d;ref2+=b*b;actual2+=a*a;dot+=a*b;max=max.max(d.abs());}
 let rel=diff2.sqrt()/ref2.sqrt().max(1e-30);let rms=(diff2/got.len() as f64).sqrt();let cosine=dot/(ref2*actual2).sqrt().max(1e-30);
 drop(vae);gpu.sync()?;let(live,peak)=stat()?;
 let raw_ref=&arrays["expected_raw"];
 let raw_diff2:f64=got.iter().zip(raw_ref).map(|(&a,&b)|(f64::from(a)-f64::from(b)).powi(2)).sum();
 let raw_ref2:f64=raw_ref.iter().map(|&v|f64::from(v).powi(2)).sum();
 let raw_rel=raw_diff2.sqrt()/raw_ref2.sqrt().max(1e-30);
 let raw_max=got.iter().zip(raw_ref).map(|(&a,&b)|(f64::from(a)-f64::from(b)).abs()).fold(0f64,f64::max);
 let passed=rel<=0.001&&raw_rel<=0.001&&live==0&&peak<allowance as u64&&peak<8_000_000_000;
 for(p,before)in &identities{if hash(p)?!=*before{return Err("identity changed during VAE check".into());}}
 fs::write(output.join("identities-after.json"),serde_json::to_vec_pretty(&identities)?)?;
 let report=json!({"status":if passed{"PASS"}else{"FAIL"},"scope":"same latent input: raw decoder vs original raw tap AND clamped public output vs original decode image; no edit/end-to-end admission","shape":[h,w,4],"public_relative_l2":rel,"raw_relative_l2":raw_rel,"raw_max_absolute":raw_max,"raw_rms":(raw_diff2/got.len() as f64).sqrt(),"rms":rms,"max_absolute":max,"cosine":cosine,"relative_l2_max":0.001,"all_finite":true,"load_seconds":load,"run_seconds":seconds,"resident_weight_bytes":resident,"tracked_peak":peak,"tracked_live_after_drop":live,"device_allowance":allowance,"fixture_manifest":m});
 fs::write(output.join("comparison.json"),serde_json::to_vec_pretty(&report)?)?;println!("{report}");if !passed{return Err("VAE numeric/memory/drop gate failed".into());}Ok(())
}
