// Live compiled UI actions, not a browser-rendering or voice-similarity test.
// --base http://host:8000/open/voxcpm2-int8/ --voice skippy --text "Hello." --seconds 12 --out NEW_DIR
import assert from 'node:assert/strict';
import {mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
const opts={},args=process.argv.slice(2);
for(let i=0;i<args.length;i+=2){assert.ok(['--base','--voice','--text','--seconds','--out'].includes(args[i])&&args[i+1]);assert.equal(opts[args[i]],undefined);opts[args[i]]=args[i+1];}
for(const key of ['--base','--voice','--text','--seconds','--out'])assert.ok(opts[key],key+' required');
const base=new URL(opts['--base']),seconds=Number(opts['--seconds']);
assert.ok(['http:','https:'].includes(base.protocol)&&base.pathname.endsWith('/')&&!base.username&&!base.password);
assert.ok(Number.isFinite(seconds)&&seconds>0&&seconds<=60);
await mkdir(opts['--out'],{recursive:false});
const report={status:'INCOMPLETE',scope:'Live compiled UI named-voice generation, WAV download and saved history; no browser rendering or voice-similarity claim',calls:[]};
const originalFetch=globalThis.fetch;
globalThis.fetch=async(url,init={})=>{const target=new URL(String(url),base);assert.equal(target.origin,base.origin);const row={path:target.pathname,method:init.method??'GET'};if(typeof init.body==='string')row.request=JSON.parse(init.body);report.calls.push(row);const response=await originalFetch(target,{...init,redirect:'error',signal:AbortSignal.timeout(600000)});row.status=response.status;return response;};
globalThis.localStorage={getItem:()=>null,setItem:()=>{}};
const {state,acceptInfo}=await import('../../dist/wfe/audio/src/state.js');
const {api}=await import('../../dist/wfe/audio/src/api.js');
const {refreshVoices,generate}=await import('../../dist/wfe/audio/src/actions.js');
try {
 acceptInfo(await api.info());assert.ok(state.info.speech?.clone);report.model=state.info.model;
 await refreshVoices();assert.ok(state.voices.some(v=>v.name===opts['--voice']));report.voice_names=state.voices.map(v=>v.name);
 Object.assign(state.form,{text:opts['--text'],voice:opts['--voice'],seconds,seed:42});
 const start=performance.now();await generate();report.action_seconds=(performance.now()-start)/1000;
 assert.equal(state.error,null);assert.equal(state.busy,false);assert.ok(state.result?.url);assert.equal(state.result.voice,opts['--voice']);
 assert.ok(state.result.seconds>0&&state.history.some(h=>h.url===state.result.url));report.result=state.result;
 const response=await fetch(state.result.url);assert.equal(response.status,200);const wav=Buffer.from(await response.arrayBuffer());
 assert.equal(wav.toString('ascii',0,4),'RIFF');assert.equal(wav.toString('ascii',8,12),'WAVE');assert.ok(wav.length>44);
 assert.equal(wav.readUInt32LE(4)+8,wav.length);
 let fmt,data,at=12;
 while(at<wav.length){assert.ok(at+8<=wav.length);const tag=wav.toString('ascii',at,at+4),size=wav.readUInt32LE(at+4),end=at+8+size;assert.ok(end<=wav.length);if(tag==='fmt ')fmt=wav.subarray(at+8,end);if(tag==='data')data=wav.subarray(at+8,end);at=end+(size%2);}
 assert.equal(at,wav.length);assert.ok(fmt?.length>=16&&data?.length>0);
 const format=fmt.readUInt16LE(0),channels=fmt.readUInt16LE(2),rate=fmt.readUInt32LE(4),align=fmt.readUInt16LE(12),bits=fmt.readUInt16LE(14);
 assert.equal(format,1);assert.equal(bits,16);assert.ok(channels>0&&rate>0);assert.equal(align,channels*2);assert.equal(fmt.readUInt32LE(8),rate*align);assert.equal(data.length%align,0);
 const duration=data.length/align/rate;assert.ok(Math.abs(duration-state.result.seconds)<=.001+1/rate);
 let power=0;for(let i=0;i<data.length;i+=2)power+=(data.readInt16LE(i)/32768)**2;const rms=Math.sqrt(power/(data.length/2));assert.ok(rms>0,'non-silent PCM');
 report.wav={format,channels,rate,bits,frames:data.length/align,duration,rms};
 const submitted=report.calls.find(c=>c.method==='POST'&&c.path.endsWith('/v1/audio/speech'))?.request;
 assert.ok(submitted);assert.equal(submitted.voice,opts['--voice']);assert.equal(submitted.input,opts['--text']);assert.equal(submitted.model,report.model);assert.equal(submitted.seed,42);
 report.wav_sha256=createHash('sha256').update(wav).digest('hex');report.wav_bytes=wav.length;
 await writeFile(opts['--out']+'/speech.wav',wav);report.status='PASS';
} catch(error){report.status='FAIL';report.error=String(error);process.exitCode=1;}
await writeFile(opts['--out']+'/report.json',JSON.stringify(report,null,2)+'\n');console.log(report.status);
