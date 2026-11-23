// Live compiled UI actions, not a browser rendering test. The server decodes the real WAV.
// --base http://127.0.0.1:18087/ --input actual.wav --expected expected.txt --out NEW_DIRECTORY
import assert from 'node:assert/strict';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve,basename} from 'node:path';
const args=process.argv.slice(2),opts={};
for(let i=0;i<args.length;i+=2){assert.ok(['--base','--input','--expected','--out'].includes(args[i])&&args[i+1]);assert.equal(opts[args[i]],undefined);opts[args[i]]=args[i+1];}
for(const k of ['--base','--input','--expected','--out'])assert.ok(opts[k],k+' required');
const base=new URL(opts['--base']);assert.ok(['http:','https:'].includes(base.protocol)&&base.pathname.endsWith('/'));assert.ok(!base.username&&!base.password);
const out=resolve(opts['--out']);await mkdir(out,{recursive:false});
const input=await readFile(opts['--input']),expected=await readFile(opts['--expected']);
assert.equal(input.toString('ascii',0,4),'RIFF');assert.equal(input.toString('ascii',8,12),'WAVE');
const sha=x=>createHash('sha256').update(x).digest('hex');
const report={status:'INCOMPLETE',scope:'Live compiled UI upload/transcription/result/download actions, Node FileReader adapter; no browser rendering or general ASR accuracy claim',input_sha256:sha(input),expected_sha256:sha(expected),calls:[]};
const originalFetch=globalThis.fetch;
globalThis.fetch=async(url,init={})=>{const target=new URL(String(url),base);assert.equal(target.origin,base.origin);const body=typeof init.body==='string'?JSON.parse(init.body):undefined;const row={path:target.pathname,method:init.method??'GET',...(body?{request:{...body,audio:body.audio?'sha256:'+sha(Buffer.from(body.audio)):undefined}}:{})};report.calls.push(row);const response=await originalFetch(target,{...init,signal:AbortSignal.timeout(180000)});row.status=response.status;return response;};
globalThis.localStorage={getItem:()=>null,setItem:()=>{}};
globalThis.FileReader=class {readAsDataURL(blob){blob.arrayBuffer().then(bytes=>{this.result='data:audio/wav;base64,'+Buffer.from(bytes).toString('base64');this.onload();},()=>this.onerror());}};
const {state,acceptInfo}=await import('../../dist/wfe/audio/src/state.js');
const {api}=await import('../../dist/wfe/audio/src/api.js');
const {uploadRecording,transcribe}=await import('../../dist/wfe/audio/src/recognition.js');
const {App}=await import('../../dist/wfe/audio/src/components/App.js');
const {TranscriptResult}=await import('../../dist/wfe/audio/src/components/Transcribe.js');
const nodes=n=>[n,...(n?.children??[]).flatMap(nodes)].filter(n=>n&&typeof n==='object');
const text=n=>typeof n==='string'?n:(n?.text??'')+(n?.children??[]).map(text).join('');
try {
 acceptInfo(await api.info());report.info=state.info;assert.ok(state.info.recogniser,'recogniser required');assert.match(text(App()),/Transcribe/);
 const blob=new Blob([input],{type:'audio/wav'});Object.defineProperty(blob,'name',{value:basename(opts['--input'])});
 await uploadRecording(blob);assert.equal(state.error,null);assert.equal(Buffer.from(state.recording.audio.split(',')[1],'base64').compare(input),0);
 const start=performance.now();await transcribe();report.action_seconds=(performance.now()-start)/1000;
 assert.equal(state.error,null);assert.equal(state.busy,false);assert.ok(state.transcript);
 report.transcript=state.transcript;
 // Only surrounding whitespace is ignored; spelling, punctuation and internal whitespace must match.
 assert.equal(state.transcript.text.trim(),expected.toString('utf8').trim(),'original expected transcript');
 const links=nodes(TranscriptResult()).filter(n=>n.sel==='a');
 const txt=links.find(n=>n.data?.attrs?.download==='transcript.txt');const json=links.find(n=>n.data?.attrs?.download==='transcript.json');assert.ok(txt&&json);
 const content=decodeURIComponent(txt.data.attrs.href.split(',').slice(1).join(','));assert.equal(content,state.transcript.text);
 assert.deepEqual(JSON.parse(decodeURIComponent(json.data.attrs.href.split(',').slice(1).join(','))),state.transcript);
 await writeFile(out+'/transcript.txt',content);await writeFile(out+'/transcript.json',JSON.stringify(state.transcript,null,2)+'\n');
 assert.equal(sha(await readFile(opts['--input'])),report.input_sha256);assert.equal(sha(await readFile(opts['--expected'])),report.expected_sha256);
 report.status='PASS';
} catch(error) {report.status='FAIL';report.error=String(error);process.exitCode=1;}
await writeFile(out+'/report.json',JSON.stringify(report,null,2)+'\n');console.log(report.status);
