// Live application-action test, not browser rendering or a numerical oracle.
// Build image first. Run under the parent's GPU/process watchdog against an isolated server.
// node --experimental-default-type=module wfe/test/image-upscale-live.mjs --base http://127.0.0.1:18086/ --input picture.png --out NEW_DIR [--variant NAME] [--scale 4] [--tile 200]
import assert from 'node:assert/strict';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {resolve} from 'node:path';
const argv=process.argv.slice(2), opts={};
for(let i=0;i<argv.length;i+=2){assert.ok(['--base','--input','--out','--variant','--scale','--tile'].includes(argv[i])&&argv[i+1],'unknown option or missing value');assert.equal(opts[argv[i]],undefined,'duplicate option');opts[argv[i]]=argv[i+1];}
assert.ok(opts['--base']&&opts['--input']&&opts['--out'],'--base, --input and --out required');
const base=new URL(opts['--base']);assert.ok(['http:','https:'].includes(base.protocol));assert.ok(base.pathname.endsWith('/'),'base must end in /');assert.ok(!base.username&&!base.password,'credentials must not be embedded in base URL');
const out=resolve(opts['--out']);await mkdir(out,{recursive:false});
const hash=b=>createHash('sha256').update(b).digest('hex');
const pngSize=bytes=>{assert.ok(bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])),'PNG required');assert.equal(bytes.toString('ascii',12,16),'IHDR');return [bytes.readUInt32BE(16),bytes.readUInt32BE(20)];};
const inputPath=resolve(opts['--input']), input=await readFile(inputPath), dimensions=pngSize(input);
const report={status:'INCOMPLETE',scope:'Compiled UI upload/actions/API/result/history against a live server; Node upload adapter, no browser rendering or pixel-accuracy claim',base:base.href,input_sha256:hash(input),input_dimensions:dimensions,calls:[]};
const fetchNative=globalThis.fetch;
globalThis.fetch=async(url,init={})=>{
 const target=new URL(String(url),base);assert.equal(target.origin,base.origin,'unexpected cross-origin request');
 const body=typeof init.body==='string'?JSON.parse(init.body):undefined;
 const row={path:target.pathname,method:init.method??'GET',...(body?{request:{...body,image:body.image?`sha256:${hash(Buffer.from(body.image))}`:undefined}}:{})};report.calls.push(row);
 const response=await fetchNative(target,{...init,signal:AbortSignal.timeout(180000)});row.status=response.status;return response;
};
globalThis.localStorage={getItem:()=>null,setItem:()=>{}};
// Node has no browser FileReader/Image. Read the actual bytes and use their PNG header;
// the real server subsequently decodes the uploaded PNG. No fabricated source dimensions.
globalThis.FileReader=class {readAsDataURL(blob){blob.arrayBuffer().then(bytes=>{this.result=`data:${blob.type};base64,${Buffer.from(bytes).toString('base64')}`;this.onload();},error=>{this.error=error;this.onerror();});}};
globalThis.Image=class {set src(value){try{const bytes=Buffer.from(value.split(',')[1],'base64');[this.naturalWidth,this.naturalHeight]=pngSize(bytes);queueMicrotask(()=>this.onload());}catch{queueMicrotask(()=>this.onerror());}}};
const {state}=await import('../../dist/wfe/image/src/state.js');
const {api,here}=await import('../../dist/wfe/image/src/api.js');
const {uploadSource,transformPicture,refreshHistory}=await import('../../dist/wfe/image/src/actions.js');
const {App}=await import('../../dist/wfe/image/src/components/App.js');
const text=n=>typeof n==='string'?n:(n?.text??'')+(n?.children??[]).map(text).join('');
try {
 state.info=await api.info();report.info=state.info;
 assert.ok(!state.info.loading&&state.info.upscale?.length,'loaded upscaler required');
 assert.match(text(App()),/Upscale/);assert.doesNotMatch(text(App()),/this page draws them/);
 await refreshHistory();report.history_before=state.history.length;
 await uploadSource(new File([input],'source.png',{type:'image/png'}));assert.equal(state.error,null);assert.equal(state.pictures.length,1);
 assert.deepEqual([state.pictures[0].w,state.pictures[0].h],dimensions);
 const variant=state.info.upscale.find(v=>v.variant===opts['--variant'])??(opts['--variant']?undefined:state.info.upscale[0]);assert.ok(variant,'requested variant not advertised');
 state.transform.variant=variant.variant;
 if(opts['--scale']){state.transform.scale=Number(opts['--scale']);assert.ok(variant.scales.includes(state.transform.scale));}
 if(opts['--tile']){state.transform.tile=Number(opts['--tile']);assert.ok(Number.isInteger(state.transform.tile)&&state.transform.tile>=32&&state.transform.tile<=4096);}
 const start=performance.now();await transformPicture();report.action_seconds=(performance.now()-start)/1000;
 assert.equal(state.error,null);assert.equal(state.busy,false);assert.equal(state.started,null);assert.equal(state.results.length,1);
 const result=state.results[0];report.result={...result,url:result.url.startsWith('data:')?'inline PNG':result.url};
 const bytes=result.url.startsWith('data:')?Buffer.from(result.url.split(',')[1],'base64'):Buffer.from(await (await fetch(new URL(here(result.url),base))).arrayBuffer());
 const outputSize=pngSize(bytes);assert.deepEqual(outputSize,[result.width,result.height]);
 if(state.transform.scale!==null)assert.deepEqual(outputSize,dimensions.map(x=>x*state.transform.scale));
 report.output_sha256=hash(bytes);report.output_dimensions=outputSize;await writeFile(resolve(out,'output.png'),bytes,{flag:'wx'});
 if(state.info.saves){assert.ok(state.history.some(p=>p.kind==='upscale'&&p.url===result.url),'saved output missing from actual history');}
 report.history_after=state.history.length;
 assert.match(text(App()),/Result/);assert.match(text(App()),/save/);
 assert.equal(hash(await readFile(inputPath)),report.input_sha256,'input changed during request');
 report.status='PASS';
} catch(error){report.status='FAIL';report.error=error.stack??String(error);process.exitCode=1;}
finally{await writeFile(resolve(out,'report.json'),JSON.stringify(report,null,2)+'\n',{flag:'wx'});}
console.log(JSON.stringify({status:report.status,output:out,seconds:report.action_seconds,error:report.error}));
