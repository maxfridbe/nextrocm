import { jsx } from "../jsx.js";
import { cancelTranscription, transcribe, uploadRecording } from "../recognition.js";
import { render, state } from "../state.js";

export function Transcribe() {
  const f = state.recording, cap = state.info?.recogniser, locked = state.busy || state.uploading;
  const pick = (files: FileList | null | undefined) => { if (files?.[0]) void uploadRecording(files[0]); };
  return <div class="create">
    <div on={{dragover:(e:DragEvent)=>e.preventDefault(),drop:(e:DragEvent)=>{e.preventDefault(); if (!locked) pick(e.dataTransfer?.files);}}}>
      <label>Recording<input attrs={{type:"file",accept:".wav,.mp3,.flac,.ogg,.oga,.m4a",disabled:locked}}
        on={{change:(e:Event)=>{const el=e.target as HTMLInputElement;pick(el.files);el.value="";}}}/></label>
      <div class="hint">Drop or choose one recording, up to 64 MiB. The server decodes the audio.</div>
      {f.audio ? <div><p>{f.name}</p><audio attrs={{controls:true,src:f.audio}}/>
        <button type="button" attrs={{disabled:locked}} on={{click:()=>{f.audio="";f.name="";state.transcript=null;render();}}}>remove</button></div> : null}
    </div>
    <label>Language<select attrs={{disabled:locked}} props={{value:f.language}} on={{change:(e:Event)=>{f.language=(e.target as HTMLSelectElement).value;render();}}}>
      <option attrs={{value:""}}>detect automatically</option>{cap?.languages.map(l=><option attrs={{value:l}}>{l}</option>)}
    </select></label>
    <label>Initial prompt (optional)<textarea attrs={{disabled:locked,placeholder:"Names, spelling or context"}} props={{value:f.prompt}}
      on={{input:(e:Event)=>{f.prompt=(e.target as HTMLTextAreaElement).value;render();}}}/></label>
    {cap?.words ? <label><input attrs={{type:"checkbox",disabled:locked}} props={{checked:f.words}} on={{change:()=>{f.words=!f.words;render();}}}/> Word timestamps</label> : null}
    {cap?.translate ? <label><input attrs={{type:"checkbox",disabled:locked}} props={{checked:f.translate}} on={{change:()=>{f.translate=!f.translate;render();}}}/> Translate into English</label> : null}
    <button type="button" attrs={{id:"go",disabled:locked || !cap || !f.audio}} on={{click:()=>void transcribe()}}>{state.uploading ? "Reading recording…" : state.busy ? "Transcribing…" : "Transcribe"}</button>
    {state.busy ? <button type="button" attrs={{id:"stop"}} on={{click:()=>void cancelTranscription()}}>Cancel</button> : null}
  </div>;
}

export function TranscriptResult() {
  const t=state.transcript;
  if (!t) return null;
  const download=(name:string,type:string,body:string)=><a attrs={{download:name,href:`data:${type};charset=utf-8,${encodeURIComponent(body)}`}}>save {name}</a>;
  const timings=(rows: typeof t.segments)=><div>{rows.map(s=><p><span class="hint">{s.start.toFixed(2)}–{s.end.toFixed(2)} s</span> {s.text}</p>)}</div>;
  return <div>
    <pre>{t.text || "No speech detected."}</pre>
    <div class="hint">{t.language} · {t.duration.toFixed(2)} s of audio · {t.nextsycl.took.toFixed(2)} s to transcribe</div>
    <div class="row">{download("transcript.txt","text/plain",t.text)} {download("transcript.json","application/json",JSON.stringify(t,null,2))}</div>
    {t.segments.length ? <details><summary>Timed segments ({t.segments.length})</summary>{timings(t.segments)}</details> : null}
    {t.words.length ? <details><summary>Word timestamps ({t.words.length})</summary>{timings(t.words)}</details> : null}
    <div class="hint">Download to keep this transcript; it is not stored in song history.</div>
  </div>;
}
