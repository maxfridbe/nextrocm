/** Picture-to-picture capabilities use the same page, progress, results and history. */
import { jsx } from "../jsx.js";
import { transformPicture, transformVariant, uploadSource } from "../actions.js";
import { render, state } from "../state.js";

export function Transform() {
  const info = state.info, source = state.pictures[0], vector = !!info?.vectorize, cutout = !!info?.removes_background;
  const f = state.transform, variant = transformVariant();
  const locked = state.busy || state.uploading;
  const pick = (files: FileList | null | undefined) => { const file = files?.[0]; if (file) void uploadSource(file); };
  const number = (key: "tile" | "beams" | "maxLength", min: number, max: number, label: string) => <label>{label}
    <input attrs={{type:"number", min, max, step:1, placeholder:"model default", disabled:locked}}
      props={{value:f[key] === null ? "" : String(f[key])}}
      on={{input:(e:Event) => {const value=(e.target as HTMLInputElement).value; f[key]=value === "" ? null : Number(value); render();}}}/>
  </label>;
  return <div class="create">
    <div class="drop" on={{dragover:(e:DragEvent)=>e.preventDefault(), drop:(e:DragEvent)=>{e.preventDefault();if(!locked) pick(e.dataTransfer?.files);}}}>
      <label>Source picture
        <input attrs={{type:"file",accept:"image/png,image/jpeg,image/webp",disabled:locked}}
          on={{change:(e:Event)=>{const input=e.target as HTMLInputElement;pick(input.files);input.value="";}}}/>
      </label>
      <div class="hint">Drop or choose one PNG, JPEG or WebP. {state.uploading ? "Reading picture…" : "Choose another file to replace it."}</div>
      {source ? <figure class="pic"><img attrs={{src:source.url,alt:source.name}}/><figcaption>
        <span>{source.name} · {source.w}×{source.h}</span>
        <button type="button" class="tbtn" attrs={{disabled:locked}} on={{click:()=>{state.pictures=[];render();}}}>remove</button>
      </figcaption></figure> : null}
    </div>
    {cutout ? <div class="row">
      <label class="inline"><input attrs={{type:"checkbox",disabled:locked}} props={{checked:f.mask}}
        on={{change:()=>{f.mask=!f.mask;render();}}}/> Include separate mask</label>
      <label class="inline"><input attrs={{type:"checkbox",disabled:locked}} props={{checked:f.postProcess}}
        on={{change:()=>{f.postProcess=!f.postProcess;render();}}}/> Post-process mask (hard edges)</label>
    </div> : vector ? <div class="row">
      {number("beams",1,8,"Beams")}{number("maxLength",260,8192,"Maximum sequence length")}
      <div class="hint">Blank fields keep the model defaults. The SVG can stop at the length limit; unfinished results are marked.</div>
    </div> : <div>
      <div class="row">
        <label>Variant<select attrs={{disabled:locked}} props={{value:variant?.variant ?? ""}} on={{change:(e:Event)=>{
          f.variant=(e.target as HTMLSelectElement).value;f.scale=null;f.noise=null;render();
        }}}>{(info?.upscale ?? []).map(v=><option attrs={{value:v.variant}}>{v.variant}</option>)}</select></label>
        <label>Scale<select attrs={{disabled:locked}} props={{value:f.scale === null ? "" : String(f.scale)}} on={{change:(e:Event)=>{
          const v=(e.target as HTMLSelectElement).value;f.scale=v === "" ? null : Number(v);render();
        }}}><option attrs={{value:""}}>model default</option>{(variant?.scales ?? []).map(v=><option attrs={{value:String(v)}}>{v}×</option>)}</select></label>
        {(variant?.noise.length ?? 0)>0 ? <label>Denoise<select attrs={{disabled:locked}} props={{value:f.noise === null ? "" : String(f.noise)}} on={{change:(e:Event)=>{
          const v=(e.target as HTMLSelectElement).value;f.noise=v === "" ? null : Number(v);render();
        }}}><option attrs={{value:""}}>model default</option>{variant?.noise.map(v=><option attrs={{value:String(v)}}>{v === -1 ? "none" : String(v)}</option>)}</select></label> : null}
        {number("tile",32,4096,"Tile size (input pixels)")}
      </div>
      <div class="hint">{variant?.about} Blank settings keep the model defaults; a smaller tile uses less working memory.</div>
      {source && f.scale !== null ? <div class="hint">Output: {source.w*f.scale}×{source.h*f.scale}</div> : null}
    </div>}
    <button type="button" attrs={{id:"go",disabled:locked || !info || !!info.loading || !source}}
      on={{click:()=>void transformPicture()}}>{state.busy ? "Working…" : cutout ? "Remove background" : vector ? "Write SVG" : "Upscale"}</button>
  </div>;
}
