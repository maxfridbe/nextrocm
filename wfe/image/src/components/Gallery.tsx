/** Pictures: the last request's, large; the session's, a grid; one opened over the page. */
import { jsx } from "../jsx.js";
import { editPicture, reuse } from "../actions.js";
import { render, state } from "../state.js";
import type { Picture } from "../types.js";
import { fmtT } from "../util.js";
import { Spinner } from "./Spinner.js";

function meta(p: Picture): string {
  if (p.kind) return `${p.prompt} · ${p.width && p.height ? `${p.width}×${p.height} · ` : ""}${fmtT(p.seconds)}`;
  return `${p.pictures ? `edit of ${p.pictures} · ` : ""}seed ${p.seed} · ${p.steps} steps · ${p.width}×${p.height} · ${fmtT(p.seconds)}` +
    (p.loras.length ? ` · ${p.loras.join(", ")}` : "");
}

function Card(props: { p: Picture; big: boolean }) {
  const p = props.p;
  return (
    <figure class={{ pic: true, big: props.big }}>
      {p.url ? <img attrs={{ src: p.url, alt: p.prompt, loading: "lazy" }} on={{ click: () => { state.zoom = p; render(); } }} /> : <div class="hint">SVG history entry; download the result when it is created.</div>}
      <figcaption>
        <span class="cap" attrs={{ title: p.prompt }}>{props.big ? meta(p) : p.prompt}</span>
        <span class="acts">
          {!p.kind ? <button type="button" class="tbtn" attrs={{ title: "use this prompt and seed again" }} on={{ click: () => reuse(p) }}>reuse</button> : null}
          {state.info?.edits
            ? <button type="button" class="tbtn" attrs={{ title: "add this picture to an edit" }} on={{ click: () => void editPicture(p) }}>edit</button>
            : null}
          {p.url ? <a class="tbtn" attrs={{ href: p.url, download: p.kind === "vectorize" ? "picture.svg" : p.url.startsWith("data:") ? (p.kind === "cutout" ? (p.prompt === "Background mask" ? "mask.png" : "cutout.png") : "upscaled.png") : p.url.split("/").pop() ?? "picture.png" }}>save</a> : null}
        </span>
      </figcaption>
    </figure>
  );
}

export function Results() {
  const one = state.results.length === 1;
  return <div class={{ pics: true, one }}>{state.results.map((p) => <Card p={p} big={true} />)}</div>;
}

export function Gallery() {
  if (!state.historyLoaded) return <Spinner label="Loading what was made…" />;
  if (!state.history.length) return <div class="hint">Nothing made yet. Submit a picture or prompt above.</div>;
  return <div class="pics small">{state.history.map((p) => <Card p={p} big={false} />)}</div>;
}

export function Zoom() {
  const p = state.zoom;
  if (!p) return <div />;
  const close = () => { state.zoom = null; render(); };
  return (
    <div class="zoom" on={{ click: close }}>
      <img attrs={{ src: p.url, alt: p.prompt }} />
      <div class="zcap">{p.prompt}<br /><span class="hint">{meta(p)}</span></div>
    </div>
  );
}
