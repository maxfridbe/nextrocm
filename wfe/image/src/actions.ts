/** What the buttons do: build the request from the form, send it, keep the results. */
import { api, here } from "./api.js";
import { render, saveForm, state } from "./state.js";
import type { ImageEditBody, ImageGenerateBody } from "./gen/routes.js";
import type { Picture } from "./types.js";
import { fitSize, ratioOf, snap } from "./util.js";

/** A side of the form held to the engine's pixel budget (MOSS-Image: 4 M) against the other side: the largest grid
 *  step that keeps width x height inside it. */
export function capSide(x: number, other: number): number {
  const most = state.info?.max_pixels;
  const g = state.info?.grid ?? 16;
  if (!most || x * other <= most) return x;
  return Math.max(state.info?.min_side ?? g, Math.floor(most / other / g) * g);
}

/** The first time the model is known: the form's empty fields take its defaults. */
export function adoptDefaults(): void {
  const d = state.info?.defaults;
  if (!d) return;
  const f = state.form;
  if (!f.sampler || !(state.info?.samplers ?? []).includes(f.sampler)) f.sampler = d.sampler;
  if (!f.schedule || !(state.info?.schedules ?? []).includes(f.schedule)) f.schedule = d.schedule;
  // LoRAs the server no longer offers drop out; one merged at start shows ticked
  // a size from another model (the form is kept across reloads) is brought inside this one's sides and grid
  const [lo, hi] = [state.info?.min_side ?? 256, state.info?.max_side ?? 4096];
  const g = state.info?.grid ?? 16;
  f.width = Math.min(hi, snap(f.width, g, lo));
  f.height = capSide(Math.min(hi, snap(f.height, g, lo)), f.width);
  const known = new Set((state.info?.loras ?? []).map((l) => l.id));
  for (const id of Object.keys(f.loras)) if (!known.has(id)) delete f.loras[id];
  if (!Object.keys(f.loras).length) {
    for (const l of state.info?.loras ?? []) if (l.loaded != null) f.loras[l.id] = l.loaded;
  }
}

export function setAspect(a: number, b: number): void {
  const d = state.info?.defaults;
  const area = (d?.width ?? 1024) * (d?.height ?? 1024);
  const h = Math.sqrt((area * b) / a);
  const g = state.info?.grid ?? 16;
  const [lo, hi] = [state.info?.min_side ?? 256, state.info?.max_side ?? 4096];
  state.form.width = Math.min(hi, snap((h * a) / b, g, lo));
  state.form.height = capSide(Math.min(hi, snap(h, g, lo)), state.form.width);
  saveForm();
  render();
}

/** The size the form holds as a budget and a ratio: the budget of the preset it matches, else its own area. */
export function sizeNow(): { pixels: number; ratio: number } {
  const f = state.form;
  const s = state.info?.sizes;
  const g = state.info?.grid ?? 16;
  const own = f.width / f.height;
  const ratio = (s?.ratios ?? []).map(ratioOf).find((r) => Math.abs(r - own) < 0.02) ?? own;
  const pixels = (s?.pixels ?? []).find((p) => { const [w, h] = fitSize(p, ratio, g); return w === f.width && h === f.height; });
  return { pixels: pixels ?? f.width * f.height, ratio };
}

/** Whether a size is one the engine draws (its sides within its range). */
export function sizeFits(w: number, h: number): boolean {
  const [lo, hi] = [state.info?.min_side ?? 256, state.info?.max_side ?? 4096];
  return w >= lo && h >= lo && w <= hi && h <= hi && (!state.info?.max_pixels || w * h <= state.info.max_pixels);
}

/** A size from a budget at a ratio (either may be kept from the form's own). */
export function setBudget(pixels: number | null, ratio: number | null): void {
  const now = sizeNow();
  const [w, h] = fitSize(pixels ?? now.pixels, ratio ?? now.ratio, state.info?.grid ?? 16);
  if (!sizeFits(w, h)) return;
  state.form.width = w;
  state.form.height = h;
  saveForm();
  render();
}

export function randomSeed(): void {
  state.form.seed = Math.floor(Math.random() * 1_000_000);
  saveForm();
  render();
}

/** Use a picture's prompt and seed again. */
export function reuse(p: Picture): void {
  state.form.prompt = p.prompt;
  state.form.seed = p.seed;
  saveForm();
  render();
}

/** A file (or blob) as an edit's picture: read as a data: URL, its size from the decoded image. */
export async function addPicture(blob: Blob, name: string): Promise<void> {
  const max = state.info?.max_pictures ?? 8;
  if (state.pictures.length >= max) {
    state.error = `at most ${max} pictures`;
    render();
    return;
  }
  const url = await new Promise<string>((ok, fail) => {
    const r = new FileReader();
    r.onload = () => ok(String(r.result));
    r.onerror = () => fail(r.error);
    r.readAsDataURL(blob);
  });
  const size = await new Promise<[number, number]>((ok) => {
    const im = new Image();
    im.onload = () => ok([im.naturalWidth, im.naturalHeight]);
    im.onerror = () => ok([0, 0]);
    im.src = url;
  });
  state.pictures.push({ url, name, w: size[0], h: size[1] });
  render();
}

/** A picture made here, as an edit's picture. */
export async function editPicture(p: Picture): Promise<void> {
  const b = await (await fetch(p.url)).blob();
  await addPicture(b, p.url.split("/").pop() ?? "picture.png");
  state.panels["create"] = true;
}

export function movePicture(i: number, by: number): void {
  const j = i + by;
  if (j < 0 || j >= state.pictures.length) return;
  const [p] = state.pictures.splice(i, 1);
  if (p) state.pictures.splice(j, 0, p);
  render();
}

export function removePicture(i: number): void {
  state.pictures.splice(i, 1);
  render();
}

export async function refreshHistory(): Promise<void> {
  try {
    state.history = await api.history();
    render();
  } catch {
    /* transient */
  } finally {
    state.historyLoaded = true;
    render();
  }
}

function body(): ImageGenerateBody & Partial<ImageEditBody> {
  const f = state.form;
  const d = state.info?.defaults;
  const editing = state.pictures.length > 0;
  const b: ImageGenerateBody & Partial<ImageEditBody> = {
    model: state.info?.model, prompt: f.prompt.trim(), n: f.n, response_format: "url",
  };
  if (!editing || !state.sizeFromPictures) b.size = `${f.width}x${f.height}`;
  if (editing) b.images = state.pictures.map((p) => p.url);
  // only what the engine takes: a value left in the form from another model is not sent
  if (f.steps && state.info?.steps_free !== false) b.steps = f.steps;
  if (f.seed !== null) b.seed = f.seed;
  if (state.info?.guidance !== false && f.cfg !== null && f.cfg !== d?.cfg) b.cfg = f.cfg;
  if (state.info?.negative !== false && state.info?.guidance !== false && f.negative.trim()) b.negative_prompt = f.negative.trim();
  if (f.shift !== null && state.info?.shift !== false && (state.info?.shift_schedules ?? [f.schedule]).includes(f.schedule)) b.shift = f.shift;
  if (f.sampler && f.sampler !== d?.sampler) b.sampler = f.sampler;
  if (f.schedule && f.schedule !== d?.schedule) b.schedule = f.schedule;
  if (f.rgba && state.info?.transparent !== false) b.background = "transparent";
  const opts = Object.fromEntries(Object.entries(f.options).filter(([k, v]) =>
    v.trim() !== "" && (state.info?.options ?? []).some((o) => o.name === k)));
  if (Object.keys(opts).length) b.options = opts;
  if ((state.info?.loras ?? []).length && state.info?.takes_loras !== false) b.loras = Object.entries(f.loras).map(([name, scale]) => ({ name, scale }));
  return b;
}

export async function generate(): Promise<void> {
  if (state.busy || !state.info || state.info.loading || state.info.vectorize || state.info.removes_background || (state.info.upscale ?? []).length || !state.form.prompt.trim()) return;
  saveForm();
  state.busy = true;
  state.started = performance.now();
  state.error = null;
  state.status = "starting";
  render();
  try {
    const b = body();
    const j = await api.generate(b, state.pictures.length > 0);
    const f = state.form;
    state.results = j.data.filter((d) => d.url).map((d) => ({
      url: d.url as string, prompt: d.revised_prompt, seed: d.seed, steps: j.nextsycl.steps, width: d.width ?? f.width, height: d.height ?? f.height,
      loras: j.nextsycl.loras, seconds: j.nextsycl.seconds, wh: j.nextsycl.wh, created: j.created,
    }));
    const wh = j.nextsycl.wh != null ? ` · ${j.nextsycl.wh.toFixed(2)} Wh` : "";
    state.status = `done in ${j.nextsycl.seconds.toFixed(1)} s${wh}`;
    await refreshHistory();
  } catch (e) {
    state.error = e instanceof Error ? e.message : String(e);
    state.status = "failed";
  }
  state.busy = false;
  state.started = null;
  render();
}

/** Upload one source for an upscale or SVG. Do not accept failed decodes or concurrent additions. */
export async function uploadSource(file: File): Promise<void> {
  if (state.busy || state.uploading) return;
  state.error = null;
  if (!["image/png", "image/jpeg", "image/webp"].includes(file.type)) {
    state.error = "Choose a PNG, JPEG or WebP picture."; render(); return;
  }
  if (file.size > 64 * 1024 * 1024) {
    state.error = "Choose a picture smaller than 64 MiB."; render(); return;
  }
  state.uploading = true; render();
  try {
    const url = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error("Could not read this picture."));
      reader.readAsDataURL(file);
    });
    const [w,h] = await new Promise<[number,number]>((resolve,reject) => {
      const image = new Image();
      image.onload = () => resolve([image.naturalWidth,image.naturalHeight]);
      image.onerror = () => reject(new Error("Could not decode this picture."));
      image.src = url;
    });
    if (!w || !h || w*h > 40_000_000) throw new Error("Choose a picture with at most 40 million pixels.");
    state.pictures = [{url,name:file.name,w,h}];
    state.results = [];
  } catch (e) { state.error = e instanceof Error ? e.message : String(e); }
  finally {state.uploading = false; render();}
}

export function transformVariant() {
  const variants = state.info?.upscale ?? [];
  return variants.find(v => v.variant === state.transform.variant) ?? variants[0];
}

/** Capability-specific requests never inherit drawing prompts, sampling, or stale noise settings. */
export async function transformPicture(): Promise<void> {
  const info = state.info, source = state.pictures[0], f = state.transform;
  if (!info || info.loading || state.busy || state.uploading || !source || state.pictures.length !== 1) return;
  const variant = transformVariant();
  if (!info.vectorize && !info.removes_background && !variant) return;
  state.error = null;
  try {
    if (!info.vectorize && !info.removes_background) {
      if (f.scale !== null && !variant?.scales.includes(f.scale)) throw new Error("Choose a scale supported by this variant.");
      if (f.tile !== null && (!Number.isInteger(f.tile) || f.tile < 32 || f.tile > 4096)) throw new Error("Tile size must be a whole number from 32 to 4096.");
      if ((variant?.noise.length ?? 0) && f.noise !== null && !variant?.noise.includes(f.noise)) throw new Error("Choose a supported denoise level.");
    } else if (info.vectorize) {
      if (f.beams !== null && (!Number.isInteger(f.beams) || f.beams < 1 || f.beams > 8)) throw new Error("Beams must be a whole number from 1 to 8.");
      if (f.maxLength !== null && (!Number.isInteger(f.maxLength) || f.maxLength < 260 || f.maxLength > 8192)) throw new Error("Maximum length must be a whole number from 260 to 8192.");
    }
    state.busy = true; state.started = performance.now(); state.status = "starting"; render();
    if (info.removes_background) {
      const answer = await api.removeBackground({model:info.model,image:source.url,
        response_format:info.saves ? "url" : "b64_json",mask:f.mask,post_process_mask:f.postProcess});
      state.results = answer.data.flatMap(d => {
        const picture: Picture = {kind:"cutout",url:d.url ? here(d.url) : d.b64_json ? `data:image/png;base64,${d.b64_json}` : "",
          prompt:"Background removed",seed:0,steps:0,width:d.width,height:d.height,loras:[],seconds:answer.nextsycl.seconds,created:answer.created};
        const mask = d.mask_url ? here(d.mask_url) : d.mask_b64_json ? `data:image/png;base64,${d.mask_b64_json}` : "";
        return [picture,...(mask ? [{...picture,url:mask,prompt:"Background mask"}] : [])];
      });
      if (!state.results.length || state.results.some(p=>!p.url)) throw new Error("The server returned no output picture.");
      state.status = `done in ${answer.nextsycl.seconds.toFixed(1)} s`;
    } else if (info.vectorize) {
      const answer = await api.vectorize({model:info.model,image:source.url,
        ...(f.beams === null ? {} : {beams:f.beams}), ...(f.maxLength === null ? {} : {max_length:f.maxLength})});
      if (!answer.data.length || answer.data.some(d => typeof d.svg !== "string" || !d.svg.trim())) throw new Error("The server returned no SVG output.");
      state.results = answer.data.map(d => ({kind:"vectorize",url:`data:image/svg+xml;charset=utf-8,${encodeURIComponent(d.svg)}`,
        prompt:`${d.tokens} SVG tokens${d.closed ? "" : " · stopped before the SVG was complete"}`,closed:d.closed,
        seed:0,steps:0,width:source.w,height:source.h,loras:[],seconds:answer.nextsycl.seconds,wh:answer.nextsycl.wh,created:answer.created}));
      state.status = `done in ${answer.nextsycl.seconds.toFixed(1)} s`;
    } else if (variant) {
      const answer = await api.upscale({model:info.model,image:source.url,variant:variant.variant,
        response_format:info.saves ? "url" : "b64_json", ...(f.scale === null ? {} : {scale:f.scale}),
        ...(f.tile === null ? {} : {tile:f.tile}), ...(variant.noise.length && f.noise !== null ? {noise:f.noise} : {})});
      state.results = answer.data.map(d => ({kind:"upscale",url:d.url ? here(d.url) : d.b64_json ? `data:image/png;base64,${d.b64_json}` : "",
        prompt:`${d.variant} · ${d.scale}×${d.noise == null ? "" : ` · noise ${d.noise}`}`,seed:0,steps:0,width:d.width,height:d.height,
        loras:[],seconds:answer.nextsycl.seconds,wh:answer.nextsycl.wh,created:answer.created}));
      if (!state.results.length || state.results.some(p => !p.url)) throw new Error("The server returned no output picture.");
      state.status = `done in ${answer.nextsycl.seconds.toFixed(1)} s`;
    }
    await refreshHistory();
  } catch (e) {state.error = e instanceof Error ? e.message : String(e); state.status = "failed";}
  finally {state.busy = false; state.started = null; render();}
}
