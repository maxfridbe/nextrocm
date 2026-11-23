/** Recognition uses the existing JSON transcription endpoint; recordings stay in memory. */
import { api } from "./api.js";
import { render, state } from "./state.js";

export async function uploadRecording(file: File): Promise<void> {
  if (state.busy || state.uploading) return;
  state.error = null;
  try {
    if (!/\.(wav|mp3|flac|ogg|oga|m4a)$/i.test(file.name)) throw new Error("Choose a WAV, MP3, FLAC, Ogg or M4A recording.");
    if (!file.size || file.size > 64 * 1024 * 1024) throw new Error("Choose a nonempty recording no larger than 64 MiB.");
    state.uploading = true; render();
    const audio = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error("Could not read the recording."));
      reader.onabort = () => reject(new Error("Reading the recording was interrupted."));
      reader.onload = () => typeof reader.result === "string" && reader.result.startsWith("data:")
        ? resolve(reader.result) : reject(new Error("Could not read the recording."));
      reader.readAsDataURL(file);
    });
    state.recording.audio = audio; state.recording.name = file.name;
    state.transcript = null; state.status = "recording ready";
  } catch (e) { state.error = e instanceof Error ? e.message : String(e); }
  finally { state.uploading = false; render(); }
}

export async function transcribe(): Promise<void> {
  const info = state.info, f = state.recording, capability = info?.recogniser;
  if (!capability || state.busy || state.uploading) return;
  state.error = null;
  if (!f.audio) { state.error = "Choose a recording first."; render(); return; }
  if (f.language && !capability.languages.includes(f.language)) { state.error = "Choose a language supported by this model."; render(); return; }
  state.busy = true; state.status = "transcribing"; state.transcript = null; render();
  try {
    const result = await api.transcribe({model: info.model, audio: f.audio,
      ...(f.language ? {language:f.language} : {}), ...(f.prompt.trim() ? {prompt:f.prompt.trim()} : {}),
      words: capability.words && f.words, translate: capability.translate && f.translate});
    if (typeof result.text !== "string" || typeof result.language !== "string" || !Number.isFinite(result.duration) || result.duration < 0 || !Number.isFinite(result.nextsycl?.took) || !Array.isArray(result.segments) || !Array.isArray(result.words) || [...result.segments, ...result.words].some(s => !Number.isFinite(s.start) || !Number.isFinite(s.end) || s.start < 0 || s.end < s.start || typeof s.text !== "string")) throw new Error("The server returned an invalid transcript.");
    state.transcript = result; state.status = "done";
  } catch (e) { state.error = e instanceof Error ? e.message : String(e); state.status = "failed"; }
  finally { state.busy = false; render(); }
}

export async function cancelTranscription(): Promise<void> {
  state.status = "cancelling"; render();
  try { await api.cancel(); }
  catch (e) { state.error = e instanceof Error ? e.message : String(e); render(); }
}
