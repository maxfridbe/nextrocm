# Using and testing the image and audio UIs

The image page selects its controls from the loaded model's capabilities: image generation, upscaling, background removal or SVG conversion. A model must be installed and configured before starting its server; building the page does not download weights.

## Build and open

From the repository root, build the frontends **before** rebuilding the CLI so the executable embeds the new pages:

```sh
bash wfe/build.sh
cargo build --release -p nextrocm
NS_DIST="$PWD/dist" ./target/release/nextrocm image serve real-esrgan \
  --here --wfe --host 127.0.0.1 --port 8086 --out "$PWD/image-results"
```

This example assumes an installed `real-esrgan` registry entry and a compatible native runtime. Follow the [runtime architecture instructions](../tooling/ARCHITECTURE-MIGRATION.md) for the GPU libraries. Open `http://127.0.0.1:8086/`. Replace the model ID to use another installed image model. The `--here` option runs in the current environment, where ROCm and the runtime must be available.

For frontend development without rebuilding the executable, build the pages and set `NS_WFE_FROM_DISK=1` as well as `NS_DIST="$PWD/dist"` on the server command. Otherwise the embedded pages take precedence over files on disk. Refresh the browser after rebuilding.

## Picture workflows

**Upscale:** choose or drop one PNG, JPEG or WebP. The page displays its name and dimensions. Select an advertised variant and scale; denoising appears only for variants that support it. Leave fields at “model default” to retain server defaults. Tile size is measured in input pixels; smaller tiles reduce working memory. Select **Upscale**, then inspect the result and use **save** to download it. Choosing another source replaces the previous one. Controls are disabled while an operation is running.

**Remove background:** load a background-removal model, upload a picture and select **Remove background**. Optionally request a separate mask or mask post-processing. The result and optional mask have separate previews and download links.

**Write SVG:** load a vectorization model and upload a picture. Optional beam count and maximum sequence length retain model defaults when blank. Select **Write SVG** and download the SVG from the result. An output that stops before completion is labelled unfinished. SVG previews use an image element, not injected SVG markup. UI support does not imply that every model setting produces useful SVG output.

When the server saves image outputs, they appear in history and can be downloaded again. Inline-only outputs can be downloaded from the current result. SVG history currently contains metadata only: it does not retain the SVG for later download. Generation models continue to use their existing prompt and generation controls.

## Repeatable checks

The following checks need no model or GPU:

```sh
bash wfe/build.sh image
node --experimental-default-type=module wfe/test/image-transforms.mjs
```

They cover upload validation, variant/scale/noise/tile selection, request bodies, URL and inline results, failure and retry behavior, duplicate prevention, SVG handling, background removal with masks, empty-response rejection, history and the existing generation flow. They use mocked browser and network interfaces; they do not establish browser rendering or model accuracy.

Against a separately started, isolated Real-ESRGAN server, run the compiled application actions with a real PNG:

```sh
node --experimental-default-type=module wfe/test/image-upscale-live.mjs \
  --base http://127.0.0.1:8086/ --input /absolute/path/input.png \
  --out /absolute/path/new-ui-check-directory \
  --variant realesrgan-x4plus --scale 4 --tile 200
```

The output directory must not exist. This check uploads real bytes through the UI action, calls the live API, downloads the output, checks its dimensions and saved-history entry, and writes `report.json` and `output.png`. It uses Node upload adapters, so it is not a browser rendering test. Compare `output.png` against an independently produced reference to establish numerical accuracy separately.

At this revision the interaction checks pass. The new runtime's strict-silo native RGB upscale check also passed two runs against its reference (61.64 dB PSNR, maximum channel difference 1, no retained tracked allocations). The live compiled-UI action checks also passed for RGB (17×13 → 68×52, tile 200) and RGBA (35×9 → 140×36, tile 32): both uploaded actual PNG bytes, obtained a saved result, downloaded it and found it in history. Comparison with the original CPU reference found maximum channel difference 1 in both outputs, with the RGBA alpha channel exactly equal. The isolated server exited successfully and its process group drained; sampled DRM allocations peaked at 218,464,256 bytes. A separate served-page smoke check parsed all 27 modules and constructed 160 virtual DOM nodes across three panels. These are Node application-action and page smoke tests, not an interactive browser rendering test. Background-removal and SVG controls have mocked interaction coverage here, not new live-model validation. These results do not claim a production deployment or validate every image model on every GPU.

## Audio transcription

With an installed recognition model, open its audio server with `--wfe`, using the same frontend build and embedding order described above. For example, substitute your installed model ID in:

```sh
NS_DIST="$PWD/dist" ./target/release/nextrocm audio serve MODEL_ID \
  --here --wfe --host 127.0.0.1 --port 8087
```

Open `http://127.0.0.1:8087/`. A recogniser displays **Transcribe**, while music and text-to-speech models retain their existing controls. Choose or drop a WAV, MP3, FLAC, Ogg or M4A recording up to 64 MiB. The server decodes it; browser playback support may vary by format. Choose an advertised language or leave automatic detection selected. Word timestamps and English translation appear only when supported. An optional initial prompt supplies names, spelling or context.

Select **Transcribe** to see the text, detected language, duration and available timed segments or words. **Cancel** requests cancellation through the existing server API. Download `transcript.txt` or `transcript.json` to keep the result; transcripts are not saved in song history. Empty recognised text is shown as “No speech detected.”

CPU-only checks:

```sh
bash wfe/build.sh audio
node --experimental-default-type=module wfe/test/audio-transcription.mjs
```

These check capability routing, preservation of capabilities during busy info polling, upload limits and read errors, exact request fields, transcript downloads, timestamps, errors/retries/cancellation, duplicate prevention and the existing music/TTS branches.

For an isolated live server and an independently retained expected transcript:

```sh
node --experimental-default-type=module wfe/test/audio-transcription-live.mjs \
  --base http://127.0.0.1:8087/ --input /absolute/path/recording.wav \
  --expected /absolute/path/expected.txt --out /absolute/path/new-audio-check
```

The harness uploads the real WAV through the compiled UI action and checks the returned transcript against the expected file, ignoring only surrounding whitespace. It also checks text/JSON download content and input identities. It uses a Node FileReader adapter, not browser automation. The audio interaction checks and two live UI-action runs passed with `whisper-ggml-tiny.en` on the retained 11-second JFK recording. Both returned the exact expected transcript (apart from permitted surrounding whitespace), two timed segments and matching text/JSON downloads. Reported recognition times were 0.231 and 0.223 seconds; these are this short fixture’s request times, not a general throughput benchmark. Both isolated server runs exited successfully with their process groups drained and sampled DRM allocations peaking at 363,450,368 bytes. The final check found the GPU idle, no KFD compute processes and automatic power management restored. This validates the compiled UI-to-native recognition workflow for that fixture; it does not establish general ASR accuracy, browser rendering or production deployment.

## Deployed on Mycroft

The updated interfaces are deployed at <http://mycroft:8000/>. Use
[ESRGAN](http://mycroft:8000/open/real-esrgan/) or
[Whisper transcription](http://mycroft:8000/open/whisper-ggml-tiny.en/).
Both were tested through that deployed front door on the 680M. The
[deployment record](deployments/mycroft-2026-11-23.md) includes results and rollback
instructions; test workers were unloaded afterward so normal model loading
remains available.
