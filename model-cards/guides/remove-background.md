# Background removal

`image.remove_background` keeps a picture's object and makes its background transparent: an RGBA PNG at the
picture's size (its own colours, the mask as straight alpha) and, with `"mask": true`, the 8-bit mask on its own.
A job like the other image methods (wait, reentrant_id, label, image.status).

| model | | |
|---|---|---|
| `birefnet` | BiRefNet (Swin-L), MIT | the best edges: hair, fur, thin parts |
| `isnet-general-use` | IS-Net general-use, rembg's default model | what rembg gives today |

```json
{"image": "upload:up_1792400000_chair.png", "model": "birefnet", "mask": true, "response_format": "url"}
```

- `image`: PNG, JPEG or WebP as base64, a data: URL or `upload:<name>` (image.upload).
- `response_format`: `b64_json` (default) or `url`.
- The answer's `data[0]`: `b64_json` or `url` (the RGBA picture), `width`, `height`, `model`, and with `mask`
  `mask_b64_json` or `mask_url`.
- rembg's own cut-out multiplies the colours by the mask; here they are kept (multiply RGB by A / 255 for rembg's bytes).


Both original models are deployed on development-host; the latest admission records26
enabled models with existing settings preserved. ISNet and BiRefNet use the
original1024 network and return pictures at the input size. Conservative card
budgets are2 GiB and4 GiB respectively; these are placement estimates, not
measured peaks. See deployment and measured scopes (private development evidence is not included).
