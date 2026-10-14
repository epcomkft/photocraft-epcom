# EPCOM AI background removal test

Browser-only proof of concept for:

**BEN2 → automatic trimap → ViTMatte → transparent PNG**

This is intentionally separate from the PhotoCraft UI. It exists to validate hair/fur quality before the models are wired into PhotoCraft's `layer.removeBackground` command.

## Run

Serve this folder over HTTPS and open `index.html` in a recent Chrome or Edge browser with WebGPU.

For the EPCOM host a convenient test location is:

`/admin/photocraft-ai-test/`

Copy these three files there:

- `index.html`
- `app.js`
- `style.css`

The first run downloads the models from Hugging Face. Later builds can self-host pinned model files.

## Models

- `onnx-community/BEN2-ONNX` — foreground/background estimate.
- `Xenova/vitmatte-small-composition-1k` — alpha matting from image + trimap.

## Current limitation

The proof of concept downsizes large source images to the configured maximum long side before matting. This is deliberate for validation.

The production PhotoCraft implementation should refine only the uncertain edge band in overlapping original-resolution tiles, then blend the tile alpha results into one mask. That avoids destroying single-pixel hair/fur detail on large photographs.
