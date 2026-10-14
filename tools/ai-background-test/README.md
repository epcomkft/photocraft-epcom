# EPCOM AI background removal test

Browser-only proof of concept for a high-detail background removal pipeline:

**BEN2 → SAM2 semantic support → safe trimap → ViTMatte HQ → transparent PNG**

This stays separate from the PhotoCraft UI until hair/fur quality is validated.

## What changed in HQ mode

- The original image is kept at its full resolution.
- `maxSide` only limits the BEN2/SAM2 semantic analysis image.
- BEN2 is the primary subject mask.
- SAM2 is advisory only: it may widen the candidate subject area, but it can no longer erase foreground that BEN2 already recognized.
- The trimap keeps the object interior as certain foreground and sends a wide edge band to ViTMatte.
- ViTMatte runs on the original-resolution pixels in overlapping 512 px tiles with 128 px overlap.
- The wider unknown band is intentional for human hair, dog fur, whiskers, ear fur and other thin/partially transparent details.
- The exported PNG keeps the original image dimensions.

This specifically fixes the failure mode where a single SAM2 point selected only hair/fur and the old hard SAM2 gate removed the face/body.

## Run

Serve this folder over HTTPS and open `index.html` in a recent Chrome or Edge browser with WebGPU.

For the EPCOM host a convenient test location is:

`/admin/photocraft-ai-test/`

Copy these files there:

- `index.html`
- `app.js`
- `sam2.js`
- `style.css`

The first run downloads the models from Hugging Face. Later builds can self-host pinned model files.

## Models

- `onnx-community/BEN2-ONNX` — primary foreground/background estimate.
- SAM2 Hiera Tiny ONNX — semantic main-subject support.
- `Xenova/vitmatte-small-composition-1k` — high-resolution alpha matting from image + trimap.

## Recommended defaults

- BEN2/SAM2 analysis max side: **2048 px**
- sure-foreground inner shrink: **6 px**
- hair/fur search band: **64 px**

For very difficult long hair or fluffy dog fur, increasing the search band to 80–96 px can help, at the cost of more ViTMatte tiles and longer processing time.
