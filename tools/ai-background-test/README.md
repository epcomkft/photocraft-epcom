# PhotoCraft background removal — v6

Browser-only background-removal test module. It stays separate from PhotoCraft's
native Rust application. Images are processed locally; model files are downloaded,
not the user's images uploaded. No API key or Python server is needed to use it.

**BEN2 confidence → editable trimap → full-resolution ViTMatte → edge colour estimation → RGBA PNG**

## The quality fix

The previous trimap treated BEN2 values as low as **20/255** as certain foreground
(after a small erosion), and could also promote weak BEN2 pixels through an
unprompted SAM2 mask. These pixels were forced to alpha 255 after matting. This
locked grey background bands around hair into the exported image.

- Certain foreground now requires **250/255** confidence, followed by erosion.
  Uncertain edges remain unknown. There is no closing operation that fills hair gaps.
- The candidate region grows from a 128/255 subject estimate, not every weak
  background response. Small isolated components are optional noise candidates;
  all substantial subjects are preserved. Nearby small strands remain inside the
  unknown band. The speck filter can be turned off for small separate objects.
- Automatic one-point SAM2 union is no longer part of this pipeline. `sam2.js`
  remains as a legacy experimental module; v5 neither imports nor downloads it.
- ViTMatte operates on original pixels in **512 × 512** tiles with **128 px overlap**.
  Predictions are blended as floats, then quantized once. Known trimap pixels stay
  exact; missing, malformed or non-finite outputs are errors, never silent fallback.
- Partially transparent edge RGB is corrected using nearby foreground/background
  colours and regularized colour unmixing. Opaque colours are unchanged. This
  reduces grey/white contamination when compositing on dark engraving backgrounds.
  Set the correction to 0 to preserve all original edge colours. For an already
  transparent source, colour unmixing is skipped and original alpha is multiplied
  by the new matte, never overwritten.
- The analysis resize now uses the returned image. The source and PNG dimensions
  stay unchanged. The first pixel of bilinear mask resizing no longer mixes with
  the wrong neighbouring pixel.
- BEN2 is released before loading ViTMatte; tensors are disposed after every tile.
  Refinements on the same image reuse the BEN2 mask. File changes and settings are
  locked while processing. Cancellation stops after the current model call.
- Sources are limited to 40 MB, 16 megapixels and 8192 pixels on either side, with
  visible errors rather than silent downsampling. Inputs: PNG, JPEG, WebP.

## Install / update

Serve over HTTPS, for example at `/admin/photocraft-ai-test/`. A fresh install
requires only these **three runtime files**:

```
index.html
app.js
style.css
```

**Updating v5: replace `index.html` and `app.js` together.** The stylesheet has not
changed. The header should show **PhotoCraft · háttéreltávolítás v6** after reload.
The existing site's authentication/access protection should remain in place.

### v6 startup fix for shared hosting

The affected host returned the v5 `.mjs` files with HTTP 200 **but no Content-Type**.
The browser therefore rejected the module graph before any image-selection handler
could be registered. The selected filename was visible while the status stayed
“Válassz egy képet.” and all action buttons remained disabled.

The distributed `app.js` is now one self-contained **classic script**. It has no
local module imports and does not need `.mjs` MIME configuration or an `.htaccess`
change. The three `.mjs` algorithm files and `app-source.mjs` are development sources;
they do not need to be uploaded. `sam2.js` is also not used by this version.
Only the external Transformers.js library is loaded dynamically when AI processing
starts, after the local UI is already working.

File selection is disabled until startup finishes. If `app.js` is missing, fails
to initialize or stalls, an on-page message identifies the file to upload instead
of leaving a seemingly functional, inert file picker. The HTML requests `app.js?v=6`
and `style.css?v=6` to invalidate v5 browser caches.

After changing the source modules, rebuild the checked-in distribution file:

```sh
node tools/ai-background-test/build.mjs
node tools/ai-background-test/build.mjs --check
```

The build has no npm dependencies. Edit `app-source.mjs` for UI/controller changes;
`app.js` is generated. Source modules remain directly testable in Node.

Use a WebGPU-capable Chrome, Edge or Opera with hardware acceleration enabled.
First run downloads approximately 320 MB. Model revisions and Transformers.js
3.8.1 are pinned for reproducibility:

| Model | Revision |
| --- | --- |
| `onnx-community/BEN2-ONNX` (fp16) | `c552aa82688edce09f0ac9d2e31ad53d9d629010` |
| `Xenova/vitmatte-small-composition-1k` (fp32) | `6bc1297f6140f055a227b6d2cfe8c093281f35d2` |

## Use and corrections

1. Choose an image and run background removal.
2. Inspect at 100% on black and white backgrounds, or view the alpha mask.
   Preview backgrounds never affect the transparent PNG export.
3. If needed, paint **background** (red), **keep** (green), or **unknown fine edge**
   (blue) on the original. Green locks opaque foreground, so use blue around hair,
   fur and whiskers. Marks apply only where explicitly painted.
4. Rerun refinement; the subject model does not need to rerun. Undo and erasing
   marks are available. Changed settings/marks disable stale PNG exports until
   refinement succeeds.
5. Download `<original-name>-hatter-nelkul.png` at the original resolution.

Defaults: inner band 12 px, outer band 32 px, colour correction 80%. Band radii
scale with the image's longest side relative to 2048 px. Increase the outer band
for long whiskers or flyaway hair; widening it also admits more background and
increases processing time. The blue brush can designate a missed region directly.

This is image matting, not generative reconstruction. No model can guarantee every
hair on every low-contrast, blurred or compressed image. Difficult foreground
selection can still require correction marks. Do not infer quality from a SAM2
predicted-IoU score; it is no longer displayed as an output-quality indicator.

## Verification

Run the deterministic, dependency-free regressions:

```sh
node --test tools/ai-background-test/tests/matting.test.mjs
node tools/ai-background-test/build.mjs --check
```

They cover weak-background foreground locks, hair gaps, soft strands, isolated
specks and multiple subjects, manual trimap overrides, alpha preservation, resize
edges, tiled overlap continuity, small images, cancellation, malformed model
outputs and foreground colour unmixing.

Actual pinned BEN2 and ViTMatte ONNX inference was also exercised on the supplied
1365 × 2048 portrait using ONNX Runtime CPU. The exact JavaScript trimap, tile
blending and edge-colour modules in this folder produced the corrected output.
The hair edge was inspected on black and white backgrounds at original resolution;
the isolated right-side speck was removed and sampled opaque face pixels remained
opaque. No ground-truth alpha was available; this is a visual regression check,
not a numeric quality benchmark or a validation across dog/cat breeds.

Browser/GPU validation status is recorded in `tests/VALIDATION.md`. Private user
photos and their derived outputs are deliberately not included in the repository.
