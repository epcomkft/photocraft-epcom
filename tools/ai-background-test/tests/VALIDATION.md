# v5 validation — 2026-10-14

## Passed

- `node --test tools/ai-background-test/tests/matting.test.mjs`: **13/13**.
- `node --check tools/ai-background-test/app.js` and `git diff --check`.
- `tests/ui-flow.cjs` with JSDOM 26.1.0 / native canvas and mocked model calls:
  image loading, processing locks, disposal, transparent PNG despite white preview,
  correction brush and undo, cached refinement, cancellation, returned analysis
  resize, full original-size output. This tests application flow, not GPU inference.
- Real BEN2 fp16 ONNX graph and ViTMatte fp32 ONNX graph (revisions in README),
  ONNX Runtime **1.23.2 CPU**, on the user's private **1365 × 2048** portrait.
  BEN2 input: bilinear 1024-square RGB, ImageNet normalization, float32 input to the
  fp16-weight graph. ViTMatte input: RGB/127.5-1 plus trimap/255, 512-square tiles.
  The actual `matting-core.mjs`, `matting-engine.mjs` and `edge-colors.mjs` were
  used via a local stdio inference bridge, not a Python reimplementation of the fix.
- 19 ViTMatte tiles; approximately 16.5 seconds for matting/blending/edge colours
  on this CPU environment (not a browser performance promise). Original dimensions
  and sampled opaque face pixels preserved. A known isolated background speck on
  the right was zero-alpha after correction. Hair inspected at 100% on black and
  white composites: the thick grey/white fringe was substantially reduced.

Private images, intermediate masks, model binaries and outputs are not committed.
No ground-truth alpha exists for this photo; do not turn these checks into a claimed
percentage accuracy or a general hair/fur quality score.

## Still to verify on the user's machine

- Real browser WebGPU execution, driver memory behaviour and visual layout.
  Local Chromium launch was blocked by the execution environment's socket
  restriction (`Operation not permitted`); no browser/GPU pass is claimed.
- Additional low-contrast human hair, long cat whiskers and fluffy dog coats.
  Automatic support estimation can still miss ambiguous strands. Use the unknown
  edge brush / wider band, and explicitly mark foreground/background if necessary.

## Optional application-flow check

Install test-only dependencies outside the Rust workspace, then expose them through
`NODE_PATH` and run:

```sh
node --experimental-vm-modules tools/ai-background-test/tests/ui-flow.cjs
```

The runtime folder has no npm build step or additional runtime package dependency.
