# Validation · v4 edge colours / UI · 2026-10-15

## Scope

Only edge-colour postprocessing, its UI/control flow, export filenames and
related diagnostics changed. The v3 pinned model, graph adapter, inference
resolution/precision, ONNX Runtime and browser model cache are unchanged.
The v3 model checks below are historical evidence, not a new full inference run.

## Passed for v4

- Twenty Node tests pass, including two independent upstream PyMatting CPU
  oracle fixtures. The serial Python reference was run with NumPy and numba's
  decorators replaced by no-ops; its RGB results match the JavaScript multilevel
  estimate within one 8-bit channel unit (mean difference below 0.05).
- Synthetic composites with known foreground/background colours verify recovery
  of dark and white fine edges, including an overestimated-alpha fringe. Tests
  verify unchanged alpha and opaque interiors, distinct 10/20/30% outputs,
  byte-identical restoration at 0%, a single preparation per image, transparent
  PNG handling and explicit invalid-input rejection.
- Real Chromium UI checks pass: the cleanup slider is visible with diagnostics
  collapsed, preset buttons work, it remains interactive while composing, rapid
  changes settle at the latest selected value, and PNG filenames include that
  applied percentage. Existing upload/export/reference/cancel/error-report and
  mobile overflow tests also pass. GPU model output is mocked in these UI tests.
- Desktop and 390px mobile screenshots were visually inspected. Committed UI
  fixtures are synthetic. No private photo is included in the repository/ZIP.
- The real new colour code was also evaluated locally with the user's original
  1365 x 2048 photograph and the alpha channel of their v3 export, without rerunning
  the GPU model. The top-of-head grey fringe is visibly reduced at stronger
  settings on black and white composites; some fringe remains. This is a visual
  assessment without a ground-truth matte or a measured Photoshop comparison.
  Export alpha and every fully opaque source pixel remain byte-identical.
- That 2.80 MP sample took about 0.68 s for first preparation+composition and
  0.03 s for later strengths in local Node. A synthetic 16 MP (the app's maximum)
  took 3.70 s initially and 0.15 s cached; process peak RSS was about 734 MiB.
  For comparison, v3 on the same 16 MP fixture took 0.47 s initially and
  0.48 s on the next strength, with about 548 MiB peak process RSS. v4 trades
  a more expensive first estimate for cheaper subsequent updates.
  These are local CPU measurements of colour processing, not browser/GPU
  inference timings or a guarantee for the user's machine.
- Generated worker matches sources; classic app/worker syntax checks pass.

## Remaining limits

- Alpha is preserved deliberately. Missing hairs and wrong mask coverage cannot
  be repaired by this colour-only correction. Strong settings may change the
  apparent colour of highlights or translucent materials; compare against 0%.
- Previously transparent input RGB is left alone because it does not provide a
  reliable photographed background. Missing foreground/background context also
  leaves original RGB unchanged; the UI reports these cases.
- No new full model run on a physical GPU, dog/cat benchmark or broad portrait
  benchmark was completed here. The user's successful v3 export demonstrates
  their prior model run, not universal v4 quality or hardware compatibility.

## Reproduce

```sh
node build.mjs --check
node --test tests/core.test.mjs tests/worker.test.mjs tests/gpu-errors.test.mjs tests/edge-colors.test.mjs
BROWSER_EXECUTABLE=/path/to/chromium node tests/browser.cjs
```

The source of `tests/fixtures/foreground-oracle.json` is the independently
executed serial PyMatting function, source blob
`c117f291779fcf2c8b01997dbc84c2a0c5f5b699`, default parameters, with generated
small gradient composites. PyMatting's MIT notice is included alongside the
other licenses; the JavaScript adaptation is recorded in `ATTRIBUTION.md`.

---

# Previous v3 model validation · 2026-10-15

## Observed failure

The user's v2 JSON report identified an exact WebGPU validation error during
inference: a 3,221,225,472-byte buffer exceeded the device's 2,147,483,648-byte
maximum buffer size. This establishes a per-buffer limit, not exhaustion of
the RTX 5070 Ti's 16 GB of total VRAM. The earlier generic memory/device-loss
message did not make that distinction.

Shape inference on the pinned model locates a 3 GiB final Resize tensor
`[1,192,2048,2048]`, followed by a 3.75 GiB Concat tensor
`[1,240,2048,2048]`. A one-output-channel 1x1 Conv immediately follows.

## Change

The adapter splits that final Conv's weights at channel 192. It projects the
low-resolution branch to one channel before bilinear Resize, independently
projects the 48-channel high-resolution branch, and adds both results before
the original Sigmoid. The original bias is applied once. This uses linearity:

`W * [Resize(x), y] + b = Resize(Wlow * x) + Whigh * y + b`.

Full 2048 x 2048 inference and FP32 precision are retained. Both oversized
intermediate tensors are removed. The original valid node order is preserved
where possible, avoiding gratuitous extension of tensor lifetimes. Unexpected
graph structure fails closed rather than applying this rewrite to another model.

## Passed

- Actual pinned FP32 source: 932,150,975 bytes; SHA-256
  `a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd`.
  Input `image: float32[1,3,2048,2048]`, output
  `alpha: float32[1,1,2048,2048]`, 29,105 source nodes, 873 finite FP32
  initializers. Original and adapted files pass the ONNX checker.
- The shipped JavaScript adapter produces 932,152,642 bytes, 29,107 nodes
  and 876 initializers. All 873 original initializers remain byte-identical;
  two new weight slices match their original bytes exactly and one new small
  int64 index tensor is added. The existing 300 GridSample / 48 GELU
  compatibility substitutions are also checked.
- Shape inference finds no statically known FP32 tensor of 2 GiB or larger
  in the adapted graph. The largest known tensor is 1.25 GiB. This is not a
  guarantee about runtime work buffers, total GPU memory or operator support.
- Eight real ONNX Runtime 1.23.2 CPU comparisons of the original and rewritten
  head passed. The tests run the actual JavaScript adapter on synthetic graph
  fixtures; they vary dimensions, degenerate axes and input amplitudes. Maximum
  absolute alpha difference: 0.0000014901161193847656. Maximum logit difference:
  0.000019073486328125. Shape outputs and exact weight slices match. A changed
  final-convolution shape is rejected. These are synthetic head tests, not full
  network or learned-photo quality comparisons.
- Small GridSample-20 / exact GELU versus GridSample-19 / Microsoft GELU CPU
  oracle: maximum absolute difference 0.0.
- Fourteen Node regression tests passed, including soft alpha, interpolation,
  original transparency, no double Sigmoid, color cleanup, invalid input,
  invalid-alpha rejection, FP32 configuration without shader-f16, basic graph
  optimization and cleanup. The exact reported 3 GiB / 2 GiB error is classified
  as `GPU_BUFFER_LIMIT`; genuine allocation and device-loss errors remain
  distinct. Worker control-flow tests execute the generated worker with mocked
  model downloads and inference. Invalid masks and failed runs emit no result.
- Real Chromium UI tests passed upload, previews, reference comparison,
  original-size PNG export, cancellation, invalid upload, missing app.js,
  error recovery, no stale-result export, JSON diagnostic download, mobile
  overflow checks and stale-v2-worker rejection. **GPU/model output is mocked
  in these UI tests.** Desktop and mobile error-state screenshots were visually
  inspected. Test graphics are synthetic; no private photo is shipped.
- Generated worker matches source; app and worker syntax checks pass.

## Limits

- Full 2048px FP32 inference on a physical GPU was not run in this environment.
  Test the release on the user's hardware with the same original photograph.
  No measured speed, guaranteed VRAM minimum or Photoshop quality comparison
  is claimed.
- A full native CPU run with the v3 head rewrite was attempted before the final
  node-order-preservation refinement. Session creation succeeded; inference was
  killed with exit 137 under this environment's 8 GiB memory limit. It produced
  no completed full-model output. CPU is not a shipping fallback.
- v3 reuses the v2 FP32 download/cache deliberately; its URL, hash and weights
  are unchanged. No v2 model-cache deletion is required. The full final adapted
  graph was checked after the node-order refinement.
- Export uses the browser's 8-bit RGBA canvas path, not HDR/16-bit color.

## Reproduce

```sh
node build.mjs --check
node --test tests/core.test.mjs tests/worker.test.mjs tests/gpu-errors.test.mjs
# Requires onnx, onnxruntime and numpy:
python tests/head.test.py
# After running adaptPinnedModel on the pinned source model:
python tests/verify-model.py /path/to/model.onnx /path/to/adapted-fp32-v3.onnx
# Requires Playwright and @napi-rs/canvas:
BROWSER_EXECUTABLE=/path/to/chromium node tests/browser.cjs
```

Native Rust PhotoCraft and `tools/ai-background-test` remain unchanged.
