# Validation · v2 FP32 · 2026-10-15

## Reason for this update

The user reported the v1 alpha-validity error after inference. That error
combined NaN, infinity and out-of-range cases, so it did not establish the exact
underlying GPU failure. v2 uses the publisher's unquantized FP32 reference and
basic graph optimization to avoid FP16 arithmetic and extended/all fusions.
It does not claim that the original failure was conclusively reproduced.

## Passed

- Downloaded the actual pinned FP32 `model.onnx`: 932,150,975 bytes;
  SHA-256 `a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd`.
- Verified input `image: float32[1,3,2048,2048]`, output
  `alpha: float32[1,1,2048,2048]`, 29,105 nodes, 873 finite FP32 initializers,
  and final Sigmoid. Both original and adapted files pass the ONNX checker.
- The exact shipped JavaScript adapter produces 932,151,164 bytes. All 873
  initializers remain byte-identical. Only the 300 GridSample and 48 GELU node
  encodings change, plus opset imports and topological ordering.
- Small GridSample-20 / exact GELU versus GridSample-19 / Microsoft GELU CPU
  oracle on ONNX Runtime 1.23.2: maximum absolute difference 0.0.
  This tests the operator substitutions, not the full HR network.
- Ten Node regression tests passed: normalization, soft alpha, interpolation,
  original transparency, no double Sigmoid, color cleanup, invalid input,
  invalid alpha statistics, FP32 configuration without shader-f16,
  basic optimization, resource cleanup and missing GPU / memory failure.
  Worker control-flow tests run the real generated worker with mocked model
  downloads and inference. NaN, infinities and out-of-range alpha produce no
  result message. All-NaN masks fail rather than becoming transparent pixels.
- Real Chromium UI test passed upload, previews, reference comparison,
  original-size PNG export, cancellation, invalid upload, missing app.js,
  error recovery, no stale-result export after a failed rerun, JSON diagnostic
  download without the photo filename, and 390px mobile overflow checks.
  **GPU/model output is mocked in these browser UI tests.**
- Inspected desktop and mobile screenshots, including the error/report state.
  Test graphics are synthetic; no private photo is shipped or committed.
- Generated worker matches source; app and worker syntax checks pass.

## Limits

- Full 2048px FP32 inference on a physical GPU was not run in this environment.
  This is not evidence of measured speed, VRAM requirements or improved visual
  quality. Test on the user's machine with the same original photograph.
- This environment has an 8 GiB RAM limit and no representative discrete GPU.
  The earlier v1 full-model CPU attempt exceeded that memory limit. CPU is
  not a shipping fallback, and v2 does not reduce the model resolution.
- No completed real hair/fur benchmark against Photoshop is claimed.
- The FP32 download and GPU memory use are larger than v1 FP16. The page shows
  this before processing. Its cache key and pinned URL differ from v1.
- Export uses the browser's 8-bit RGBA canvas path, not HDR/16-bit color.

## Reproduce

```sh
node build.mjs --check
node --test tests/core.test.mjs tests/worker.test.mjs
BROWSER_EXECUTABLE=/path/to/chromium node tests/browser.cjs
# Optional actual model checks, after running adaptPinnedModel on model.onnx:
python tests/verify-model.py /path/to/model.onnx /path/to/adapted-fp32.onnx
```

Research references: [ONNX Runtime float16 / mixed-precision guidance](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
and [WebGPU graph-fusion non-finite-output report](https://github.com/microsoft/onnxruntime/issues/32145).
The latter concerns another model/runtime and is a reason for conservative
configuration, not proof that it caused this user's failure.

Native Rust PhotoCraft and `tools/ai-background-test` remain unchanged.
