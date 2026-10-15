# Validation · 2026-10-15

## Passed

- Downloaded the actual 471,320,622-byte pinned FP16 ONNX export. SHA-256 matches the Hugging Face LFS object.
- Inspected real input `image: float32[1,3,2048,2048]`, output `alpha: float32[1,1,2048,2048]`, default opset 20, and the final Sigmoid.
- Ran the exact JavaScript compatibility adapter on that file. Adapted graph passes `onnx.checker.check_model`; every initializer is byte-for-byte equal to the original. Only the intended 300 GridSample and 48 exact GELU nodes change, plus opset imports and node ordering.
- Numerically compared ONNX GridSample-20 + exact GELU against GridSample-19 bilinear + Microsoft GELU on a nontrivial sampled tensor (including out-of-bound sampling). ONNX Runtime 1.23.2 CPU output maximum absolute difference: 0.0.
- Six core regression tests cover NCHW/ImageNet normalization, transparent input compositing, fractional alpha and edge interpolation, no double Sigmoid, original PNG alpha preservation, color cleanup preserving alpha and opaque regions, and bounded invalid inputs.
- Real Chromium UI test passes upload, controls, previews, same-size reference import, wrong-size rejection, optional cleanup, original-size PNG export, cancellation, invalid file cleanup, missing app.js startup reporting, and 390px mobile overflow check. **The browser UI test uses clearly labeled mocked model results.** It is not a model quality benchmark.
- Inspected desktop and mobile screenshots with synthetic test graphics. Private user photos are not shipped or committed.

## Limits of validation

- The environment provides a software Vulkan/WebGPU adapter without `shader-f16`. It cannot execute this FP16 WebGPU model. No claim of completed real WebGPU inference or measured GPU speed/VRAM requirement is made.
- Native CPU execution of the full 2048px FP16 export exceeded this environment's 8 GiB memory limit, both with default optimization and with the CPU arena/memory pattern disabled. CPU is not a shipping fallback in this test app.
- No real portrait/dog/cat quality comparison against Photoshop was completed for this model in this environment. Run those comparisons on the user's FP16-capable GPU. This is the purpose of the separate test build.
- PNG output follows the browser 8-bit RGBA canvas path. Color-managed 16-bit/HDR production export is outside this isolated prototype.

The native Rust application, its existing browser build, and `tools/ai-background-test` are not modified by this test folder.
