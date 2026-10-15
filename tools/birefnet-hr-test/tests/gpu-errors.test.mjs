import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyGpuError, gpuLimits } from '../gpu-errors.mjs';
test('the reported 3 GiB / 2 GiB validation error is not reported as insufficient VRAM', () => {
  const error = new Error('failed to call OrtRun(). ERROR_CODE: 1, ERROR_MESSAGE: WebGPU validation failed. Buffer size (3221225472) exceeds the max buffer size limit (2147483648).\n - While calling [Device].CreateBuffer([BufferDescriptor]).');
  const out = classifyGpuError(error);
  assert.equal(out.code, 'GPU_BUFFER_LIMIT');
  assert.deepEqual(out.buffer, { requested: 3221225472, limit: 2147483648 });
  assert.match(out.message,/3,00 GiB/); assert.match(out.message,/2,00 GiB/);
  assert.doesNotMatch(out.message,/Zárd be|nincs elég/);
});
test('allocation failures, device loss, binding limits and alpha failures stay distinct', () => {
  assert.equal(classifyGpuError(new Error('allocation failed')).code,'MEMORY_ALLOCATION_FAILED');
  assert.equal(classifyGpuError(new Error('device was lost')).code,'GPU_DEVICE_LOST');
  assert.equal(classifyGpuError(new Error('exceeded maxStorageBufferBindingSize')).code,'GPU_BUFFER_LIMIT');
  const e=new Error('Invalid alpha');e.code='MODEL_ALPHA_INVALID';assert.equal(classifyGpuError(e).code,e.code);
});
test('adapter and device diagnostics record only supported numeric limits', () => {
  assert.deepEqual(gpuLimits({ maxBufferSize:2147483648, maxStorageBufferBindingSize:2147483648, maxComputeWorkgroupsPerDimension:65535, unrelated:'no' }),{ maxBufferSize:2147483648, maxStorageBufferBindingSize:2147483648, maxComputeWorkgroupsPerDimension:65535 });
  assert.deepEqual(gpuLimits(undefined),{});
});
