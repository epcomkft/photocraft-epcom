import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeRgb, resizeAlpha, readModelAlpha, validateSize, mask, exportAlpha } from '../core.mjs';
import { composeCutout } from '../edge-colors.mjs';
import { adaptPinnedModel } from '../onnx-compat.mjs';
test('RGB ImageNet normalization is NCHW; transparent input is white', () => {
  const x = normalizeRgb(Uint8ClampedArray.from([255,0,0,255,0,0,0,0]),2,1);
  assert.ok(Math.abs(x[0] - (1-.485)/.229) < 1e-6);
  assert.ok(Math.abs(x[2] - (0-.456)/.224) < 1e-6);
  assert.ok(Math.abs(x[5] - (1-.406)/.225) < 1e-6);
});
test('soft alpha has no threshold and resize does not wrap boundary pixels', () => {
  const m = resizeAlpha(Float32Array.from([0,1]),2,1,4,1);
  assert.deepEqual([...m.data],[0,64,191,255]);
});
test('model alpha is already sigmoid; only finite [0,1] floats are accepted', () => {
  const t={type:'float32',dims:[1,1,2048,2048],data:new Float32Array(2048*2048)};t.data[0]=.25;t.data[1]=1;
  const out=readModelAlpha(t);assert.equal(out[0],.25);assert.equal(out[1],1);assert.equal(out[2],0);
  t.data[10]=NaN;assert.throws(()=>readModelAlpha(t));
  assert.throws(()=>readModelAlpha({...t,dims:[1,1,1024,1024]}));
});
test('a wholly invalid output is rejected with JSON-safe stats; no NaN-to-zero masking', () => {
  const t={type:'float32',dims:[1,1,2048,2048],data:new Float32Array(2048*2048).fill(NaN)};
  assert.throws(()=>readModelAlpha(t), e => e.code==='MODEL_ALPHA_INVALID' && e.alphaStats.invalid===4194304 && e.alphaStats.min===null && e.alphaStats.max===null);
  t.data.fill(.5); t.data[0]=-1e-8; t.data[1]=1+1e-7;
  const out=readModelAlpha(t); assert.equal(out[0],0);assert.equal(out[1],1);assert.equal(out[2],.5);
  t.data[5]=1.005;assert.throws(()=>readModelAlpha(t),e=>e.alphaStats.outOfRange===1);
});
test('original PNG alpha is multiplied, not replaced', () => {
  const src={width:2,height:1,channels:4,data:Uint8ClampedArray.from([80,90,100,128,20,30,40,0])};
  const out=composeCutout(src,mask(Uint8ClampedArray.from([128,255]),2,1),0);
  assert.deepEqual([...exportAlpha(out.data)],[64,0]);assert.deepEqual([...out.data.slice(0,3)],[80,90,100]);
});
test('edge cleanup never changes alpha or opaque interior', () => {
  const src={width:3,height:1,channels:4,data:Uint8ClampedArray.from([240,240,240,255,120,110,100,255,20,10,0,255])};
  const m=mask(Uint8ClampedArray.from([0,128,255]),3,1), before=composeCutout(src,m,0),after=composeCutout(src,m,1);
  assert.deepEqual(exportAlpha(before.data),exportAlpha(after.data));assert.deepEqual(after.data.slice(8),before.data.slice(8));
});
test('oversized, malformed dimensions and damaged model fail explicitly', () => {
  for(const [w,h] of [[0,1],[8193,1],[4001,4001],[NaN,1],[1.5,2]]) assert.throws(()=>validateSize(w,h));
  assert.throws(()=>adaptPinnedModel(new Uint8Array([58,128])));
});
