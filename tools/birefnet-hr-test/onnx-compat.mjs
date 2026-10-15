// Narrow, zero-copy protobuf traversal for the pinned PinkPixel FP32 export only.
// ORT WebGPU 1.23.2 supports GridSample 16-19 (2D bilinear), not opset 20
// (renames this mode 'linear'). Map exact GELU to the equivalent Microsoft op.
// Original initializers are retained byte-for-byte. The high-resolution head
// adds two byte-exact channel slices of its 1x1 weights and one index constant.
const utf8 = new TextDecoder(), encode = new TextEncoder();
function varint(bytes, start, end) {
  let n = 0, factor = 1, p = start;
  for (let i = 0; i < 8 && p < end; i++) {
    const b = bytes[p++]; n += (b & 127) * factor;
    if (!(b & 128)) { if (!Number.isSafeInteger(n)) break; return [n, p]; }
    factor *= 128;
  }
  throw new Error('Sérült ONNX mező. Töröld a modellgyorsítótárat és próbáld újra.');
}
function fields(bytes) {
  const result = []; let p = 0;
  while (p < bytes.length) {
    const start = p; let tag, n; [tag, p] = varint(bytes, p, bytes.length);
    const wire = tag % 8, id = Math.floor(tag / 8); let from = p;
    if (!id) throw new Error('Érvénytelen ONNX azonosító.');
    if (wire === 2) { [n, p] = varint(bytes, p, bytes.length); from = p; p += n; }
    else if (wire === 0) { [n, p] = varint(bytes, p, bytes.length); }
    else if (wire === 1) p += 8;
    else if (wire === 5) p += 4;
    else throw new Error('Nem támogatott ONNX mező.');
    if (p > bytes.length) throw new Error('Hiányos modellletöltés.');
    result.push({ id, wire, number: n, data: bytes.subarray(from, p), raw: bytes.subarray(start, p) });
  }
  return result;
}
function vint(n) { const b = []; do { const r = n % 128; n = Math.floor(n / 128); b.push(r + (n ? 128 : 0)); } while (n); return Uint8Array.from(b); }
function join(parts) { const out = new Uint8Array(parts.reduce((n, b) => n + b.length, 0)); let p = 0; for (const b of parts) { out.set(b, p); p += b.length; } return out; }
function field(id, body) { return join([vint(id * 8 + 2), vint(body.length), body]); }
function str(id, text) { return field(id, encode.encode(text)); }
function num(id, n) { return join([vint(id * 8), vint(n)]); }
function textField(list, id) { const f = list.find(f => f.id === id); return f ? utf8.decode(f.data) : ''; }
function strings(list, id) { return list.filter(f => f.id === id).map(f => utf8.decode(f.data)); }
function makeNode(name, op, inputs, outputs, attrs = []) {
  const body = join([...inputs.map(s => str(1, s)), ...outputs.map(s => str(2, s)), str(3, name), str(4, op), ...attrs]);
  return { fields: fields(body), raw: field(1, body) };
}
function integerAttribute(name, value) { return field(5, join([str(1, name), num(3, value), num(20, 2)])); }
function tensorDims(list) {
  const result = [];
  for (const f of list.filter(f => f.id === 1)) {
    if (f.wire === 0) result.push(f.number);
    else if (f.wire === 2) { let p = 0; while (p < f.data.length) { let n; [n, p] = varint(f.data, p, f.data.length); result.push(n); } }
    else throw new Error('Hibás modellsúly-méret.');
  }
  return result;
}
function rewriteHighResolutionHead(nodes, other) {
  // Linear 1x1 channel projection commutes with per-channel bilinear Resize:
  // W*[Resize(x), y] + b = Resize(Wx*x) + Wy*y + b.
  // v2 allocated 3 GiB for Resize(x), then 3.75 GiB for the concatenation.
  // Keep the same 2048px output and FP32 arithmetic without either tensor.
  const root = '/model/decoder/', prefix = 'photocraft_hr_head_v3/';
  const names = { resize: root + 'Resize_7', concat: root + 'Concat_27', conv: root + 'conv_out1/conv_out1.0/Conv', shapeA: root + 'Shape_20', shapeB: root + 'Shape_21' };
  const byName = new Map(nodes.map(n => [textField(n.fields, 3), n]));
  function required(name, op) { const n = byName.get(name); if (!n || textField(n.fields, 4) !== op) throw new Error('Eltérő HR kimeneti gráf; a memóriaoptimalizálás leállt.'); return n; }
  const resize = required(names.resize, 'Resize'), concat = required(names.concat, 'Concat'), conv = required(names.conv, 'Conv');
  const shapes = [required(names.shapeA, 'Shape'), required(names.shapeB, 'Shape')];
  const ri = strings(resize.fields, 1), ro = strings(resize.fields, 2), ci = strings(concat.fields, 1), co = strings(concat.fields, 2), wi = strings(conv.fields, 1), wo = strings(conv.fields, 2);
  const attributes = n => n.fields.filter(f => f.id === 5).map(f => fields(f.data));
  const attr = (n, name) => attributes(n).find(a => textField(a, 1) === name);
  if (ri.length !== 4 || ri[1] || ri[2] || !ri[3] || ro.length !== 1 || ci.length !== 2 || ci[0] !== ro[0] || co.length !== 1 || wi.length !== 3 || wi[0] !== co[0] || wo.length !== 1 ||
      textField(attr(resize, 'mode') ?? [], 4) !== 'linear' || textField(attr(resize, 'coordinate_transformation_mode') ?? [], 4) !== 'align_corners' ||
      attr(concat, 'axis')?.find(f => f.id === 3)?.number !== 1 ||
      shapes.some(n => strings(n.fields, 1).join(',') !== ro[0])) throw new Error('Váratlan HR kimeneti művelet.');
  const consumers = output => nodes.filter(n => strings(n.fields, 1).includes(output)).map(n => textField(n.fields, 3)).sort().join(',');
  if (consumers(ro[0]) !== [names.concat, names.shapeA, names.shapeB].sort().join(',') || consumers(co[0]) !== names.conv) throw new Error('A HR kimenetnek nem várt felhasználója van.');
  // Enforce the exact 1x1, stride-one, no-padding projection before commuting it.
  const expectedConv = { dilations: [1, 1], group: 1, kernel_shape: [1, 1], pads: [0, 0, 0, 0], strides: [1, 1] };
  if (attributes(conv).length !== 5) throw new Error('Eltérő HR vetítési paraméterek.');
  for (const [name, value] of Object.entries(expectedConv)) {
    const a = attr(conv, name);
    const actual = Array.isArray(value) ? a?.filter(f => f.id === 8).flatMap(f => {
      if (f.wire === 0) return [f.number]; let p = 0; const values = []; while (p < f.data.length) { let n; [n, p] = varint(f.data, p, f.data.length); values.push(n); } return values;
    }) : a?.find(f => f.id === 3)?.number;
    if (JSON.stringify(actual) !== JSON.stringify(value)) throw new Error('A HR vetítés nem az elvárt 1×1 művelet.');
  }
  const weight = other.find(f => f.id === 5 && textField(fields(f.data), 8) === wi[1]);
  if (!weight) throw new Error('Hiányzó HR vetítési súlyok.');
  const wf = fields(weight.data), raw = wf.find(f => f.id === 9)?.data;
  if (tensorDims(wf).join(',') !== '1,240,1,1' || wf.find(f => f.id === 2)?.number !== 1 || raw?.length !== 960) throw new Error('Eltérő HR vetítési súlyméret.');
  function addTensor(name, dims, type, payload) {
    const body = join([...dims.map(d => num(1, d)), num(2, type), str(8, name), payload]);
    other.push({ id: 5, data: body, raw: field(5, body) });
  }
  const wl = prefix + 'low_weight', wh = prefix + 'high_weight', indices = prefix + 'size_indices';
  addTensor(wl, [1, 192, 1, 1], 1, field(9, raw.subarray(0, 192 * 4)));
  addTensor(wh, [1, 48, 1, 1], 1, field(9, raw.subarray(192 * 4)));
  addTensor(indices, [4], 7, join([0, 0, 2, 3].map(i => num(7, i))));
  const low = prefix + 'low', size = prefix + 'size', up = prefix + 'upsampled', high = prefix + 'high';
  const convAttrs = conv.fields.filter(f => f.id === 5).map(f => f.raw), resizeAttrs = resize.fields.filter(f => f.id === 5).map(f => f.raw);
  const result = [];
  for (const n of nodes) {
    const name = textField(n.fields, 3);
    if (name === names.resize) {
      result.push(makeNode(prefix + 'ProjectLow', 'Conv', [ri[0], wl], [low], convAttrs));
      // The pinned graph has batch=1: gather [N,N,H,W] gives [1,1,H,W].
      result.push(makeNode(prefix + 'ProjectedSize', 'Gather', [ri[3], indices], [size], [integerAttribute('axis', 0)]));
      result.push(makeNode(prefix + 'ResizeProjected', 'Resize', [low, '', '', size], [up], resizeAttrs));
    } else if (name === names.shapeA || name === names.shapeB) {
      // Shape of the removed Resize is exactly its original sizes argument.
      result.push(makeNode(name, 'Identity', [ri[3]], strings(n.fields, 2)));
    } else if (name === names.concat) { /* Never materialize 240 full-size channels. */ }
    else if (name === names.conv) {
      result.push(makeNode(prefix + 'ProjectHigh', 'Conv', [ci[1], wh, wi[2]], [high], convAttrs));
      result.push(makeNode(prefix + 'Combine', 'Add', [up, high], wo));
    } else result.push(n);
  }
  return result;
}
function sortNodes(nodes, other) {
  // Preserve dependency order explicitly. This also rejects broken/cyclic graphs.
  const producers = new Map(), sources = new Set();
  for (const f of other) {
    if (f.id === 11) sources.add(textField(fields(f.data), 1));
    if (f.id === 5) sources.add(textField(fields(f.data), 8));
  }
  nodes.forEach((n, i) => { for (const f of n.fields) if (f.id === 2) { const name = utf8.decode(f.data); if (producers.has(name)) throw new Error('Ismétlődő ONNX kimenet.'); producers.set(name, i); } });
  // FP32 is already topologically ordered. Preserve that execution schedule:
  // breadth-first reordering needlessly extends activation lifetimes.
  const available = new Set(sources); let ordered = true;
  for (const n of nodes) {
    if (strings(n.fields, 1).some(name => name && !available.has(name))) { ordered = false; break; }
    for (const name of strings(n.fields, 2)) available.add(name);
  }
  if (ordered) return nodes.map(n => n.raw);
  const counts = new Uint32Array(nodes.length), edges = Array.from({ length: nodes.length }, () => []), ready = [];
  nodes.forEach((n, i) => {
    const dependencies = new Set();
    for (const f of n.fields) if (f.id === 1) {
      const name = utf8.decode(f.data); if (!name || sources.has(name)) continue;
      const dependency = producers.get(name); if (dependency === undefined) throw new Error('Hiányzó ONNX bemenet.'); dependencies.add(dependency);
    }
    counts[i] = dependencies.size;
    for (const d of dependencies) edges[d].push(i);
    if (!counts[i]) ready.push(i);
  });
  const sorted = [];
  for (let head = 0; head < ready.length; head++) { const i = ready[head]; sorted.push(nodes[i].raw); for (const target of edges[i]) if (--counts[target] === 0) ready.push(target); }
  if (sorted.length !== nodes.length) throw new Error('Körkörös ONNX gráf.');
  return sorted;
}
export function adaptPinnedModel(bytes) {
  const top = fields(bytes), graph = top.find(f => f.id === 7);
  if (!graph) throw new Error('Az ONNX gráf hiányzik.');
  const imports = top.filter(f => f.id === 8).map(f => fields(f.data));
  if (imports.length !== 1 || textField(imports[0], 1) !== '' || imports[0].find(f => f.id === 2)?.number !== 20) throw new Error('Nem a rögzített HR modell érkezett.');
  const nodes = [], other = []; let grids = 0, gelus = 0;
  for (const f of fields(graph.data)) {
    if (f.id !== 1) { other.push(f); continue; }
    const n = fields(f.data), op = textField(n, 4);
    if (op === 'GridSample') {
      let changed = false;
      const updated = n.map(a => {
        if (a.id !== 5) return a.raw;
        const attrs = fields(a.data);
        if (textField(attrs, 1) !== 'mode') return a.raw;
        if (textField(attrs, 4) !== 'linear') throw new Error('Váratlan GridSample mód.');
        changed = true; return field(5, join(attrs.map(v => v.id === 4 ? str(4, 'bilinear') : v.raw)));
      });
      if (!changed) throw new Error('Hiányzó GridSample mód.');
      nodes.push({ raw: field(1, join(updated)), fields: n }); grids++;
    } else if (op === 'Gelu') {
      const attrs = n.filter(a => a.id === 5).map(a => fields(a.data));
      if (attrs.length !== 1 || textField(attrs[0], 1) !== 'approximate' || textField(attrs[0], 4) !== 'none') throw new Error('Váratlan GELU közelítés.');
      nodes.push({ raw: field(1, join([...n.filter(a => a.id !== 5 && a.id !== 7).map(a => a.raw), str(7, 'com.microsoft')])), fields: n }); gelus++;
    } else nodes.push({ raw: f.raw, fields: n });
  }
  if (grids !== 300 || gelus !== 48) throw new Error('Eltérő modellváltozat; a böngészős illesztés leállt.');
  const compactNodes = rewriteHighResolutionHead(nodes, other);
  const parts = [...sortNodes(compactNodes, other), ...other.map(f => f.raw)];
  // Keep the large graph as chunks until the final allocation (do not decode tensors).
  const length = parts.reduce((n, p) => n + p.length, 0), result = [];
  for (const f of top) {
    if (f.id === 7) result.push(vint(7 * 8 + 2), vint(length), ...parts);
    else if (f.id === 8) result.push(field(8, join([str(1, ''), num(2, 19)])), field(8, join([str(1, 'com.microsoft'), num(2, 1)])));
    else result.push(f.raw);
  }
  return join(result);
}
