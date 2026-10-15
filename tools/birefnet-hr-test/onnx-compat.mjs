// Narrow, zero-copy protobuf traversal for the pinned PinkPixel FP32 export only.
// ORT WebGPU 1.23.2 supports GridSample 16-19 (2D bilinear), not opset 20
// (renames this mode 'linear'). Map exact GELU to the equivalent Microsoft op.
// Initializers are retained byte-for-byte. No weights or inference precision change.
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
function sortNodes(nodes, other) {
  // Preserve dependency order explicitly. This also rejects broken/cyclic graphs.
  const producers = new Map(), sources = new Set();
  for (const f of other) {
    if (f.id === 11) sources.add(textField(fields(f.data), 1));
    if (f.id === 5) sources.add(textField(fields(f.data), 8));
  }
  nodes.forEach((n, i) => { for (const f of n.fields) if (f.id === 2) { const name = utf8.decode(f.data); if (producers.has(name)) throw new Error('Ismétlődő ONNX kimenet.'); producers.set(name, i); } });
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
  const parts = [...sortNodes(nodes, other), ...other.map(f => f.raw)];
  // Keep the large graph as chunks until the final allocation (do not decode tensors).
  const length = parts.reduce((n, p) => n + p.length, 0), result = [];
  for (const f of top) {
    if (f.id === 7) result.push(vint(7 * 8 + 2), vint(length), ...parts);
    else if (f.id === 8) result.push(field(8, join([str(1, ''), num(2, 19)])), field(8, join([str(1, 'com.microsoft'), num(2, 1)])));
    else result.push(f.raw);
  }
  return join(result);
}
