export function gpuLimits(limits) {
  const result = {};
  for (const key of ['maxBufferSize', 'maxStorageBufferBindingSize', 'maxComputeWorkgroupsPerDimension', 'maxComputeInvocationsPerWorkgroup']) {
    try { const value = limits?.[key]; if (Number.isFinite(value)) result[key] = value; } catch { /* Diagnostics must not hide the original error. */ }
  }
  return result;
}
export function classifyGpuError(error) {
  const detail = String(error?.message ?? error).slice(0, 4000);
  const match = /Buffer size\s*\((\d+)\)\s*exceeds the max buffer size limit\s*\((\d+)\)/i.exec(detail);
  if (match) {
    const requested = Number(match[1]), limit = Number(match[2]);
    const gib = value => (value / 1073741824).toFixed(2).replace('.', ',');
    return { code: 'GPU_BUFFER_LIMIT', detail, buffer: { requested, limit }, message: `A böngésző egyetlen GPU-pufferre vonatkozó korlátjába ütközött a művelet: ${gib(requested)} GiB szükséges, a határ ${gib(limit)} GiB. Ez nem a videokártya teljes memóriájának elfogyását jelenti. Töltsd le a hibajelentést.` };
  }
  if (/maxStorageBufferBindingSize|storage buffer.*(?:exceeds|limit)|max buffer size/i.test(detail)) return { code: 'GPU_BUFFER_LIMIT', detail, message: 'A böngésző egy GPU-puffer méretkorlátja miatt állította le a számítást. A videokártya teljes memóriája ettől eltérő érték. Töltsd le a hibajelentést.' };
  if (/device.*lost|GPUDevice.*destroyed/i.test(detail)) return { code: 'GPU_DEVICE_LOST', detail, message: 'A GPU-kapcsolat megszakadt. Frissítsd az oldalt, és ha ismétlődik, töltsd le a hibajelentést.' };
  if (/out of memory|out of.*memory|allocation|ResourceExhausted|memory access out of bounds/i.test(detail)) return { code: 'MEMORY_ALLOCATION_FAILED', detail, message: 'Nem sikerült memóriát lefoglalni a feldolgozáshoz. Ez a böngésző vagy a GPU memóriáját is érintheti. Zárd be a nagy memóriaigényű lapokat, és ha ismétlődik, töltsd le a hibajelentést.' };
  return { code: error?.code ?? 'INFERENCE_ERROR', detail, message: detail };
}
