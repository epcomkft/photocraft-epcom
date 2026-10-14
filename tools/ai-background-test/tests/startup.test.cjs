const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), path = require('node:path'), vm = require('node:vm');
const { JSDOM } = require('jsdom');
const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
function environment() {
  const dom = new JSDOM(html, { runScripts: 'outside-only' });
  const timers = [];
  dom.window.setTimeout = (fn) => { timers.push(fn); return timers.length; };
  dom.window.eval(dom.window.document.getElementById('startup-guard').textContent);
  return { dom, timers, document: dom.window.document };
}

test('deployment requires no .mjs module requests or host MIME configuration', () => {
  const dom = new JSDOM(html), doc = dom.window.document;
  const script = doc.getElementById('app-script');
  assert.equal(script.getAttribute('src'), './app.js?v=6');
  assert.notEqual(script.getAttribute('type'), 'module');
  assert.equal(script.defer, true);
  // Parse the deployable file as a classic script; top-level imports/exports fail.
  assert.doesNotThrow(() => new vm.Script(fs.readFileSync(path.join(root, 'app.js'), 'utf8')));
  assert.deepEqual([...doc.querySelectorAll('script[src]')].map((x) => x.getAttribute('src')), ['./app.js?v=6']);
  dom.window.close();
});
test('missing script reports an actionable startup error instead of inert file selection', () => {
  const { dom, document } = environment();
  document.getElementById('app-script').dispatchEvent(new dom.window.Event('error'));
  assert.match(document.getElementById('status').textContent, /app\.js/);
  assert.equal(document.getElementById('file').disabled, true);
  assert.equal(document.getElementById('run').disabled, true);
  dom.window.close();
});
test('a stalled startup reports an error, and a late successful app can still initialize', () => {
  const { dom, document, timers } = environment();
  timers[0]();
  assert.match(document.getElementById('status').textContent, /nem indult el/);
  // The guard never cancels the script request or installs a permanent lock.
  document.documentElement.dataset.appReady = '6';
  document.getElementById('file').disabled = false;
  document.getElementById('status').textContent = 'Válassz egy képet.';
  timers[0]();
  assert.equal(document.getElementById('file').disabled, false);
  assert.equal(document.getElementById('status').textContent, 'Válassz egy képet.');
  dom.window.close();
});
test('post-startup model or application errors do not overwrite the live UI with a boot failure', () => {
  const { dom, document, timers } = environment();
  document.documentElement.dataset.appReady = '6';
  document.getElementById('file').disabled = false;
  document.getElementById('status').textContent = 'Kész.';
  dom.window.dispatchEvent(new dom.window.ErrorEvent('error', { message: 'model error' }));
  timers[0]();
  assert.equal(document.getElementById('status').textContent, 'Kész.');
  assert.equal(document.getElementById('file').disabled, false);
  dom.window.close();
});
