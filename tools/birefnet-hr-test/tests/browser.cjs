// Browser UI regression. Model outputs are mocked; this is NOT a GPU quality test.
// Usage: NODE_PATH=... BROWSER_EXECUTABLE=/path/to/chromium node tests/browser.cjs
const assert = require('node:assert/strict'), fs = require('node:fs'), path = require('node:path'), http = require('node:http');
const { chromium } = require('playwright'), { createCanvas } = require('@napi-rs/canvas');
const root = path.resolve(__dirname, '..');
const fixture = (w, h) => { const c = createCanvas(w,h), x = c.getContext('2d'); x.fillStyle='#eee2ce';x.fillRect(0,0,w,h);x.fillStyle='#68432b';x.beginPath();x.arc(w/2,h/2,Math.min(w,h)/3,0,Math.PI*2);x.fill();return c.toBuffer('image/png'); };
const mock = `let src;self.postMessage({type:'ready',version:'4.0.0'});self.onmessage=({data:m})=>{if(m.type==='run')src=m;setTimeout(()=>{const n=src.width*src.height;const data=new Uint8ClampedArray(src.rgba.slice(0)), a=new Uint8ClampedArray(n);for(let i=0;i<n;i++){a[i]=i%src.width<src.width/2?0:180;data[i*4+3]=a[i];}self.postMessage({type:'metrics',id:m.id,seconds:.5,model:'MOCK TEST ONLY',size:2048,precision:'FP16',revision:'fixture',backend:'mock'});self.postMessage({type:'result',id:m.id,width:src.width,height:src.height,raw:m.type==='run'?data.buffer.slice(0):undefined,rgba:data.buffer,alpha:a.buffer,strength:m.strength});},350);};`;
const server=http.createServer((req,res)=>{const file=path.join(root,req.url.split('?')[0]==='/'?'index.html':req.url.split('?')[0]);if(!file.startsWith(root+path.sep)||!fs.existsSync(file)){res.statusCode=404;res.end();return;}res.setHeader('Content-Type',file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':'text/html');fs.createReadStream(file).pipe(res);});
(async()=>{
  await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
  try{
    browser=await chromium.launch({executablePath:process.env.BROWSER_EXECUTABLE,headless:true,args:['--no-sandbox','--disable-dev-shm-usage']});
    const page=await browser.newPage({viewport:{width:1440,height:1100},acceptDownloads:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/worker.js*',r=>r.fulfill({contentType:'application/javascript',body:mock}));
    await page.goto(`http://127.0.0.1:${server.address().port}/`);
    await page.waitForFunction(()=>document.documentElement.dataset.appReady==='1');
    assert.equal(await page.locator('#run').isDisabled(),true);assert.equal(await page.locator('#file').isDisabled(),false);
    assert.equal(await page.locator('#strength').isVisible(),true);assert.equal(await page.locator('#diagnosticPanel').getAttribute('open'),null);assert.equal(await page.locator('#strength').evaluate(el=>!!el.closest('details')),false);
    await page.locator('#file').setInputFiles({name:'teszt.png',mimeType:'image/png',buffer:fixture(800,600)});
    await page.waitForFunction(()=>!document.getElementById('run').disabled);assert.equal(await page.locator('#dimensions').textContent(),'800 × 600 px');
    await page.locator('#run').click();await page.waitForFunction(()=>!document.getElementById('download').disabled);
    assert.deepEqual(await page.locator('#result').evaluate(c=>[c.width,c.height]),[800,600]);
    await page.locator('#background').selectOption('black');await page.locator('#zoom').selectOption('2');assert.equal(await page.locator('#result').evaluate(c=>c.parentElement.style.width),'1600px');
    await page.locator('#referenceFile').setInputFiles({name:'old.png',mimeType:'image/png',buffer:fixture(800,600)});await page.waitForFunction(()=>!document.getElementById('referenceCard').hidden);
    await page.locator('#referenceFile').setInputFiles({name:'wrong.png',mimeType:'image/png',buffer:fixture(100,100)});await page.waitForFunction(()=>document.getElementById('status').textContent.includes('ugyanakkora'));
    await page.locator('#strength').evaluate(el=>{el.value='60';el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));});
    await page.waitForFunction(()=>!document.getElementById('download').disabled&&document.getElementById('resultTitle').textContent.includes('60%'));
    assert.equal(await page.locator('#diagnosticPanel').getAttribute('open'),null);
    const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#download').click()]);assert.equal(download.suggestedFilename(),'teszt-birefnet-hr-v4-tisztitott-60pct.png');
    const png=fs.readFileSync(await download.path());assert.equal(png.readUInt32BE(16),800);assert.equal(png.readUInt32BE(20),600);
    // Keep the slider interactive during composition and apply the latest queued value.
    await page.locator('[data-strength="30"]').click();
    assert.equal(await page.locator('#strength').isEnabled(),true);
    await page.locator('#strength').evaluate(el=>{el.value='20';el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));el.value='60';el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));});
    await page.waitForFunction(()=>!document.getElementById('download').disabled&&document.getElementById('resultTitle').textContent.includes('60%'));
    assert.equal(await page.locator('#strengthValue').textContent(),'60%');
    assert.equal(await page.locator('[data-strength="60"]').getAttribute('aria-pressed'),'true');
    await page.locator('#zoom').selectOption('fit');await page.locator('#removeReference').click();
    const shots=process.env.QA_OUTPUT_DIR;if(shots){fs.mkdirSync(shots,{recursive:true});await page.screenshot({path:path.join(shots,'desktop.png'),fullPage:true});}
    await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    if(shots)await page.screenshot({path:path.join(shots,'mobile.png'),fullPage:true});
    await page.locator('#run').click();await page.locator('#cancel').click();await page.waitForTimeout(500);assert.match(await page.locator('#status').textContent(),/Megszakítva/);assert.equal(await page.locator('#run').isEnabled(),true);
    await page.locator('#file').setInputFiles({name:'bad.txt',mimeType:'text/plain',buffer:Buffer.from('not an image')});await page.waitForFunction(()=>document.getElementById('status').textContent.includes('JPG'));
    assert.equal(await page.locator('#run').isDisabled(),true);assert.equal(await page.locator('#download').isDisabled(),true);
    // Simulate a GPU numerical failure after a prior successful result. Check
    // that the old mask is not still offered for download, and save diagnostics.
    await page.setViewportSize({width:1440,height:1100});
    await page.locator('#file').setInputFiles({name:'private-photo.png',mimeType:'image/png',buffer:fixture(800,600)});
    await page.waitForFunction(()=>!document.getElementById('run').disabled);
    await page.locator('#run').click();await page.waitForFunction(()=>!document.getElementById('download').disabled);
    await page.locator('#run').click();await page.locator('#cancel').click();
    assert.equal(await page.locator('#download').isDisabled(),true);
    await page.route('**/worker.js*',r=>r.fulfill({contentType:'application/javascript',body:`self.postMessage({type:'ready',version:'4.0.0'});self.onmessage=({data:m})=>{self.postMessage({type:'diagnostics',id:m.id,report:{appVersion:'4.0.0',precision:'FP32',error:{code:'MODEL_ALPHA_INVALID',alphaStats:{nan:4194304}}}});self.postMessage({type:'error',id:m.id,message:'A GPU számítása érvénytelen alfamaszkot adott.',detail:'fixture NaN failure'});};`}));
    await page.locator('#run').click();await page.waitForFunction(()=>document.getElementById('status').textContent.includes('érvénytelen'));
    assert.equal(await page.locator('#download').isDisabled(),true);assert.equal(await page.locator('#downloadMask').isDisabled(),true);
    assert.equal(await page.locator('#run').isEnabled(),true);assert.equal(await page.locator('#file').isEnabled(),true);
    assert.equal(await page.locator('#resultEmpty').isVisible(),true);
    await page.locator('summary').click();
    const [report]=await Promise.all([page.waitForEvent('download'),page.locator('#downloadReport').click()]);
    assert.equal(report.suggestedFilename(),'photocraft-birefnet-v4-hibajelentes.json');
    const reportText=fs.readFileSync(await report.path(),'utf8');assert.equal(JSON.parse(reportText).error.alphaStats.nan,4194304);assert.equal(reportText.includes('private-photo'),false);
    if(shots)await page.screenshot({path:path.join(shots,'error-report.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    if(shots)await page.screenshot({path:path.join(shots,'error-mobile.png'),fullPage:true});
    await page.route('**/worker.js*',r=>r.fulfill({contentType:'application/javascript',body:"self.postMessage({type:'ready',version:'2.0.0'});"}));
    await page.locator('#run').click();await page.waitForFunction(()=>document.getElementById('status').textContent.includes('Eltérő verziójú'));
    assert.equal(await page.locator('#download').isDisabled(),true);assert.equal(await page.locator('#run').isEnabled(),true);
    assert.deepEqual(errors,[]);
    const failed=await browser.newPage();await failed.route('**/app.js*',r=>r.abort());await failed.goto(`http://127.0.0.1:${server.address().port}/`);await failed.waitForFunction(()=>document.getElementById('status').textContent.includes('nem indult'));assert.equal(await failed.locator('#file').isDisabled(),true);
    console.log('PASS: browser startup, upload, result, preview, reference validation, cleanup, PNG export, responsive layout, cancellation, invalid upload, missing app.js, GPU-error retry, no stale export, JSON report, stale worker version rejection, visible cleanup slider, rapid slider updates, percentage export names. GPU outputs mocked.');
  }finally{await browser?.close();server.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
