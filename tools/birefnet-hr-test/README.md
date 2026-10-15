# PhotoCraft · BiRefNet HR-matting teszt v4 · FP32

Külön telepíthető, böngészőben működő összehasonlító változat. Az eredeti BEN2 / ViTMatte teszt mappájától függetlenül működik.

## Frissítés v3-ról

A v4 célzottan a haj és szőr körüli háttérszínt csökkenti. Több léptékű előtér/háttér színbecslést használ, és a maradék színperemet a helyi színkeverék illeszkedése alapján korrigálja. **Az alfamaszk és a teljesen fedett képrészek változatlanok maradnak.** Ez nem hoz vissza a modell által elhagyott hajszálakat, és nem garantálja minden háttérmaradvány eltűnését. Erős beállításnál a szegély színe módosulhat; fehér és fekete háttér előtt is ellenőrizd.

- A **Peremtisztítás csúszkája mindig látható**, közvetlenül a feldolgozás alatt. Csak a **Hibajelentés és diagnosztika** nyitható le.
- A 0%, 30%, 60%, 100% gyorsgombokkal egy kattintással összehasonlíthatod a hatást. A 0% pontosan visszaadja a nyers színeket.
- A csúszka mozgatására automatikusan frissül az előnézet; az utolsó beállítás érvényesül gyors változtatáskor is. Az AI nem fut újra: az első színbecslés után a számított színek újra használhatók.
- Az export neve a verziót és a tényleges erősséget is tartalmazza, például `kep-birefnet-hr-v4-tisztitott-60pct.png`. Amíg a választott érték feldolgozása tart, az export nem engedélyezett.
- A már átlátszó bemeneti PNG színeit megőrizzük; az eredeti átlátszósága továbbra is szorzódik a modell maszkjával. Az ilyen bemenetnél a felület jelzi, hogy a színkorrekció nem alkalmazható.

**Telepítés:** a ZIP összes fájljával írd felül a tesztmappát, majd Ctrl+Shift+R. A fejlécben **v4 · FP32** jelenjen meg. A korábbi FP32 modellgyorsítótárat **nem kell törölni**; a modell, annak ellenőrzőösszege és a v3 GPU-pufferjavítása változatlan.

Ha további hiba történik, a **Hibajelentés és diagnosztika → Hibajelentés letöltése** gombbal mentsd a JSON-t. Tartalmazza a GPU pufferkorlátait és a színkorrekció állapotát is. A fotó és a fájlneve nem kerül bele; semmit nem küldünk el automatikusan.

## Telepítés

1. Hozd létre az `admin/photocraft-birefnet-test/` mappát a tárhelyen.
2. A ZIP-ben lévő `photocraft-birefnet-test` mappa tartalmát töltsd bele. A működéshez az **index.html, app.js, worker.js és style.css** kell, ugyanabban a mappában. A `licenses` mappát és ezt a leírást is tartsd meg.
3. Nyisd meg: `https://epcom.hu/admin/photocraft-birefnet-test/`.
4. Asztali Chrome, Edge vagy Opera böngészőben, bekapcsolt hardveres gyorsítással teszteld. A videokártyának WebGPU támogatás kell. A v4 sem igényel `shader-f16` támogatást.

A tárhelyen nem kell Node.js, Python, telepítő, cron, API-kulcs vagy GPU. A számítás a saját géped videokártyáján fut. A kép nem kerül külső képfeldolgozó API-ra. A modell a Hugging Face-ről, a futtatókörnyezet a jsDelivr-ről töltődik le; ezekhez internet kell.

Ez önálló HTML tesztoldal, nem használja a PHP admin bejelentkezését, és nem fér hozzá az admin adataihoz. A mappát a tárhely saját könyvtárvédelmével is védheted, ha szükséges.

## Kipróbálás

1. Válaszd ki **ugyanazt az eredeti fotót**, amellyel a korábbi változatot tesztelted.
2. Hagyd a peremtisztítást **0%-on**, és kattints a **Háttér eltávolítása** gombra.
3. Első alkalommal **932 150 975 bájt, kb. 932 MB** modell töltődik le. A letöltés állapota látható. Ha van elég hely és a böngésző engedi, a modell gyorsítótárba kerül.
4. A **Korábbi eredmény összehasonlításhoz** gombbal nyisd meg a BEN2 vagy a Photoshop kivágását. Az összehasonlító képnek pontosan az eredeti méretűnek kell lennie; PNG javasolt, mert JPG nem tartalmaz átlátszóságot.
5. Váltogasd a fehér, fekete, szürke hátteret és az alfamaszkot. 100% vagy 200% nagyításnál a képek görgetése együtt mozog.
6. Ha színperem marad, a közvetlenül látható **Peremtisztítás** csúszkával vagy gyorsgombokkal emeld az erősséget. Ez a kép szélén a háttérből visszamaradt színt mérsékli; nem módosítja az alfa alakját. A nyers eredmény bármikor visszaválasztható.
7. Az **Átlátszó PNG letöltése** mindig a kiválasztott nyers vagy peremtisztított képet exportálja, az eredeti méretben. Az előnézeti háttér nem kerül bele. Az alfamaszk külön szürkeárnyalatos PNG-ként is letölthető.

**Ez összehasonlító teszt, nem ígéret a Photoshopnál jobb eredményre.** Hajon, állatszőrön és bajuszon, azonos eredeti képekkel érdemes eldönteni, hogy beválik-e.

## Ha nem indul vagy elfogy a memória

- „A program nem indult el”: mind a négy futtatási fájlt másold fel; utána Ctrl+Shift+R, Macen Cmd+Shift+R.
- A feldolgozó nem indul: ellenőrizd a `worker.js` feltöltését. A tárhely a `.js` fájlokat JavaScriptként szolgálja ki. Nem kell `.mjs` támogatást beállítani.
- „WebGPU nem érhető el”: HTTPS szükséges; ellenőrizd a hardveres gyorsítást és a videokártya illesztőprogramját.
- Memóriahiba: ez a HR modell lényegesen nehezebb a BEN2-nél, az FP32 pedig több memóriát igényel a v1 FP16 változatánál. Zárd be a GPU-t használó más alkalmazásokat és lapokat, majd próbáld újra. A kisebb feltöltött kép önmagában nem csökkenti a hálózat 2048 × 2048 pixeles bemenetét. Garantált VRAM-minimumot nem mértünk.
- A **Megszakítás** az egész feldolgozó workert leállítja; az eredeti kép megmarad.
- Letöltési hiba esetén az oldal alján törölhető a modellgyorsítótár. Ha a böngésző tiltja vagy betelt a tárhelye, a modell gyorsítótár nélkül is feldolgozható.
- Támogatott bemenet: JPG, PNG, WebP, legfeljebb 40 MB, 16 megapixel és oldalanként 8192 pixel. A bemenet a böngésző 8 bites RGBA képútját használja; ez nem HDR/CMYK vagy 16 bites fotólabor.

## Pontos modell és feldolgozás

- Eredeti modell: [ZhengPeng7/BiRefNet_HR-matting](https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting), MIT licenc.
- ONNX export: [PinkPixel/birefnet-hr-matting-onnx](https://huggingface.co/PinkPixel/birefnet-hr-matting-onnx), `model.onnx` (valódi FP32 referencia, nem az FP16 súlyok visszakonvertálása).
- Rögzített exportrevízió: `792518b9d4dfe9793891712677de25e554f948c6`.
- SHA-256: `a548a1f3307250766bef50b174026b3cfa39a695bc1e42ec131c14fe82797ffd`. A program futás előtt ellenőrzi.
- ONNX Runtime Web: `1.23.2`, WebGPU, külön classic Web Workerben; egyszálú WASM segédruntime, COOP/COEP fejléc nélkül.
- A teljes kép 2048 × 2048 pixelre méretezve, ImageNet RGB normalizálással kerül a modellbe. FP32 súlyok, számítás, bemenet és kimenet; `basic` gráfoptimalizálás. A kimenet már Sigmoid aktiválást tartalmaz, nem alkalmazzuk újra.
- Az alpha fokozatos marad, nem vágjuk bináris maszkká. Nem fut rajta automatikus trimap-zárolás, foltszűrés vagy újabb ViTMatte. Ez lehetővé teszi a modell önálló megítélését.
- A maszkot az eredeti képmérethez interpoláljuk, és összeszorozzuk az eredeti PNG alfájával. Opcionális szegélyszín-korrekció alkalmazható. Ennek alapja a [Fast Multi-Level Foreground Estimation](https://arxiv.org/abs/2006.14970) módszer [PyMatting-implementációjának](https://github.com/pymatting/pymatting/blob/master/pymatting/foreground/estimate_foreground_ml.py) JavaScript-adaptációja (MIT), helyi színkeverék-ellenőrzéssel kiegészítve. A becslés eredeti képméretben, a Web Workerben fut; nincs új futtatási függőség vagy külső képfeldolgozó szolgáltatás.
- A böngészős kompatibilitási adapter az export 300 darab 2D `GridSample` műveletét az azonos bilineáris opset-19 formára alakítja, és 48 pontos GELU műveletet az ONNX Runtime azonos `com.microsoft::Gelu` műveletéhez rendel. Ellenőrzi a függőségeket, és topológiailag rendezi a gráfot. **A modellsúlyok bájtról bájtra változatlanok.** Eltérő modellrevíziót vagy nem várt gráfot elutasít. A végső 1×1 vetítés súlyából két új, bájtpontos csatornaszeletet képez, és egy apró indexkonstanst ad hozzá. A már érvényes eredeti műveleti sorrendet megőrzi.

## Fejlesztés és ellenőrzés

Az `app.js` kézzel szerkesztett classic felületkód. A `worker.js` a `core.mjs`, `foreground.mjs`, `edge-colors.mjs`, `onnx-compat.mjs`, `gpu-errors.mjs`, `worker-source.mjs` állományokból generált classic csomag. A tárhelyen csak a generált `.js` fájlok töltődnek be; a forrás `.mjs` fájlokat nem szükséges feltölteni.

```sh
node build.mjs
node build.mjs --check
node --test tests/core.test.mjs tests/worker.test.mjs tests/gpu-errors.test.mjs tests/edge-colors.test.mjs
# onnx, onnxruntime, numpy tesztfüggőségekkel:
python tests/head.test.py
# Playwright és @napi-rs/canvas tesztfüggőségekkel:
BROWSER_EXECUTABLE=/path/to/chromium node tests/browser.cjs
```

A tesztek pontos hatóköre és a GPU-futás korlátai: `tests/VALIDATION.md`. A ZIP képeket, API-kulcsokat és modellsúlyokat nem tartalmaz.
