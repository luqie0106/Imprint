import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

const source = await readFile(new URL("../src/stores/previewCache.ts", import.meta.url), "utf8");
const compiled = ts.transpile(source, { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 });
const cache = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

function request(overrides = {}) {
  return {
    session_id: "session-1",
    photo_id: "photo-1",
    algorithm: "physical",
    params: { strength: 0.4, local_contrast: 0.2 },
    auto_mode: false,
    auto_exposure: false,
    nonlocal_mode: "off",
    basic_params: { exposure: 0, contrast: 3 },
    ricoh_preset_id: "street",
    max_edge: 1800,
    preview_level: 0,
    full_resolution: true,
    mode: "dehazed",
    color_manage_srgb: true,
    render_backend: "auto",
    basic_backend: "auto",
    ricoh_backend: "auto",
    preview_client_id: "client-a",
    preview_generation: 1,
    ...overrides,
  };
}

function response(status = 200, headers = {}) {
  return { ok: status >= 200 && status < 300, status, headers: new Headers(headers) };
}

test("same rendering request across pages ignores client and generation identity", () => {
  cache.clearPreviewCache();
  const first = request();
  const second = request({ preview_client_id: "client-b", preview_generation: 42 });
  assert.equal(cache.previewCacheKey("http://127.0.0.1:9132", first),
    cache.previewCacheKey("http://127.0.0.1:9132", second));

  const nestedOrderVariant = {
    ...second,
    params: { local_contrast: 0.2, strength: 0.4 },
    basic_params: { contrast: 3, exposure: 0 },
  };
  assert.equal(cache.previewCacheKey("http://127.0.0.1:9132", first),
    cache.previewCacheKey("http://127.0.0.1:9132", nestedOrderVariant));
});

test("parameter and preview-resolution changes have separate entries", () => {
  cache.clearPreviewCache();
  const body = request();
  const blob = { size: 3, marker: "effect" };
  cache.cachePreviewResponse("http://127.0.0.1:9132", body,
    response(200, { "X-Image-Width": "2400" }), blob);

  assert.equal(cache.getCachedPreview("http://127.0.0.1:9132", request({
    params: { strength: 0.5, local_contrast: 0.2 },
  })), undefined);
  assert.equal(cache.getCachedPreview("http://127.0.0.1:9132", request({
    preview_level: 2, full_resolution: false,
  })), undefined);
  assert.equal(cache.getCachedPreview("http://127.0.0.1:9132", body)?.blob, blob);
});

test("session and service identity isolate cached previews", () => {
  cache.clearPreviewCache();
  const body = request();
  cache.cachePreviewResponse("http://127.0.0.1:9132", body, response(), { size: 1 });
  assert.equal(cache.getCachedPreview("http://127.0.0.1:9132", request({ session_id: "session-2" })), undefined);
  assert.equal(cache.getCachedPreview("http://127.0.0.1:9133", body), undefined);
});

test("failed and 204 responses are never cached and successful headers are preserved", () => {
  cache.clearPreviewCache();
  const body = request();
  assert.equal(cache.cachePreviewResponse("http://127.0.0.1:9132", body,
    response(500, { "X-Image-Width": "999" }), { size: 2 }), false);
  assert.equal(cache.cachePreviewResponse("http://127.0.0.1:9132", body,
    response(204), { size: 2 }), false);
  assert.equal(cache.getCachedPreview("http://127.0.0.1:9132", body), undefined);

  const blob = { size: 2 };
  cache.cachePreviewResponse("http://127.0.0.1:9132", body,
    response(200, { "X-Image-Width": "2400", "X-Dehaze-Nonlocal-Status": "active" }), blob);
  const cached = cache.getCachedPreview("http://127.0.0.1:9132", body);
  assert.equal(cached?.blob, blob);
  assert.deepEqual(Object.fromEntries(cached.headers), {
    "x-dehaze-nonlocal-status": "active",
    "x-image-width": "2400",
  });
  cached.headers[0][1] = "mutated";
  assert.equal(Object.fromEntries(cache.getCachedPreview("http://127.0.0.1:9132", body).headers)["x-image-width"], "2400");
});

test("LRU eviction enforces both byte and entry limits", () => {
  cache.clearPreviewCache();
  const largeFirst = request({ photo_id: "large-first" });
  const largeSecond = request({ photo_id: "large-second" });
  cache.cachePreviewResponse("http://127.0.0.1:9132", largeFirst, response(), { size: 40 * 1024 * 1024 });
  cache.cachePreviewResponse("http://127.0.0.1:9132", largeSecond, response(), { size: 30 * 1024 * 1024 });
  assert.equal(cache.getCachedPreview("http://127.0.0.1:9132", largeFirst), undefined);
  assert.equal(cache.previewCacheStats().bytes, 30 * 1024 * 1024);

  cache.clearPreviewCache();
  for (let index = 0; index < cache.PREVIEW_CACHE_LIMIT_ENTRIES + 1; index++) {
    cache.cachePreviewResponse("http://127.0.0.1:9132", request({ photo_id: `photo-${index}` }), response(), { size: 1 });
  }
  assert.equal(cache.previewCacheStats().entries, cache.PREVIEW_CACHE_LIMIT_ENTRIES);
  assert.equal(cache.getCachedPreview("http://127.0.0.1:9132", request({ photo_id: "photo-0" })), undefined);
  assert.equal(cache.previewCacheStats().bytes, cache.PREVIEW_CACHE_LIMIT_ENTRIES);
});

test("a response larger than the cache budget is left uncached", () => {
  cache.clearPreviewCache();
  const body = request();
  assert.equal(cache.cachePreviewResponse("http://127.0.0.1:9132", body, response(), {
    size: cache.PREVIEW_CACHE_LIMIT_BYTES + 1,
  }), false);
  assert.equal(cache.previewCacheStats().entries, 0);
});

test("decoded images are shared by blob while URL ownership stays independent", () => {
  cache.clearPreviewCache();
  const blob = new Blob(["jpeg"]);
  const a = cache.createPreviewUrl(blob);
  const b = cache.createPreviewUrl(blob);
  const image = { naturalWidth: 2400, naturalHeight: 3600 };
  cache.rememberDecodedPreview(a, image);
  assert.equal(cache.getDecodedPreview(b), image);
  cache.releasePreviewUrl(a);
  assert.equal(cache.getDecodedPreview(a), undefined);
  assert.equal(cache.getDecodedPreview(b), image);
  cache.releasePreviewUrl(b);
});

test("decoded images have a separate memory budget", () => {
  cache.clearPreviewCache();
  const urls = [cache.createPreviewUrl(new Blob(['a'])), cache.createPreviewUrl(new Blob(['b']))];
  cache.rememberDecodedPreview(urls[0], { naturalWidth: 6000, naturalHeight: 6000 });
  const latest = { naturalWidth: 6000, naturalHeight: 6000 };
  cache.rememberDecodedPreview(urls[1], latest);
  assert.equal(cache.getDecodedPreview(urls[0]), undefined);
  assert.equal(cache.getDecodedPreview(urls[1]), latest);
  urls.forEach(cache.releasePreviewUrl);
});
