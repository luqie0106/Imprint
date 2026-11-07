import assert from "node:assert/strict";
import test from "node:test";
import { deferred, loadFunctions, readSource } from "./sourceHarness.mjs";

const source = readSource("src/stores/previewCache.ts").replace(/^export /gm, "");
const baseUrl = "http://local-engine";
const body = { session_id: "session-a", photo_id: "photo-a", mode: "dehazed", full_resolution: true };

function createCache(fetch = async () => ({ ok: true, json: async () => ({ version: "v1" }) })) {
  const revoked = [];
  let urlId = 0;
  const scope = {
    Blob, Headers, DOMException, fetch,
    URL: { createObjectURL: () => `blob:preview-${++urlId}`, revokeObjectURL: url => revoked.push(url) },
  };
  const cache = loadFunctions(source, [
    "previewCacheKey", "getValidatedCachedPreview", "hasCachedPreview", "cachePreviewResponse",
    "clearPreviewCache", "previewCacheStats", "createPreviewUrl", "releasePreviewUrl",
    "getDecodedPreview", "rememberDecodedPreview", "getPreviewHistogram", "rememberPreviewHistogram",
  ], scope);
  function add(request = body, version = "v1", base = baseUrl) {
    const blob = new Blob(["preview"]);
    const response = { ok: true, status: 200, headers: new Headers({ "X-Source-Version": version }) };
    assert.equal(cache.cachePreviewResponse(base, request, response, blob), true);
    return blob;
  }
  return { cache, scope, add, revoked };
}

test("empty cache makes no source-version request", async () => {
  const { cache } = createCache(() => { throw new Error("Unexpected request"); });
  assert.equal(await cache.getValidatedCachedPreview(baseUrl, body), undefined);
});

test("matching source version reuses the cached Blob and forwards cancellation", async () => {
  const calls = [];
  const { cache, add } = createCache(async (url, options) => {
    calls.push({ url, options });
    return { ok: true, json: async () => ({ version: "v1" }) };
  });
  const request = { ...body, session_id: "session /a", photo_id: "photo /a" };
  const blob = add(request);
  const controller = new AbortController();
  const result = await cache.getValidatedCachedPreview(baseUrl, request, controller.signal);
  assert.equal(result.blob, blob);
  assert.equal(calls[0].url, `${baseUrl}/api/enhance/source-version/session%20%2Fa/photo%20%2Fa`);
  assert.equal(calls[0].options.cache, "no-store");
  assert.equal(calls[0].options.signal, controller.signal);
  result.headers[0][1] = "modified-copy";
  const next = await cache.getValidatedCachedPreview(baseUrl, request);
  assert.equal(new Headers(next.headers).get("X-Source-Version"), "v1");
});

test("source changes remove all render variants without removing other sources", async () => {
  const { cache, add } = createCache(async () => ({ ok: true, json: async () => ({ version: "v2" }) }));
  add();
  const fastBody = { ...body, full_resolution: false };
  const originalBody = { ...body, mode: "original" };
  add(fastBody);
  add(originalBody);
  const otherPhoto = { ...body, photo_id: "photo-b" };
  add(otherPhoto);
  add(body, "v1", "http://other-engine");
  assert.equal(await cache.getValidatedCachedPreview(baseUrl, body), undefined);
  assert.equal(cache.hasCachedPreview(baseUrl, body), false);
  assert.equal(cache.hasCachedPreview(baseUrl, fastBody), false);
  assert.equal(cache.hasCachedPreview(baseUrl, originalBody), false);
  assert.equal(cache.hasCachedPreview(baseUrl, otherPhoto), true);
  assert.equal(cache.hasCachedPreview("http://other-engine", body), true);
  assert.equal(cache.previewCacheStats().entries, 2);
  assert.equal(cache.previewCacheStats().bytes, new Blob(["preview"]).size * 2);
});

test("version endpoint failures never return or retain stale cached variants", async t => {
  const failures = {
    http: async () => ({ ok: false }),
    network: async () => { throw new Error("Service offline"); },
    malformed: async () => ({ ok: true, json: async () => ({ version: null }) }),
    empty: async () => ({ ok: true, json: async () => ({ version: "" }) }),
    json: async () => ({ ok: true, json: async () => { throw new Error("Invalid JSON"); } }),
  };
  for (const [name, fetch] of Object.entries(failures)) {
    await t.test(name, async () => {
      const { cache, add } = createCache(fetch);
      add();
      add({ ...body, mode: "original" });
      await assert.rejects(cache.getValidatedCachedPreview(baseUrl, body));
      assert.equal(cache.previewCacheStats().entries, 0);
      assert.equal(cache.previewCacheStats().bytes, 0);
    });
  }
});

test("aborted validation does not deliver or evict a reusable candidate preview", async () => {
  const { cache, add } = createCache();
  add();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(cache.getValidatedCachedPreview(baseUrl, body, controller.signal), { name: "AbortError" });
  assert.equal(cache.hasCachedPreview(baseUrl, body), true);
});

test("responses without a source identity or version never enter the cache", () => {
  const { cache } = createCache();
  const blob = new Blob(["preview"]);
  const response = { ok: true, status: 200, headers: new Headers() };
  assert.equal(cache.cachePreviewResponse(baseUrl, body, response, blob), false);
  response.headers.set("X-Source-Version", "v1");
  assert.equal(cache.cachePreviewResponse(baseUrl, {}, response, blob), false);
  assert.equal(cache.cachePreviewResponse(baseUrl, body, { ...response, status: 204 }, blob), false);
  assert.equal(cache.cachePreviewResponse(baseUrl, body, { ...response, ok: false }, blob), false);
  assert.equal(cache.previewCacheStats().entries, 0);
});

test("a newly rendered source version evicts older cached variants", () => {
  const { cache, add } = createCache();
  add();
  const nextBody = { ...body, mode: "original" };
  add(nextBody, "v2");
  assert.equal(cache.hasCachedPreview(baseUrl, body), false);
  assert.equal(cache.hasCachedPreview(baseUrl, nextBody), true);
  assert.equal(cache.previewCacheStats().entries, 1);
});

test("validation cannot resurrect an entry replaced while its request was pending", async () => {
  const version = deferred();
  const { cache, add } = createCache(() => version.promise);
  add();
  const pending = cache.getValidatedCachedPreview(baseUrl, body);
  add(body, "v2");
  version.resolve({ ok: true, json: async () => ({ version: "v1" }) });
  assert.equal(await pending, undefined);
  assert.equal(cache.hasCachedPreview(baseUrl, body), true);
});

test("generation and client IDs do not change preview identity", () => {
  const { cache } = createCache();
  assert.equal(
    cache.previewCacheKey(baseUrl, { ...body, preview_generation: 1, preview_client_id: "a" }),
    cache.previewCacheKey(baseUrl, { ...body, preview_generation: 2, preview_client_id: "b" }),
  );
  assert.notEqual(cache.previewCacheKey(baseUrl, body), cache.previewCacheKey(baseUrl, { ...body, mode: "original" }));
});

test("releasing one page URL preserves another page URL for the same cached Blob", () => {
  const { cache, add, revoked } = createCache();
  const blob = add();
  const firstUrl = cache.createPreviewUrl(blob);
  const secondUrl = cache.createPreviewUrl(blob);
  const image = { naturalWidth: 20, naturalHeight: 10 };
  const histogram = ["M 0 0"];
  cache.rememberDecodedPreview(firstUrl, image);
  cache.rememberPreviewHistogram(firstUrl, histogram);
  cache.releasePreviewUrl(firstUrl);
  assert.equal(cache.getDecodedPreview(firstUrl), undefined);
  assert.equal(cache.getDecodedPreview(secondUrl), image);
  assert.equal(cache.getPreviewHistogram(secondUrl), histogram);
  assert.deepEqual(revoked, [firstUrl]);
  cache.releasePreviewUrl(secondUrl);
  assert.deepEqual(revoked, [firstUrl, secondUrl]);
  cache.clearPreviewCache();
  assert.equal(cache.previewCacheStats().entries, 0);
});
