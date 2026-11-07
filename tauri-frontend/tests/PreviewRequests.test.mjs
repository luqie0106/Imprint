import assert from "node:assert/strict";
import test from "node:test";
import { deferred, loadFunctions, readSource, sourceBetween } from "./sourceHarness.mjs";

const pageRequests = [
  { path: "src/views/EnhancePage.vue", start: "async function fetchPreview(", end: "\nasync function refreshPreview(", name: "fetchPreview" },
  { path: "src/views/RicohPage.vue", start: "async function requestPreview(", end: "\nfunction applyOriginalPreview(", name: "requestPreview" },
];

for (const page of pageRequests) {
  function createRequest() {
    const validation = deferred();
    const fetches = [];
    const cachedResponses = [];
    const createdUrls = [];
    const validatedSignals = [];
    const body = { session_id: "session-a", photo_id: "photo-a" };
    const context = { baseUrl: "mock-base", token: 7 };
    const controller = new AbortController();
    const blob = new Blob(["fresh-preview"]);
    const headers = new Headers({ "X-Image-Width": "6000", "X-Image-Height": "4000", "X-Source-Version": "v2" });
    const scope = {
      generation: 7, Headers, DOMException,
      getValidatedCachedPreview: (_base, _body, signal) => {
        validatedSignals.push(signal);
        return validation.promise;
      },
      createPreviewBody: () => body,
      cachePreviewResponse: (...args) => cachedResponses.push(args),
      createPreviewUrl: value => { createdUrls.push(value); return "blob:preview"; },
      releasePreviewUrl() {},
      fetch: async (...args) => {
        fetches.push(args);
        return { ok: true, status: 200, headers, blob: async () => blob };
      },
    };
    const code = sourceBetween(readSource(page.path), page.start, page.end);
    const request = loadFunctions(code, [page.name], scope)[page.name];
    const invoke = () => page.name === "fetchPreview"
      ? request("dehazed", context, 0, true, controller.signal)
      : request(body, context);
    return { validation, fetches, cachedResponses, createdUrls, validatedSignals, blob, controller, invoke };
  }

  test(`${page.name} waits for source validation before reusing a cached preview`, async () => {
    const request = createRequest();
    const pending = request.invoke();
    assert.equal(request.fetches.length, 0);
    assert.equal(request.createdUrls.length, 0);
    const cachedBlob = new Blob(["cached-preview"]);
    request.validation.resolve({ blob: cachedBlob, headers: [["X-Image-Width", "6000"], ["X-Image-Height", "4000"]] });
    const asset = await pending;
    assert.equal(asset.url, "blob:preview");
    assert.equal(asset.width, 6000);
    assert.equal(request.fetches.length, 0);
    assert.equal(request.createdUrls[0], cachedBlob);
    if (page.name === "fetchPreview") assert.equal(request.validatedSignals[0], request.controller.signal);
  });

  test(`${page.name} renders again when the cached source version changes`, async () => {
    const request = createRequest();
    const pending = request.invoke();
    request.validation.resolve(undefined);
    await pending;
    assert.equal(request.fetches.length, 1);
    assert.equal(request.fetches[0][0], "mock-base/api/enhance/preview");
    assert.equal(request.cachedResponses.length, 1);
    assert.equal(request.cachedResponses[0][3], request.blob);
    assert.equal(request.createdUrls[0], request.blob);
  });

  test(`${page.name} propagates version validation failure without displaying stale data`, async () => {
    const request = createRequest();
    const pending = request.invoke();
    request.validation.reject(new Error("Source version unavailable"));
    await assert.rejects(pending, /Source version unavailable/);
    assert.equal(request.fetches.length, 0);
    assert.equal(request.createdUrls.length, 0);
  });
}
