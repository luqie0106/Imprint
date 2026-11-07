import assert from "node:assert/strict";
import test from "node:test";
import { deferred, loadFunctions, readSource, ref, sourceBetween } from "./sourceHarness.mjs";

const source = readSource("src/views/RicohPage.vue");
const previewSource = sourceBetween(source, "function applyOriginalPreview(", "\nfunction schedulePreview(");

function createPreview() {
  const fastOriginal = deferred();
  const revoked = [];
  const scope = {
    props: { active: true }, generation: 7, originalGeneration: 0, BASE_URL: ref("mock-base"),
    originalUrl: ref(""), originalFullResolution: ref(false), effectUrl: ref(""),
    sourceImageWidth: ref(0), sourceImageHeight: ref(0), imageWidth: ref(0), imageHeight: ref(0),
    error: ref(""), mode: ref("compare"), DOMException,
    releasePreviewUrl: url => revoked.push(url), releasePreview() {},
    updateViewportSize() {}, nextTick: async callback => callback?.(),
    createPreviewBody: (_context, mode) => ({ mode }),
    requestPreview: async (body, context) => {
      if (!context.fullResolution && body.mode === "original") return fastOriginal.promise;
      return {
        url: `${context.fullResolution ? "full" : "fast"}-${body.mode}`,
        width: context.fullResolution ? 6000 : 1800, height: context.fullResolution ? 4000 : 1200,
      };
    },
  };
  const functions = loadFunctions(previewSource, ["applyOriginalPreview", "refreshPreview"], scope);
  return { scope, fastOriginal, revoked, ...functions };
}

test("late fast original cannot replace the completed full-resolution original", async () => {
  const { scope, fastOriginal, revoked, refreshPreview } = createPreview();
  const context = { token: 7, baseUrl: "mock-base", sessionId: "session-a", photoId: "photo-a" };
  await refreshPreview(true, { ...context, fullResolution: false });
  await refreshPreview(false, { ...context, fullResolution: true });
  assert.equal(scope.originalUrl.value, "full-original");
  fastOriginal.resolve({ url: "late-fast-original", width: 1800, height: 1200 });
  await Promise.resolve();
  await Promise.resolve();
  await Promise.resolve();
  assert.equal(scope.originalUrl.value, "full-original");
  assert.equal(scope.originalFullResolution.value, true);
  assert.equal(scope.effectUrl.value, "full-dehazed");
  assert.equal(scope.sourceImageWidth.value, 6000);
  assert.ok(revoked.includes("late-fast-original"));
  assert.ok(!revoked.includes("full-original"));
});

test("a new generation can replace a previous generation full original with its fast original", () => {
  const { scope, revoked, applyOriginalPreview } = createPreview();
  applyOriginalPreview({ url: "old-full", width: 6000, height: 4000 }, {
    token: 7, baseUrl: "mock-base", fullResolution: true,
  });
  scope.generation = 8;
  applyOriginalPreview({ url: "new-fast", width: 1800, height: 1200 }, {
    token: 8, baseUrl: "mock-base", fullResolution: false,
  });
  assert.equal(scope.originalUrl.value, "new-fast");
  assert.equal(scope.originalFullResolution.value, false);
  assert.equal(scope.originalGeneration, 8);
  assert.deepEqual(revoked, ["old-full"]);
});

test("superseded generations and backend responses release their unused URLs", () => {
  const { scope, revoked, applyOriginalPreview } = createPreview();
  scope.originalUrl.value = "current";
  applyOriginalPreview({ url: "stale-generation" }, { token: 6, baseUrl: "mock-base" });
  applyOriginalPreview({ url: "stale-backend" }, { token: 7, baseUrl: "old-base" });
  assert.equal(scope.originalUrl.value, "current");
  assert.deepEqual(revoked, ["stale-generation", "stale-backend"]);
});
