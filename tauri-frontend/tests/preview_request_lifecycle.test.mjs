import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const enhanceScript = await readVueScript("../src/views/EnhancePage.vue", "EnhancePage.vue");
const ricohScript = await readVueScript("../src/views/RicohPage.vue", "RicohPage.vue");
const appSource = await readFile(new URL("../src/App.vue", import.meta.url), "utf8");
const enhanceFunctions = extractFunctions(enhanceScript, [
  "fetchPreview", "refreshPreview", "clearDraggingPreviewState", "scheduleDraggingPreview",
  "schedulePreview", "cancelPendingPreview",
]);
const ricohFunctions = extractFunctions(ricohScript, [
  "cloneBasicParams", "releasePreview", "requestPreview", "refreshPreview", "schedulePreview",
  "cancelPendingPreview",
]);
const ricohBasicParamsDeclaration = extractVariableStatement(ricohScript, "defaultBasicParams");

async function readVueScript(relativePath, filename) {
  const source = await readFile(new URL(relativePath, import.meta.url), "utf8");
  const script = source.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)?.[1];
  assert.ok(script, `${filename} has a TypeScript setup script`);
  return script;
}

function extractFunctions(script, names) {
  const sourceFile = ts.createSourceFile("PreviewLifecycle.ts", script, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  const functions = new Map();
  for (const statement of sourceFile.statements) {
    if (ts.isFunctionDeclaration(statement) && statement.name && names.includes(statement.name.text)) {
      functions.set(statement.name.text, statement.getText(sourceFile));
    }
  }
  for (const name of names) assert.ok(functions.has(name), `could not extract ${name}`);
  return functions;
}

function extractVariableStatement(script, variableName) {
  const sourceFile = ts.createSourceFile("PreviewLifecycle.ts", script, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  for (const statement of sourceFile.statements) {
    if (ts.isVariableStatement(statement)
      && statement.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name)
        && declaration.name.text === variableName)) return statement.getText(sourceFile);
  }
  assert.fail(`could not extract ${variableName}`);
}

function compileFunctions(functions, declarations, exposedNames, suffix = "") {
  const source = [declarations, ...functions.values(),
    `globalThis.__api = { ${exposedNames.join(", ")} };`, suffix].filter(Boolean).join("\n\n");
  return ts.transpile(source, { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
}

function makeResponse({ status = 200, width = 6000, height = 4000 } = {}) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: (name) => ({ "X-Image-Width": String(width), "X-Image-Height": String(height) })[name] ?? null },
    json: async () => ({}),
    blob: async () => ({}),
  };
}

function makeUrlApi() {
  let next = 0;
  const revoked = [];
  return {
    revoked,
    createObjectURL: () => `blob:preview-${++next}`,
    revokeObjectURL: (url) => revoked.push(url),
  };
}

async function flushMicrotasks(rounds = 24) {
  for (let index = 0; index < rounds; index += 1) await Promise.resolve();
}

class FakeClock {
  now = 0;
  nextId = 1;
  timers = new Map();

  setTimeout(callback, delay = 0) {
    const id = this.nextId++;
    this.timers.set(id, { due: this.now + Math.max(0, Number(delay) || 0), callback });
    return id;
  }

  clearTimeout(id) { this.timers.delete(id); }

  async advance(ms) {
    const target = this.now + ms;
    for (;;) {
      const next = [...this.timers.entries()]
        .filter(([, timer]) => timer.due <= target)
        .sort((a, b) => a[1].due - b[1].due || a[0] - b[0])[0];
      if (!next) break;
      const [id, timer] = next;
      this.now = timer.due;
      this.timers.delete(id);
      timer.callback();
      await flushMicrotasks();
    }
    this.now = target;
    await flushMicrotasks();
  }

  window = {
    setTimeout: (callback, delay) => this.setTimeout(callback, delay),
    clearTimeout: (id) => this.clearTimeout(id),
  };
}

function makeEnhanceHarness(responses = []) {
  const clock = new FakeClock();
  const calls = [];
  const urlApi = makeUrlApi();
  const context = {
    props: { active: true },
    BASE_URL: { value: "http://127.0.0.1" },
    sessionId: { value: "session-a" },
    selectedId: { value: "photo-a" },
    previewLoading: { value: false },
    window: clock.window,
    DOMException,
    URL: urlApi,
    fetch: async (url, options) => {
      calls.push({ url, options, body: JSON.parse(options.body) });
      return responses[calls.length - 1] ?? makeResponse();
    },
  };
  const declarations = [
    "let previewGeneration = 0;",
    "const previewClientId = 'enhance-client-test';",
    "let previewTimer; let intermediatePreviewTimer; let fullResolutionTimer;",
    "let draggingPreviewTimer; let draggingPreviewActive = false; let draggingPreviewGeneration = 0;",
    "let draggingPreviewContext; let draggingPreviewIncludeOriginal = false; let draggingPreviewRunning = false;",
    "let draggingPreviewStarted = false; let draggingPreviewQueued = false; let draggingPreviewNotBefore = 0;",
    "let previewController; let sliderAdjusting = false; let sliderFinishing = false;",
  ].join("\n");
  const suffix = `globalThis.__lifecycle = {
    get generation() { return previewGeneration; },
    set generation(value) { previewGeneration = value; },
    set controller(value) { previewController = value; },
  };`;
  const javascript = compileFunctions(enhanceFunctions, declarations,
    ["fetchPreview", "refreshPreview", "scheduleDraggingPreview", "schedulePreview", "cancelPendingPreview"], suffix);
  vm.runInNewContext(javascript, context, { filename: "EnhancePage.preview-request-lifecycle.test.js" });
  return { api: context.__api, calls, urlApi, context, clock, lifecycle: context.__lifecycle };
}

function makeRicohHarness(fetchImpl = async () => makeResponse()) {
  const clock = new FakeClock();
  const calls = [];
  const urlApi = makeUrlApi();
  const state = {
    photoId: "photo-a",
    sessionId: "session-a",
    basicByPhoto: { "photo-a": { exposure: 0 }, "photo-b": { exposure: 0 } },
    originalUrl: "",
    effectUrl: "",
  };
  const ref = (value) => ({ value });
  const context = {
    props: { active: true },
    BASE_URL: ref("http://127.0.0.1"),
    sessionId: { get value() { return state.sessionId; } },
    selectedId: { get value() { return state.photoId; }, set value(value) { state.photoId = value; } },
    renderBackend: ref("metal"),
    basicBackend: ref("cpu"),
    ricohBackend: ref("cpu"),
    sharedDehazeAutoByPhoto: ref({}),
    sharedDehazeAutoExposureByPhoto: ref({}),
    sharedDehazeNonlocalByPhoto: ref({}),
    dehazeParams: { get value() { return {}; } },
    basicParams: { get value() { return state.basicByPhoto[state.photoId]; } },
    selectedPreset: ref("preset-a"),
    originalUrl: { get value() { return state.originalUrl; }, set value(value) { state.originalUrl = value; } },
    originalFullResolution: ref(false),
    effectUrl: { get value() { return state.effectUrl; }, set value(value) { state.effectUrl = value; } },
    generation: 0,
    previewTimer: undefined,
    fullResolutionTimer: undefined,
    scheduledPreviewTask: undefined,
    previewClientId: "ricoh-client-test",
    previewViewport: ref(null),
    previewLoading: ref(false),
    error: ref(""),
    nonlocalRenderStatus: ref("off"),
    nonlocalRenderReason: ref(""),
    autoExposurePreview: ref(null),
    displayedPhotoId: ref(""),
    displayedSessionId: ref(""),
    sourceImageWidth: ref(0),
    sourceImageHeight: ref(0),
    imageWidth: ref(0),
    imageHeight: ref(0),
    imageSize: ref(""),
    previewError: ref(""),
    nextTick: async () => {},
    updateViewportSize() {},
    window: clock.window,
    DOMException,
    URL: urlApi,
    fetch: async (url, options) => {
      const call = { url, options, body: JSON.parse(options.body) };
      calls.push(call);
      return fetchImpl(call, calls.length - 1);
    },
  };
  const declarations = [
    "let generation = 0;",
    "const previewClientId = 'ricoh-client-test';",
    ricohBasicParamsDeclaration,
    "let previewTimer; let fullResolutionTimer; let scheduledPreviewTask;",
  ].join("\n");
  const suffix = `globalThis.__lifecycle = {
    get generation() { return generation; },
    set generation(value) { generation = value; },
    get scheduledPreviewTask() { return scheduledPreviewTask; },
  };`;
  const javascript = compileFunctions(ricohFunctions, declarations,
    ["releasePreview", "requestPreview", "refreshPreview", "schedulePreview", "cancelPendingPreview"], suffix);
  vm.runInNewContext(javascript, context, { filename: "RicohPage.preview-request-lifecycle.test.js" });
  return { api: context.__api, lifecycle: context.__lifecycle, calls, clock, state, context, urlApi };
}

test("Enhance requests keep the captured client and generation across quality stages and treat 204 as AbortError", async () => {
  const harness = makeEnhanceHarness([makeResponse(), makeResponse(), makeResponse({ status: 204 })]);
  const context = {
    generation: 41,
    sessionId: "session-a",
    photoId: "photo-a",
    params: { strength: 0.4 },
    autoMode: false,
    autoExposure: false,
    nonlocalMode: "off",
    basicParams: { exposure: 0 },
    ricohPresetId: null,
    renderBackend: "metal",
    basicBackend: "cpu",
    ricohBackend: "cpu",
  };

  const low = await harness.api.fetchPreview("dehazed", context, 2, false, new AbortController().signal);
  harness.lifecycle.generation = 42;
  const full = await harness.api.fetchPreview("dehazed", context, 0, true, new AbortController().signal);

  assert.equal(harness.calls[0].body.preview_client_id, "enhance-client-test");
  assert.equal(harness.calls[0].body.preview_generation, 41,
    "the request uses its captured context generation even after the global generation changes");
  assert.equal(harness.calls[1].body.preview_client_id, harness.calls[0].body.preview_client_id);
  assert.equal(harness.calls[1].body.preview_generation, harness.calls[0].body.preview_generation);
  assert.deepEqual([harness.calls[0].body.preview_level, harness.calls[0].body.full_resolution], [2, false]);
  assert.deepEqual([harness.calls[1].body.preview_level, harness.calls[1].body.full_resolution], [0, true]);
  assert.ok(low.url && full.url);
  await assert.rejects(
    harness.api.fetchPreview("dehazed", context, 0, true, new AbortController().signal),
    (error) => error instanceof DOMException && error.name === "AbortError",
  );
});

test("Ricoh reuses a full-resolution original for later exposure generations and reloads it after release", async () => {
  const harness = makeRicohHarness();
  const { api, calls, clock, state, context, lifecycle } = harness;

  api.schedulePreview(true, true);
  await clock.advance(0);
  assert.deepEqual(calls.map(({ body }) => [body.mode, body.preview_level, body.full_resolution]), [
    ["original", 2, false], ["dehazed", 2, false],
  ]);
  await clock.advance(500);
  assert.deepEqual(calls.slice(2).map(({ body }) => [body.mode, body.preview_level, body.full_resolution]), [
    ["original", 0, true], ["dehazed", 0, true],
  ]);

  const firstGeneration = calls[0].body.preview_generation;
  assert.ok(calls.slice(0, 4).every(({ body }) => body.preview_client_id === "ricoh-client-test"));
  assert.ok(calls.slice(0, 4).every(({ body }) => body.preview_generation === firstGeneration));
  assert.ok(calls.slice(0, 4).every(({ body }) => body.render_backend === "metal"));
  assert.equal(context.originalFullResolution.value, true);

  state.basicByPhoto["photo-a"].exposure = 1;
  api.schedulePreview(false, true);
  await clock.advance(0);
  await clock.advance(500);
  const laterCalls = calls.slice(4);
  assert.deepEqual(laterCalls.map(({ body }) => [body.mode, body.preview_level, body.full_resolution]), [
    ["dehazed", 2, false], ["dehazed", 0, true],
  ]);
  assert.ok(laterCalls.every(({ body }) => body.preview_generation > firstGeneration));
  assert.ok(laterCalls.every(({ body }) => body.render_backend === "metal"));

  api.releasePreview();
  assert.equal(context.originalFullResolution.value, false);
  state.photoId = "photo-b";
  api.schedulePreview(false, true);
  await clock.advance(0);
  const nextPhotoCalls = calls.slice(6);
  assert.deepEqual(nextPhotoCalls.map(({ body }) => body.mode), ["original", "dehazed"]);
  assert.ok(nextPhotoCalls.every(({ body }) => body.photo_id === "photo-b"));
  assert.ok(nextPhotoCalls.every(({ body }) => body.render_backend === "metal"));
  assert.equal(context.originalFullResolution.value, false,
    "the next photo's low-resolution original is not marked full resolution");
  assert.ok(nextPhotoCalls.every(({ body }) => body.preview_generation === lifecycle.generation));
});

test("a stale Ricoh 204 response cannot set the current error or replace its image URLs", async () => {
  const staleResponses = [];
  const harness = makeRicohHarness((call) => {
    if (call.body.photo_id === "photo-a") {
      return new Promise((resolve) => staleResponses.push(resolve));
    }
    return Promise.resolve(makeResponse());
  });
  const { api, calls, state, context, lifecycle } = harness;

  lifecycle.generation = 1;
  const staleRefresh = api.refreshPreview(true, false, 1);
  await flushMicrotasks();
  assert.equal(staleResponses.length, 2, "the stale original and effect requests are both pending");

  api.releasePreview();
  state.photoId = "photo-b";
  lifecycle.generation = 2;
  await api.refreshPreview(true, false, 2);
  const currentOriginal = context.originalUrl.value;
  const currentEffect = context.effectUrl.value;
  assert.ok(currentOriginal && currentEffect);
  assert.equal(context.error.value, "");

  staleResponses.forEach((resolve) => resolve(makeResponse({ status: 204 })));
  await staleRefresh;
  await flushMicrotasks();

  assert.equal(calls.length, 4);
  assert.equal(context.originalUrl.value, currentOriginal, "the old generation cannot replace the current original");
  assert.equal(context.effectUrl.value, currentEffect, "the old generation cannot replace the current effect");
  assert.equal(context.error.value, "", "an expired 204 from an old generation does not show an error");
});

test("inactive Enhance and Ricoh pages do not schedule or fetch previews", async () => {
  const enhance = makeEnhanceHarness();
  enhance.context.props.active = false;
  await enhance.api.refreshPreview(0, {
    sessionId: "session-a", photoId: "photo-a", params: {}, autoExposure: false,
  }, true, 2, false);
  enhance.api.schedulePreview(false, "settled");
  enhance.api.scheduleDraggingPreview(true);
  await enhance.clock.advance(1000);
  assert.equal(enhance.calls.length, 0);
  assert.equal(enhance.clock.timers.size, 0);

  const ricoh = makeRicohHarness();
  ricoh.context.props.active = false;
  await ricoh.api.refreshPreview(false, false, ricoh.lifecycle.generation);
  ricoh.api.schedulePreview(false, true);
  await ricoh.clock.advance(1000);
  assert.equal(ricoh.calls.length, 0);
  assert.equal(ricoh.clock.timers.size, 0);
});

test("App passes each kept-alive page its active-tab state", () => {
  assert.match(appSource, /<EnhancePage[^>]*v-show="activeTab === 'enhance'"[^>]*:active="activeTab === 'enhance'"/);
  assert.match(appSource, /<RicohPage[^>]*v-show="activeTab === 'ricoh'"[^>]*:active="activeTab === 'ricoh'"/);
});

test("page cancellation sends a generation-scoped cancel-only request without decoding a response", async () => {
  let enhanceDecodeCount = 0;
  const enhanceResponse = {
    ...makeResponse(),
    json: async () => { enhanceDecodeCount += 1; return {}; },
    blob: async () => { enhanceDecodeCount += 1; return {}; },
  };
  const enhance = makeEnhanceHarness([enhanceResponse]);
  enhance.lifecycle.generation = 8;
  enhance.api.cancelPendingPreview();
  await flushMicrotasks();
  assert.deepEqual(enhance.calls.map(({ url, body }) => [url, body]), [[
    "http://127.0.0.1/api/enhance/preview",
    {
      session_id: "session-a", photo_id: "photo-a", preview_client_id: "enhance-client-test",
      preview_generation: 9, preview_cancel_only: true,
    },
  ]]);
  assert.equal(enhance.lifecycle.generation, 9);
  assert.equal(enhanceDecodeCount, 0);

  let ricohDecodeCount = 0;
  const ricohResponse = {
    ...makeResponse(),
    json: async () => { ricohDecodeCount += 1; return {}; },
    blob: async () => { ricohDecodeCount += 1; return {}; },
  };
  const ricoh = makeRicohHarness(() => Promise.resolve(ricohResponse));
  ricoh.lifecycle.generation = 12;
  ricoh.api.cancelPendingPreview();
  await flushMicrotasks();
  assert.deepEqual(ricoh.calls.map(({ url, body }) => [url, body]), [[
    "http://127.0.0.1/api/enhance/preview",
    {
      session_id: "session-a", photo_id: "photo-a", preview_client_id: "ricoh-client-test",
      preview_generation: 13, preview_cancel_only: true,
    },
  ]]);
  assert.equal(ricoh.lifecycle.generation, 13);
  assert.equal(ricohDecodeCount, 0);
});
