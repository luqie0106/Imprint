import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const vuePath = new URL("../src/views/EnhancePage.vue", import.meta.url);
const vueSource = await readFile(vuePath, "utf8");
const script = vueSource.match(/<script setup lang="ts">([\s\S]*?)<\/script>/)?.[1];
assert.ok(script, "EnhancePage.vue has a TypeScript setup script");

const sourceFile = ts.createSourceFile("EnhancePage.ts", script, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
const functions = new Map();
let draggingIntervalDeclaration;
for (const statement of sourceFile.statements) {
  if (ts.isFunctionDeclaration(statement) && statement.name) {
    functions.set(statement.name.text, statement.getText(sourceFile));
  } else if (ts.isVariableStatement(statement)
    && statement.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name)
      && declaration.name.text === "draggingPreviewIntervalMs")) {
    draggingIntervalDeclaration = statement.getText(sourceFile);
  }
}
assert.ok(draggingIntervalDeclaration, "could not extract the drag interval constant");

const requiredFunctions = [
  "refreshPreview",
  "previewOriginalRequestPlan",
  "clearDraggingPreviewState",
  "queueDraggingPreview",
  "runDraggingPreview",
  "scheduleDraggingPreview",
  "schedulePreview",
  "scheduleParameterPreview",
  "endSliderInteraction",
];
for (const name of requiredFunctions) assert.ok(functions.has(name), `could not extract ${name}`);

class FakeClock {
  now = 0;
  nextId = 1;
  timers = new Map();

  setTimeout(callback, delay = 0) {
    const id = this.nextId++;
    this.timers.set(id, { due: this.now + Math.max(0, Number(delay) || 0), callback });
    return id;
  }

  clearTimeout(id) {
    this.timers.delete(id);
  }

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

async function flushMicrotasks(rounds = 24) {
  for (let i = 0; i < rounds; i += 1) await Promise.resolve();
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

function makeHarness(fetchPreviewImpl = async (call) => assetFor(call)) {
  const clock = new FakeClock();
  const calls = [];
  const revoked = [];
  const state = {
    basicParams: { exposure: 0, contrast: 0 },
    previewMode: "enhanced",
    photoId: "photo-a",
  };
  const context = {
    props: { active: true },
    window: clock.window,
    performance: { now: () => clock.now },
    AbortController,
    DOMException,
    BASE_URL: { value: "http://127.0.0.1" },
    sessionId: { value: "session-a" },
    selectedId: { get value() { return state.photoId; }, set value(value) { state.photoId = value; } },
    previewMode: { get value() { return state.previewMode; }, set value(value) { state.previewMode = value; } },
    previewGeneration: 0,
    previewController: undefined,
    previewTimer: undefined,
    intermediatePreviewTimer: undefined,
    fullResolutionTimer: undefined,
    sliderAdjusting: false,
    sliderChanged: false,
    sliderFinishing: false,
    draggingPreviewActive: false,
    draggingPreviewGeneration: 0,
    draggingPreviewContext: undefined,
    draggingPreviewIncludeOriginal: false,
    draggingPreviewRunning: false,
    draggingPreviewStarted: false,
    draggingPreviewQueued: false,
    draggingPreviewNotBefore: 0,
    draggingPreviewTimer: undefined,
    previewLoading: { value: false },
    previewError: { value: "" },
    originalUrl: { value: "" },
    enhancedUrl: { value: "" },
    originalPhotoId: { value: "" },
    originalSessionId: { value: "" },
    originalFullResolution: { value: false },
    nonlocalRenderStatus: { value: "off" },
    nonlocalRenderReason: { value: "" },
    autoExposurePreview: { value: null },
    displayedPhotoId: { value: "" },
    displayedSessionId: { value: "" },
    sourceImageWidth: { value: 0 },
    sourceImageHeight: { value: 0 },
    imageWidth: { value: 0 },
    imageHeight: { value: 0 },
    imageSize: { value: "" },
    nextTick: (callback) => callback ? Promise.resolve().then(callback) : Promise.resolve(),
    updateViewportSize() {},
    releaseUrl(url) { if (url) revoked.push(url); },
    capturePreviewContext() {
      return {
        sessionId: "session-a",
        photoId: state.photoId,
        params: {},
        autoMode: false,
        autoExposure: false,
        nonlocalMode: "off",
        basicParams: { ...state.basicParams },
        ricohPresetId: null,
        renderBackend: "cpu",
        basicBackend: "cpu",
        ricohBackend: "cpu",
      };
    },
    async fetchPreview(mode, requestContext, previewLevel, fullResolution, signal) {
      const call = {
        mode,
        context: requestContext,
        previewLevel,
        fullResolution,
        signal,
      };
      calls.push(call);
      return fetchPreviewImpl(call, calls.length - 1);
    },
  };

  const executable = [
    draggingIntervalDeclaration,
    ...requiredFunctions.map((name) => functions.get(name)),
    `globalThis.__previewFunctions = { ${requiredFunctions.join(", ")} };`,
  ].join("\n\n");
  const javascript = ts.transpile(executable, { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None });
  vm.runInNewContext(javascript, context, { filename: "EnhancePage.preview-schedule.test.js" });
  return { api: context.__previewFunctions, context, clock, calls, revoked, state };
}

function assetFor(call) {
  const quality = call.fullResolution ? "full" : `L${call.previewLevel}`;
  const photoId = call.context.basicParams.exposure;
  return {
    url: `${call.mode}:${photoId}:${quality}`,
    width: call.fullResolution ? 6000 : 1800,
    height: call.fullResolution ? 4000 : 1200,
    nonlocalStatus: "off",
    nonlocalReason: "",
    autoExposureEV: null,
    autoExposureReason: "off",
  };
}

test("slider release renders the final basic value at L2 immediately and waits before full resolution", async () => {
  const finalL2 = deferred();
  const harness = makeHarness((call) => {
    if (call.mode === "dehazed" && call.context.basicParams.exposure === 7
      && call.previewLevel === 2 && !call.fullResolution) return finalL2.promise;
    return Promise.resolve(assetFor(call));
  });
  const { api, context, clock, calls } = harness;

  context.sliderAdjusting = true;
  harness.state.basicParams.exposure = 1;
  api.scheduleParameterPreview();
  harness.state.basicParams.exposure = 7;
  api.scheduleParameterPreview();
  api.endSliderInteraction();
  await flushMicrotasks();

  assert.equal(calls.length, 2, "drag starts L2 immediately and release starts a final L2");
  assert.equal(calls[0].mode, "dehazed");
  assert.equal(calls[0].context.basicParams.exposure, 1);
  assert.equal(calls[0].previewLevel, 2);
  assert.equal(calls[0].fullResolution, false);
  assert.equal(calls[0].signal.aborted, true, "release invalidates the drag request");
  assert.equal(calls[1].context.basicParams.exposure, 7);
  assert.equal(calls[1].previewLevel, 2);
  assert.equal(calls[1].fullResolution, false);

  await clock.advance(99);
  assert.equal(calls.length, 2, "full resolution has not started before 100 ms");
  await clock.advance(1);
  assert.equal(calls.length, 2, "the 100 ms upgrade waits for the final L2 response");

  finalL2.resolve(assetFor(calls[1]));
  await flushMicrotasks();
  assert.equal(calls.length, 3);
  assert.equal(calls[2].context.basicParams.exposure, 7);
  assert.equal(calls[2].previewLevel, 0);
  assert.equal(calls[2].fullResolution, true);
  assert.equal(calls.some((call) => call.previewLevel === 0 && !call.fullResolution), false,
    "settled release skips the intermediate L0 render");
});

test("continuous slider input keeps L2 serialized and renders the latest queued value", async () => {
  const firstL2 = deferred();
  const secondL2 = deferred();
  const thirdL2 = deferred();
  let l2Count = 0;
  const harness = makeHarness((call) => {
    if (call.mode === "dehazed" && !call.fullResolution && call.previewLevel === 2) {
      l2Count += 1;
      if (l2Count === 1) return firstL2.promise;
      if (l2Count === 2) return secondL2.promise;
      if (l2Count === 3) return thirdL2.promise;
    }
    return Promise.resolve(assetFor(call));
  });
  const { api, clock, calls, context, revoked } = harness;
  harness.context.sliderAdjusting = true;
  api.scheduleParameterPreview();
  const firstSignal = calls[0].signal;

  for (let value = 1; value <= 10; value += 1) {
    await clock.advance(16);
    harness.state.basicParams.exposure = value;
    api.scheduleParameterPreview();
    await flushMicrotasks();
  }
  assert.equal(clock.now, 160, "input continues for more than the old 140 ms debounce");
  assert.equal(calls.length, 1, "no overlapping L2 request starts while the first is still decoding");
  assert.equal(firstSignal.aborted, false, "input changes do not abort the in-flight L2 request");

  firstL2.resolve(assetFor(calls[0]));
  await flushMicrotasks();
  assert.equal(context.enhancedUrl.value, "dehazed:0:L2", "the leading L2 becomes visible during continuous input");
  await clock.advance(0);
  assert.equal(calls.length, 2);
  assert.equal(calls[1].context.basicParams.exposure, 10, "the next request captures the latest queued value");

  for (let value = 11; value <= 13; value += 1) {
    await clock.advance(16);
    harness.state.basicParams.exposure = value;
    api.scheduleParameterPreview();
  }
  assert.equal(calls.length, 2);
  assert.equal(calls[1].signal.aborted, false);
  secondL2.resolve(assetFor(calls[1]));
  await flushMicrotasks();

  for (const value of [90, 99]) {
    harness.state.basicParams.exposure = value;
    api.scheduleParameterPreview();
  }
  await clock.advance(0);
  assert.equal(calls.length, 3);
  assert.equal(calls[2].context.basicParams.exposure, 99, "queued updates replace older pending values");

  harness.state.basicParams.exposure = 101;
  api.scheduleParameterPreview();
  api.endSliderInteraction();
  await flushMicrotasks();
  assert.equal(calls.length, 4, "release immediately starts a new final-value L2 request");
  assert.equal(calls[2].signal.aborted, true, "release cancels the active drag-generation request");
  assert.equal(calls[3].context.basicParams.exposure, 101);
  thirdL2.resolve(assetFor(calls[2]));
  await flushMicrotasks();
  assert.equal(context.enhancedUrl.value, "dehazed:101:L2", "a late drag result cannot replace the final-value preview");
  assert.ok(revoked.includes("dehazed:99:L2"), "the late drag object URL is released");
  await clock.advance(100);
  assert.equal(calls.length, 5, "only the final full-resolution upgrade starts after release");
  assert.equal(calls[4].fullResolution, true);
  assert.equal(calls[4].context.basicParams.exposure, 101);
});

test("fast slider previews are capped at one L2 start per 16 ms", async () => {
  const harness = makeHarness();
  const { api, clock, calls } = harness;
  harness.context.sliderAdjusting = true;
  api.scheduleParameterPreview();
  await flushMicrotasks();
  assert.equal(calls.length, 1, "the first L2 starts immediately");

  harness.state.basicParams.exposure = 1;
  api.scheduleParameterPreview();
  await clock.advance(15);
  assert.equal(calls.length, 1, "a fast response does not remove the 16 ms start interval");
  await clock.advance(1);
  assert.equal(calls.length, 2, "the latest pending L2 starts as soon as 16 ms elapses");
  assert.equal(calls[1].context.basicParams.exposure, 1);
});

test("changing photos clears queued drag work and prevents the old photo from starting another L2", async () => {
  const firstL2 = deferred();
  const harness = makeHarness((call) => {
    if (call.context.photoId === "photo-a" && call.previewLevel === 2 && !call.fullResolution) return firstL2.promise;
    return Promise.resolve(assetFor(call));
  });
  const { api, clock, calls, context, revoked } = harness;
  harness.context.sliderAdjusting = true;
  api.scheduleParameterPreview();
  harness.state.basicParams.exposure = 1;
  api.scheduleParameterPreview();
  assert.equal(harness.context.draggingPreviewQueued, true, "the newest value is queued after the in-flight L2");

  await clock.advance(10);
  harness.context.sliderAdjusting = false;
  harness.context.sliderChanged = false;
  harness.state.photoId = "photo-b";
  harness.state.basicParams.exposure = 2;
  api.schedulePreview(true);
  await flushMicrotasks();
  assert.equal(calls.length, 2, "selecting a photo starts its enhanced fast preview immediately");
  assert.deepEqual(calls.slice(1).map((call) => [call.mode, call.context.photoId, call.previewLevel, call.fullResolution]), [
    ["dehazed", "photo-b", 2, false],
  ]);
  const currentPreview = context.enhancedUrl.value;
  firstL2.resolve(assetFor(calls[0]));
  await flushMicrotasks();
  assert.equal(context.enhancedUrl.value, currentPreview, "a late response for photo A cannot replace photo B");
  assert.ok(revoked.includes("dehazed:0:L2"), "the stale photo A object URL is released");
  await clock.advance(35);
  assert.equal(calls.length, 2, "the old queued callback does not run for photo A");
  await clock.advance(105);
  assert.equal(calls.length, 4, "the new photo completes its full-resolution upgrade at 100 ms");
  assert.deepEqual(calls.slice(1).map((call) => [call.previewLevel, call.fullResolution]),
    [[2, false], [0, false], [0, true]]);
  assert.equal(calls.slice(1).every((call) => call.context.photoId === "photo-b"), true);
  assert.equal(calls[1].context.photoId, "photo-b");
});

test("first photo preview starts the fast request immediately and reuses it for later upgrades", async () => {
  const harness = makeHarness();
  const { api, clock, calls } = harness;
  api.schedulePreview(true);
  assert.equal(calls.length, 1, "cold photo selection starts L2 without the usual 140 ms delay");
  assert.equal(calls[0].previewLevel, 2);
  assert.equal(calls[0].fullResolution, false);
  await flushMicrotasks();
  await clock.advance(100);
  await flushMicrotasks();
  assert.deepEqual(calls.map((call) => [call.previewLevel, call.fullResolution]), [
    [2, false], [0, false], [0, true],
  ], "progressive stages reuse the already-started fast task without repeating L2");
});

test("effect-only requests skip original while compare requests include it", async () => {
  const effect = makeHarness();
  effect.api.schedulePreview(false, "settled");
  await flushMicrotasks();
  await effect.clock.advance(100);
  await flushMicrotasks();
  assert.deepEqual(effect.calls.map((call) => call.mode), ["dehazed", "dehazed"]);
  assert.deepEqual(effect.calls.map((call) => call.fullResolution), [false, true]);

  const compare = makeHarness();
  compare.state.previewMode = "compare";
  compare.api.schedulePreview(false, "settled");
  await flushMicrotasks();
  await compare.clock.advance(100);
  await flushMicrotasks();
  assert.deepEqual(compare.calls.map((call) => call.mode), ["original", "dehazed", "original", "dehazed"]);
  assert.deepEqual(compare.calls.map((call) => call.fullResolution), [false, false, true, true]);
});

test("a new generation aborts an in-flight full request and its late response cannot replace the new preview", async () => {
  const staleFull = deferred();
  let staleSignal;
  const harness = makeHarness((call) => {
    if (call.context.basicParams.exposure === 1 && call.fullResolution) {
      staleSignal = call.signal;
      return staleFull.promise;
    }
    return Promise.resolve(assetFor(call));
  });
  const { api, context, clock, calls, revoked } = harness;

  harness.state.basicParams.exposure = 1;
  api.schedulePreview(false, "settled");
  await flushMicrotasks();
  await clock.advance(100);
  await flushMicrotasks();
  assert.ok(staleSignal, "first generation reached its full-resolution request");
  assert.equal(staleSignal.aborted, false);

  harness.state.basicParams.exposure = 2;
  api.schedulePreview(false, "settled");
  await flushMicrotasks();
  assert.equal(staleSignal.aborted, true, "starting a new generation aborts the previous request");
  const currentPreview = context.enhancedUrl.value;
  staleFull.resolve(assetFor(calls.find((call) => call.signal === staleSignal)));
  await flushMicrotasks();
  assert.equal(context.enhancedUrl.value, currentPreview);
  assert.ok(revoked.includes("dehazed:1:full"), "late stale object URL is released");
});
