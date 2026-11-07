import assert from "node:assert/strict";
import test from "node:test";
import { deferred, loadFunctions, readSource, ref, sourceBetween } from "./sourceHarness.mjs";

const source = readSource("src/components/PhotoExport.vue");
const requestSource = sourceBetween(source, "async function request(", "\nasync function pollJob(");
const startSource = sourceBetween(source, "async function startExport()", "\nasync function cancelExport()");

function createExport() {
  const save = deferred();
  const calls = [];
  const photoSource = {
    session_id: "session-a", revision: 1,
    files: [{ photo_id: "photo-a" }], dehaze_defaults: { strength: 0.5 },
  };
  const scope = {
    BASE_URL: ref("http://old-engine"), sharedPhotoSource: ref(photoSource),
    exportFiles: ref(photoSource.files), photoExportRunning: ref(false), starting: ref(false),
    error: ref(""), warningDialogOpen: ref(false), autoSaveError: ref(""),
    sharedDehazeByPhoto: ref({ "photo-a": { strength: 0.91 } }),
    sharedDehazeAutoByPhoto: ref({ "photo-a": true }),
    sharedDehazeAutoExposureByPhoto: ref({ "photo-a": true }),
    sharedDehazeNonlocalByPhoto: ref({ "photo-a": "strong" }),
    sharedBasicByPhoto: ref({ "photo-a": { exposure: 1.25 } }),
    sharedPresetByPhoto: ref({ "photo-a": "positive-film" }),
    renderBackend: ref("auto"), basicBackend: ref("python"), ricohBackend: ref("native"),
    compression: ref("lossless_jpeg"), bitDepth: ref("source"), outputDir: ref("output-a"),
    job: ref(null), jobBaseUrl: "", pollTimer: undefined,
    flushPendingSaves: () => save.promise, pollJob: async () => {},
    window: { clearInterval() {}, setInterval() { return 1; } },
    fetch: async (url, options) => {
      calls.push({ url, payload: JSON.parse(options.body) });
      return { ok: true, json: async () => ({ job_id: "job-a", status: "queued", files: [] }) };
    },
  };
  const { startExport } = loadFunctions(`${requestSource}\n${startSource}`, ["startExport"], scope);
  return { scope, save, calls, startExport };
}

test("export freezes every setting before pending XMP saves finish", async () => {
  const { scope, save, calls, startExport } = createExport();
  const task = startExport();
  scope.sharedDehazeByPhoto.value["photo-a"].strength = 0.1;
  scope.sharedBasicByPhoto.value["photo-a"].exposure = -2;
  scope.sharedDehazeAutoByPhoto.value["photo-a"] = false;
  scope.sharedDehazeAutoExposureByPhoto.value["photo-a"] = false;
  scope.sharedDehazeNonlocalByPhoto.value["photo-a"] = "off";
  scope.sharedPresetByPhoto.value["photo-a"] = null;
  scope.outputDir.value = "output-b";
  scope.renderBackend.value = "cpu";
  scope.basicBackend.value = "native";
  scope.ricohBackend.value = "python";
  scope.compression.value = "jpegxl";
  scope.bitDepth.value = "16";
  save.resolve();
  await task;
  assert.deepEqual(calls, [{
    url: "http://old-engine/api/enhance/run",
    payload: {
      session_id: "session-a", photo_ids: ["photo-a"], output_dir: "output-a",
      params_by_photo: { "photo-a": { strength: 0.91 } },
      auto_modes_by_photo: { "photo-a": true },
      auto_exposures_by_photo: { "photo-a": true },
      nonlocal_modes_by_photo: { "photo-a": "strong" },
      basic_params_by_photo: { "photo-a": { exposure: 1.25 } },
      preset_ids_by_photo: { "photo-a": "positive-film" },
      render_backend: "auto", basic_backend: "python", ricoh_backend: "native",
      compression: "lossless_jpeg", bit_depth: "source",
    },
  }]);
  assert.equal(scope.jobBaseUrl, "http://old-engine");
  assert.equal(scope.error.value, "");
});

test("source or backend replacement while saving aborts export", async t => {
  for (const change of ["session", "revision", "backend"]) {
    await t.test(change, async () => {
      const { scope, save, calls, startExport } = createExport();
      const task = startExport();
      if (change === "session") {
        scope.sharedPhotoSource.value = { session_id: "session-b", revision: 2 };
        scope.sharedDehazeByPhoto.value = { "photo-b": { strength: 0.2 } };
        scope.sharedBasicByPhoto.value = { "photo-b": { exposure: 0 } };
      } else if (change === "revision") scope.sharedPhotoSource.value.revision += 1;
      else scope.BASE_URL.value = "http://new-engine";
      save.resolve();
      await task;
      assert.equal(calls.length, 0);
      assert.match(scope.error.value, /照片来源或本地服务已变化/);
      assert.equal(scope.photoExportRunning.value, false);
      assert.equal(scope.starting.value, false);
    });
  }
});

test("failed saves block export without leaving the running state set", async () => {
  const { scope, save, calls, startExport } = createExport();
  const task = startExport();
  scope.autoSaveError.value = "XMP save failed";
  save.resolve();
  await task;
  assert.equal(calls.length, 0);
  assert.equal(scope.error.value, "XMP save failed");
  assert.equal(scope.photoExportRunning.value, false);
});

test("poll and cancel keep using the backend that created the job", async () => {
  const calls = [];
  const scope = {
    BASE_URL: ref("http://new-engine"), jobBaseUrl: "http://old-engine", polling: false,
    job: ref({ job_id: "job-a", status: "running" }), cancelling: ref(false),
    photoExportRunning: ref(true), error: ref(""), warnedFiles: ref([]),
    pollTimer: undefined, window: { clearInterval() {} },
    fetch: async url => {
      calls.push(url);
      return { ok: true, json: async () => ({ job_id: "job-a", status: "running" }) };
    },
  };
  const pollSource = sourceBetween(source, "async function pollJob()", "\nasync function startExport()");
  const cancelSource = sourceBetween(source, "async function cancelExport()", "\nonBeforeUnmount(");
  const { pollJob, cancelExport } = loadFunctions(
    `${requestSource}\n${pollSource}\n${cancelSource}`, ["pollJob", "cancelExport"], scope,
  );
  await pollJob();
  await cancelExport();
  assert.deepEqual(calls, [
    "http://old-engine/api/enhance/job/job-a",
    "http://old-engine/api/enhance/cancel/job-a",
    "http://old-engine/api/enhance/job/job-a",
  ]);
});
