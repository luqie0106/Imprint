<script setup lang="ts">
import { photoExportRunning } from "../stores/photoExport";
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { photoChangeSummary, rangeChangeStyle } from "../photoUi";
import { BASE_URL, isServerReady } from "../stores/api";
import { basicBackend, renderBackend, ricohBackend } from "../stores/renderOptions";
import {
  autoSaveError, flushPendingSaves, markPhotoChanged, sharedBasicByPhoto,
  sharedDehazeAutoByPhoto, sharedDehazeAutoExposureByPhoto,
  sharedDehazeByPhoto, sharedDehazeNonlocalByPhoto,
  sharedPhotoSource, sharedPresetByPhoto, sharedSelectedPhotoId, sharePhotoSource,
  type PhotoSource,
} from "../stores/photoSource";
import {
  Columns2, FolderOpen, Image as ImageIcon, ImagePlus, Images, LoaderCircle,
  Maximize2, Minus, Plus, Rows2, Sparkles,
} from "lucide-vue-next";

interface Preset { id: string; model: string; name: string; description: string }
interface BasicParams {
  exposure: number;
  contrast: number;
  highlights: number;
  shadows: number;
  whites: number;
  blacks: number;
  vibrance: number;
  saturation: number;
}
interface Photo { photo_id: string; name: string; ricoh_preset_id?: string | null; dehaze_params?: object | null; basic_params?: Partial<BasicParams> | null }
type PreviewMode = "compare" | "original" | "effect";
type ThumbnailState = "loading" | "loaded" | "error";
type PhotoListLayout = "vertical" | "horizontal";

const defaultBasicParams: BasicParams = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0,
  whites: 0, blacks: 0, vibrance: 0, saturation: 0,
};
const basicParamControls: Array<{ key: keyof BasicParams; label: string; min: number; max: number; step: number }> = [
  { key: "exposure", label: "曝光", min: -5, max: 5, step: 0.05 },
  { key: "contrast", label: "对比度", min: -100, max: 100, step: 1 },
  { key: "highlights", label: "高光", min: -100, max: 100, step: 1 },
  { key: "shadows", label: "阴影", min: -100, max: 100, step: 1 },
  { key: "whites", label: "白色色阶", min: -100, max: 100, step: 1 },
  { key: "blacks", label: "黑色色阶", min: -100, max: 100, step: 1 },
  { key: "vibrance", label: "自然饱和度", min: -100, max: 100, step: 1 },
  { key: "saturation", label: "饱和度", min: -100, max: 100, step: 1 },
];

function cloneBasicParams(source?: Partial<BasicParams> | null): BasicParams {
  return { ...defaultBasicParams, ...source };
}

const presets = ref<Preset[]>([]);
const selectedPreset = computed({
  get: () => sharedPresetByPhoto.value[selectedId.value] || "",
  set: (presetId: string) => {
    if (!selectedId.value) return;
    sharedPresetByPhoto.value[selectedId.value] = presetId || null;
    markPhotoChanged(selectedId.value);
  },
});
const sessionId = ref("");
const files = ref<Photo[]>([]);
const thumbnailStates = ref<Record<string, ThumbnailState>>({});
const photoListLayout = ref<PhotoListLayout>("vertical");
const basicParamsByPhoto = sharedBasicByPhoto;
const selectedId = ref("");
const originalUrl = ref("");
const effectUrl = ref("");
const mode = ref<PreviewMode>("effect");
const split = ref(50);
const previewViewport = ref<HTMLElement | null>(null);
const zoom = ref(1);
const lastZoom = ref<number | null>(null);
const panX = ref(0);
const panY = ref(0);
const viewportWidth = ref(0);
const viewportHeight = ref(0);
const devicePixelRatio = ref(1);
const imageWidth = ref(0);
const imageHeight = ref(0);
const sourceImageWidth = ref(0);
const sourceImageHeight = ref(0);
const isScrubbing = ref(false);
const loading = ref(false);
const error = ref("");
const message = ref("");
const currentFile = computed(() => files.value.find(file => file.photo_id === selectedId.value));
const basicParams = computed<BasicParams>(() => basicParamsByPhoto.value[selectedId.value] ?? defaultBasicParams);
const dehazeParams = computed(() => sharedDehazeByPhoto.value[selectedId.value]);
const effectReady = computed(() => Boolean(effectUrl.value));
const showComparePreview = computed(() => mode.value === "compare" && effectReady.value);
const previewImageUrl = computed(() => mode.value !== "original" && effectReady.value ? effectUrl.value : originalUrl.value);
let generation = 0;
let previewTimer: number | undefined;
let fullResolutionTimer: number | undefined;
let scheduledPreviewTask: Promise<void> | undefined;
let resizeObserver: ResizeObserver | undefined;
let pointerDownX = 0;
let pointerDownY = 0;
let pointerDownZoom = 1;
let pointerMoved = false;

const fitWidth = computed(() => {
  if (!imageWidth.value || !imageHeight.value || !viewportWidth.value || !viewportHeight.value) return Math.max(1, viewportWidth.value);
  const scale = Math.min(viewportWidth.value / imageWidth.value, viewportHeight.value / imageHeight.value);
  return Math.max(1, imageWidth.value * scale);
});
const fitHeight = computed(() => {
  if (!imageWidth.value || !imageHeight.value || !viewportWidth.value || !viewportHeight.value) return Math.max(520, viewportHeight.value);
  const scale = Math.min(viewportWidth.value / imageWidth.value, viewportHeight.value / imageHeight.value);
  return Math.max(1, imageHeight.value * scale);
});
const pixelScale = computed(() => sourceImageWidth.value > 0
  ? fitWidth.value * devicePixelRatio.value / sourceImageWidth.value
  : 0);
const maxZoom = computed(() => pixelScale.value > 0 ? Math.max(1, 4 / pixelScale.value) : 1);
const canZoom = computed(() => sourceImageWidth.value > 0 && sourceImageHeight.value > 0 && maxZoom.value > 1.001);
const zoomDisplayText = computed(() => {
  if (pixelScale.value <= 0) return "适合";
  const percent = Math.round(zoom.value * pixelScale.value * 100);
  return zoom.value <= 1.001 ? `适合（${percent}%）` : `${percent}%`;
});
const imageStageStyle = computed(() => ({
  width: `${fitWidth.value * zoom.value}px`,
  height: `${fitHeight.value * zoom.value}px`,
  transform: `translate(${panX.value}px, ${panY.value}px)`,
}));

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function clampPan() {
  if (!previewViewport.value) return;
  const maxX = Math.max(0, (fitWidth.value * zoom.value - viewportWidth.value) / 2);
  const maxY = Math.max(0, (fitHeight.value * zoom.value - viewportHeight.value) / 2);
  panX.value = clamp(panX.value, -maxX, maxX);
  panY.value = clamp(panY.value, -maxY, maxY);
}

function preservePixelScale(
  previousPixelScale: number,
  nextPixelScale: number,
  previousFitWidth: number,
  nextFitWidth: number,
) {
  if (previousPixelScale <= 0 || nextPixelScale <= 0) return;
  if (zoom.value > 1.001) {
    const previousZoom = zoom.value;
    zoom.value = clamp(zoom.value * previousPixelScale / nextPixelScale, 1, maxZoom.value);
    const panRatio = (zoom.value * nextFitWidth) / (previousZoom * previousFitWidth);
    panX.value *= panRatio;
    panY.value *= panRatio;
  } else {
    zoom.value = 1;
  }
  if (lastZoom.value !== null)
    lastZoom.value = clamp(lastZoom.value * previousPixelScale / nextPixelScale, 1, maxZoom.value);
  clampPan();
}

function resetView(rememberCurrent = true) {
  if (rememberCurrent && zoom.value > 1.001) lastZoom.value = zoom.value;
  zoom.value = 1;
  panX.value = 0;
  panY.value = 0;
}

function setZoom(nextZoom: number, clientX?: number, clientY?: number) {
  if (!canZoom.value) return;
  const previousZoom = zoom.value;
  const targetZoom = clamp(nextZoom, 1, maxZoom.value);
  if (Math.abs(targetZoom - previousZoom) < 0.001) return;
  let offsetX = 0;
  let offsetY = 0;
  if (previewViewport.value && clientX !== undefined && clientY !== undefined) {
    const rect = previewViewport.value.getBoundingClientRect();
    offsetX = clientX - (rect.left + rect.width / 2);
    offsetY = clientY - (rect.top + rect.height / 2);
  }
  const ratio = targetZoom / previousZoom;
  panX.value = offsetX - (offsetX - panX.value) * ratio;
  panY.value = offsetY - (offsetY - panY.value) * ratio;
  zoom.value = targetZoom;
  if (targetZoom > 1.001) lastZoom.value = targetZoom;
  else if (previousZoom > 1.001) lastZoom.value = previousZoom;
  clampPan();
}

function zoomBy(delta: number) {
  if (pixelScale.value > 0) setZoom(zoom.value + delta / pixelScale.value);
}

function onPreviewPointerDown(event: PointerEvent) {
  const target = event.target as HTMLElement | null;
  if (target?.closest(".compare-split") || event.button !== 0 || !originalUrl.value || !canZoom.value) return;
  isScrubbing.value = true;
  pointerMoved = false;
  pointerDownX = event.clientX;
  pointerDownY = event.clientY;
  pointerDownZoom = zoom.value;
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
  event.preventDefault();
}

function onPreviewPointerMove(event: PointerEvent) {
  if (!isScrubbing.value) return;
  const deltaX = event.clientX - pointerDownX;
  if (Math.abs(deltaX) > 4) pointerMoved = true;
  if (pointerMoved) setZoom(pointerDownZoom * Math.exp(deltaX * 0.005), pointerDownX, pointerDownY);
}

function finishPreviewPointer(event: PointerEvent, cancelled = false) {
  if (!isScrubbing.value) return;
  const wasMoved = pointerMoved;
  isScrubbing.value = false;
  if (previewViewport.value?.hasPointerCapture(event.pointerId)) previewViewport.value.releasePointerCapture(event.pointerId);
  if (cancelled || wasMoved) {
    if (zoom.value > 1.001) lastZoom.value = zoom.value;
    return;
  }
  if (pointerDownZoom > 1.001) {
    lastZoom.value = pointerDownZoom;
    resetView();
  } else {
    setZoom(lastZoom.value ?? Math.max(1, 1 / pixelScale.value), pointerDownX, pointerDownY);
  }
}

function onPreviewPointerUp(event: PointerEvent) { finishPreviewPointer(event); }
function onPreviewPointerCancel(event: PointerEvent) { finishPreviewPointer(event, true); }

function updateViewportSize() {
  devicePixelRatio.value = window.devicePixelRatio || 1;
  if (!previewViewport.value) return;
  viewportWidth.value = previewViewport.value.clientWidth;
  viewportHeight.value = previewViewport.value.clientHeight;
}

function onPreviewImageLoad(event: Event) {
  const image = event.currentTarget as HTMLImageElement;
  if (!image.naturalWidth || !image.naturalHeight) return;
  imageWidth.value = image.naturalWidth;
  imageHeight.value = image.naturalHeight;
  void nextTick(updateViewportSize);
}

function formatParam(key: keyof BasicParams, value: number) {
  return key === "exposure" ? value.toFixed(2) : String(Math.round(value));
}

function resetBasicParams() {
  if (selectedId.value) {
    basicParamsByPhoto.value[selectedId.value] = cloneBasicParams();
    markPhotoChanged(selectedId.value);
  }
}

async function postJson(endpoint: string, body: object) {
  const response = await fetch(`${BASE_URL.value}${endpoint}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "操作失败");
  return data;
}

async function loadPresets() {
  if (!isServerReady.value || !BASE_URL.value) return;
  try {
    const response = await fetch(`${BASE_URL.value}/api/ricoh/presets`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法读取预设");
    presets.value = data.presets ?? [];
  } catch (cause) { error.value = cause instanceof Error ? cause.message : "无法读取预设"; }
}

function adoptSession(source: PhotoSource) {
  const sessionChanged = sessionId.value !== source.session_id;
  sessionId.value = source.session_id;
  files.value = source.files as Photo[];
  thumbnailStates.value = Object.fromEntries(source.files.map(file => [file.photo_id, "loading"]));
  selectedId.value = sharedSelectedPhotoId.value || source.files[0]?.photo_id || "";
  message.value = "";
  if (sessionChanged) {
    sourceImageWidth.value = 0;
    sourceImageHeight.value = 0;
    lastZoom.value = null;
    resetView(false);
  }
}

async function createSession(source: { paths?: string[]; input_dir?: string }) {
  if (!BASE_URL.value || photoExportRunning.value) return;
  loading.value = true;
  error.value = "";
  try {
    await flushPendingSaves();
    const data = await postJson("/api/enhance/session", source);
    await sharePhotoSource("ricoh", source, data);
    if (sharedPhotoSource.value) adoptSession(sharedPhotoSource.value);
  } catch (cause) { error.value = cause instanceof Error ? cause.message : "无法读取照片"; }
  finally { loading.value = false; }
}

async function choosePhotos() {
  const selected = await open({ multiple: true, directory: false, title: "选择照片" });
  const paths = typeof selected === "string" ? [selected] : selected;
  if (paths?.length) await createSession({ paths });
}
async function chooseFolder() {
  const selected = await open({ multiple: false, directory: true, title: "选择照片文件夹" });
  if (typeof selected === "string") await createSession({ input_dir: selected });
}
function thumbnailUrl(file: Photo) {
  const baseUrl = BASE_URL.value;
  if (!baseUrl || !sessionId.value || !file.photo_id) return "";
  return `${baseUrl}/api/enhance/thumbnail/${encodeURIComponent(sessionId.value)}/${encodeURIComponent(file.photo_id)}`;
}
function thumbnailState(photoId: string): ThumbnailState {
  return thumbnailStates.value[photoId] ?? "loading";
}
function setThumbnailState(photoId: string, state: ThumbnailState) {
  thumbnailStates.value = { ...thumbnailStates.value, [photoId]: state };
}
function isPhotoListLayout(layout: PhotoListLayout) {
  return photoListLayout.value === layout;
}
function releasePreview() {
  if (originalUrl.value) URL.revokeObjectURL(originalUrl.value);
  if (effectUrl.value) URL.revokeObjectURL(effectUrl.value);
  originalUrl.value = "";
  effectUrl.value = "";
}
interface PreviewAsset { url: string; width: number; height: number }

async function requestPreview(endpoint: string, body: object, token: number): Promise<PreviewAsset> {
  const response = await fetch(`${BASE_URL.value}${endpoint}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || "预览失败");
  }
  const width = response.headers.get("X-Image-Width");
  const height = response.headers.get("X-Image-Height");
  const url = URL.createObjectURL(await response.blob());
  if (token !== generation) URL.revokeObjectURL(url);
  return {
    url: token === generation ? url : "",
    width: Number(width) || 0,
    height: Number(height) || 0,
  };
}
async function refreshPreview(includeOriginal = false, fullResolution = false, token = ++generation) {
  if (!sessionId.value || !selectedId.value || !BASE_URL.value) {
    releasePreview();
    return;
  }
  error.value = "";
  try {
    const common = {
      session_id: sessionId.value, photo_id: selectedId.value, max_edge: 1800,
      preview_level: fullResolution ? 0 : 2, full_resolution: fullResolution,
    };
    const autoMode = sharedDehazeAutoByPhoto.value[selectedId.value] ?? false;
    const autoExposure = sharedDehazeAutoExposureByPhoto.value[selectedId.value] ?? false;
    const nonlocalMode = sharedDehazeNonlocalByPhoto.value[selectedId.value] ?? "off";
    const dehazeContext = { ...common, auto_mode: autoMode, auto_exposure: autoExposure, nonlocal_mode: nonlocalMode };
    const requests: Array<Promise<PreviewAsset>> = [];
    const requestOriginal = includeOriginal || fullResolution || !originalUrl.value;
    if (requestOriginal) requests.push(requestPreview("/api/enhance/preview", {
      ...dehazeContext, mode: "original", color_manage_srgb: true,
    }, token));
    requests.push(requestPreview("/api/enhance/preview", {
      ...dehazeContext, mode: "dehazed", color_manage_srgb: true,
      params: dehazeParams.value,
      basic_params: cloneBasicParams(basicParams.value),
      ricoh_preset_id: selectedPreset.value || null,
      render_backend: renderBackend.value,
      basic_backend: basicBackend.value, ricoh_backend: ricohBackend.value,
    }, token));
    const results = await Promise.allSettled(requests);
    if (token !== generation) {
      for (const result of results) {
        if (result.status === "fulfilled" && result.value.url) URL.revokeObjectURL(result.value.url);
      }
      return;
    }
    const rejected = results.find(result => result.status === "rejected");
    if (rejected?.status === "rejected") {
      for (const result of results) {
        if (result.status === "fulfilled" && result.value.url) URL.revokeObjectURL(result.value.url);
      }
      throw rejected.reason;
    }
    let resultIndex = 0;
    if (requestOriginal) {
      const result = results[resultIndex++];
      if (result?.status === "fulfilled" && result.value.url) {
        if (originalUrl.value) URL.revokeObjectURL(originalUrl.value);
        originalUrl.value = result.value.url;
      }
    }
    const effectResult = results[resultIndex];
    if (effectResult?.status === "fulfilled" && effectResult.value.url) {
      if (effectUrl.value) URL.revokeObjectURL(effectUrl.value);
      effectUrl.value = effectResult.value.url;
      const dimensions = effectResult.value.width && effectResult.value.height
        ? effectResult.value
        : requestOriginal && results[0]?.status === "fulfilled" ? results[0].value : null;
      if (dimensions?.width && dimensions.height) {
        if (fullResolution) {
          sourceImageWidth.value = dimensions.width;
          sourceImageHeight.value = dimensions.height;
        }
        imageWidth.value = dimensions.width;
        imageHeight.value = dimensions.height;
        await nextTick(updateViewportSize);
      }
    }
  } catch (cause) {
    if (token === generation) error.value = cause instanceof Error ? cause.message : "预览失败";
  }
}

function schedulePreview(includeOriginal = false, immediate = false) {
  window.clearTimeout(previewTimer);
  window.clearTimeout(fullResolutionTimer);
  const token = ++generation;
  previewTimer = window.setTimeout(() => {
    if (token !== generation) return;
    scheduledPreviewTask = refreshPreview(includeOriginal, false, token);
  }, immediate ? 0 : 180);
  fullResolutionTimer = window.setTimeout(() => {
    void (async () => {
      if (token !== generation) return;
      await scheduledPreviewTask;
      if (token === generation) await refreshPreview(true, true, token);
    })();
  }, 900);
}

watch([isServerReady, BASE_URL], () => { void loadPresets(); }, { immediate: true });
watch(sharedPhotoSource, source => {
  if (source?.owner === "enhance") adoptSession(source);
}, { immediate: true });
watch([sessionId, selectedId], () => {
  if (selectedId.value && sharedPhotoSource.value?.session_id === sessionId.value)
    sharedSelectedPhotoId.value = selectedId.value;
  releasePreview();
  sourceImageWidth.value = 0;
  sourceImageHeight.value = 0;
  lastZoom.value = null;
  resetView(false);
  split.value = 50;
  imageWidth.value = 0;
  imageHeight.value = 0;
  schedulePreview(true, true);
});
watch(selectedPreset, () => { schedulePreview(); });
watch(dehazeParams, () => schedulePreview(), { deep: true });
watch(() => ({
  photoId: selectedId.value,
  autoMode: sharedDehazeAutoByPhoto.value[selectedId.value] ?? false,
  autoExposure: sharedDehazeAutoExposureByPhoto.value[selectedId.value] ?? false,
  nonlocalMode: sharedDehazeNonlocalByPhoto.value[selectedId.value] ?? "off",
}), (current, previous) => {
  if (current.photoId === previous.photoId
    && (current.autoMode !== previous.autoMode || current.autoExposure !== previous.autoExposure
      || current.nonlocalMode !== previous.nonlocalMode))
    schedulePreview();
});
watch([renderBackend, basicBackend, ricohBackend], () => schedulePreview());
watch(basicParams, (next, previous) => {
  schedulePreview();
  if (selectedId.value && next === previous) {
    markPhotoChanged(selectedId.value);
  }
}, { deep: true });
watch(sharedSelectedPhotoId, photoId => {
  if (photoId && photoId !== selectedId.value && files.value.some(file => file.photo_id === photoId))
    selectedId.value = photoId;
});
watch([pixelScale, fitWidth], ([nextScale, nextFitWidth], [previousScale, previousFitWidth]) => {
  preservePixelScale(previousScale, nextScale, previousFitWidth, nextFitWidth);
});
watch([fitWidth, fitHeight], clampPan);
onMounted(() => {
  window.addEventListener("resize", updateViewportSize);
  updateViewportSize();
  if (previewViewport.value) {
    resizeObserver = new ResizeObserver(updateViewportSize);
    resizeObserver.observe(previewViewport.value);
  }
});
onBeforeUnmount(() => {
  void flushPendingSaves();
  generation++;
  releasePreview();
  window.clearTimeout(previewTimer);
  window.clearTimeout(fullResolutionTimer);
  window.removeEventListener("resize", updateViewportSize);
  resizeObserver?.disconnect();
});
</script>

<template>
  <div class="h-full min-h-0 overflow-hidden bg-slate-50 p-5 dark:bg-zinc-950">
    <div class="mx-auto grid h-full min-h-0 max-w-[1500px] grid-cols-[clamp(150px,17vw,220px)_minmax(0,1fr)_clamp(270px,23vw,300px)] gap-4">
      <aside class="flex min-h-0 flex-col gap-4">
        <section class="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="mb-3 flex items-center gap-2"><ImagePlus class="h-4 w-4 text-blue-600" /><h2 class="text-sm font-semibold">输入照片</h2></div>
          <div class="grid gap-2">
            <button type="button" @click="choosePhotos" :disabled="photoExportRunning" class="rounded-xl border border-slate-200 px-3 py-2 text-left text-xs transition hover:border-blue-400 dark:border-zinc-700"><Images class="mr-2 inline h-3.5 w-3.5" />选择照片</button>
            <button type="button" @click="chooseFolder" :disabled="photoExportRunning" class="rounded-xl border border-slate-200 px-3 py-2 text-left text-xs transition hover:border-blue-400 dark:border-zinc-700"><FolderOpen class="mr-2 inline h-3.5 w-3.5" />选择照片文件夹</button>
          </div>
          <div class="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:bg-zinc-800 dark:text-zinc-400">
            <LoaderCircle v-if="loading" class="mr-1 inline h-3 w-3 animate-spin" />
            {{ files.length ? `当前共 ${files.length} 张` : "尚未选择照片" }}
          </div>
        </section>

        <section v-if="files.length && photoListLayout === 'vertical'" class="photo-list-panel min-h-0 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="flex shrink-0 items-center justify-between border-b border-slate-100 px-3 py-2 dark:border-zinc-800">
            <h2 class="text-xs font-semibold">照片列表</h2>
            <div class="flex shrink-0 items-center gap-0.5 rounded-lg border border-slate-200 bg-slate-50 p-1 dark:border-zinc-700 dark:bg-zinc-800" role="group" aria-label="照片列表布局">
              <button type="button" @click="photoListLayout = 'vertical'" :aria-pressed="isPhotoListLayout('vertical')" title="竖向列表" aria-label="竖向列表" class="flex h-7 w-7 items-center justify-center rounded-md p-1.5 transition" :class="isPhotoListLayout('vertical') ? 'bg-blue-600 text-white shadow-sm dark:bg-blue-500' : 'text-slate-500 hover:bg-white dark:text-zinc-400 dark:hover:bg-zinc-700'"><Rows2 class="h-3.5 w-3.5" /></button>
              <button type="button" @click="photoListLayout = 'horizontal'" :aria-pressed="isPhotoListLayout('horizontal')" title="横向列表" aria-label="横向列表" class="flex h-7 w-7 items-center justify-center rounded-md p-1.5 transition" :class="isPhotoListLayout('horizontal') ? 'bg-blue-600 text-white shadow-sm dark:bg-blue-500' : 'text-slate-500 hover:bg-white dark:text-zinc-400 dark:hover:bg-zinc-700'"><Columns2 class="h-3.5 w-3.5" /></button>
            </div>
          </div>
          <div class="photo-list min-h-0 overflow-y-auto p-2">
            <button v-for="file in files" :key="file.photo_id" @click="selectedId = file.photo_id" type="button" :title="file.name" :aria-label="file.name" class="photo-list-card flex min-w-0 w-full items-center gap-2 rounded-lg p-1.5 text-left text-xs transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500" :class="selectedId === file.photo_id ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300' : 'hover:bg-slate-50 dark:hover:bg-zinc-800'">
              <span class="relative block h-10 w-10 shrink-0 overflow-hidden rounded-md bg-slate-100 dark:bg-zinc-800">
                <span v-if="thumbnailState(file.photo_id) !== 'loaded'" class="absolute inset-0 flex items-center justify-center text-slate-400 dark:text-zinc-500"><ImageIcon class="h-4 w-4" /></span>
                <img v-if="thumbnailUrl(file)" :src="thumbnailUrl(file)" :alt="file.name" loading="lazy" decoding="async" class="absolute inset-0 block h-full w-full object-contain transition-opacity" :class="thumbnailState(file.photo_id) === 'loaded' ? 'opacity-100' : 'opacity-0'" @load="setThumbnailState(file.photo_id, 'loaded')" @error="setThumbnailState(file.photo_id, 'error')" />
              </span>
              <span class="min-w-0 flex-1">
                <span class="block truncate">{{ file.name }}</span>
                <span v-if="photoChangeSummary(file.photo_id)" class="block truncate text-[10px] text-slate-500 dark:text-zinc-400">{{ photoChangeSummary(file.photo_id) }}</span>
              </span>
            </button>
          </div>
        </section>
      </aside>
      <main class="flex min-h-0 min-w-0 flex-col gap-4">
        <section class="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-4 py-3 dark:border-zinc-800">
            <div class="min-w-0"><h2 class="truncate text-sm font-semibold">{{ currentFile?.name || "理光风格预览" }}</h2><p class="mt-1 text-[11px] text-slate-500">{{ imageWidth && imageHeight ? `${imageWidth} × ${imageHeight} · ` : "" }}同时预览去朦胧、理光预设和基础参数；理光色彩仍是近似模拟</p></div>
            <div class="flex flex-wrap items-center gap-2">
              <div class="flex shrink-0 items-center gap-0.5 rounded-lg border border-slate-200 bg-slate-50 p-1 dark:border-zinc-700 dark:bg-zinc-800" role="group" aria-label="预览模式">
                <button type="button" @click="mode = 'original'" :aria-pressed="mode === 'original'" title="仅原图" aria-label="仅原图" class="flex h-7 w-7 items-center justify-center rounded-md p-1.5 transition" :class="mode === 'original' ? 'bg-blue-600 text-white shadow-sm dark:bg-blue-500' : 'text-slate-500 hover:bg-white dark:text-zinc-400 dark:hover:bg-zinc-700'"><ImageIcon class="h-3.5 w-3.5" /></button>
                <button type="button" @click="mode = 'compare'" :aria-pressed="mode === 'compare'" title="原图/效果图对比" aria-label="原图/效果图对比" class="flex h-7 w-7 items-center justify-center rounded-md p-1.5 transition" :class="mode === 'compare' ? 'bg-blue-600 text-white shadow-sm dark:bg-blue-500' : 'text-slate-500 hover:bg-white dark:text-zinc-400 dark:hover:bg-zinc-700'"><Columns2 class="h-3.5 w-3.5" /></button>
                <button type="button" @click="mode = 'effect'" :aria-pressed="mode === 'effect'" title="仅效果图" aria-label="仅效果图" class="flex h-7 w-7 items-center justify-center rounded-md p-1.5 transition" :class="mode === 'effect' ? 'bg-blue-600 text-white shadow-sm dark:bg-blue-500' : 'text-slate-500 hover:bg-white dark:text-zinc-400 dark:hover:bg-zinc-700'"><Sparkles class="h-3.5 w-3.5" /></button>
              </div>
              <div class="flex shrink-0 items-center gap-1 rounded-lg bg-slate-50 p-1 dark:bg-zinc-800">
                <button type="button" @click="zoomBy(-0.25)" :disabled="!canZoom || zoom <= 1.001" title="缩小" aria-label="缩小" class="rounded p-1.5 hover:bg-white disabled:cursor-not-allowed disabled:opacity-35 dark:hover:bg-zinc-700"><Minus class="h-3.5 w-3.5" /></button>
                <button type="button" @click="resetView()" title="适合窗口" aria-label="适合窗口" class="w-[6.5rem] shrink-0 whitespace-nowrap rounded px-1.5 py-1 text-[11px] font-medium tabular-nums hover:bg-white dark:hover:bg-zinc-700">{{ zoomDisplayText }}</button>
                <button type="button" @click="resetView()" title="适合窗口" aria-label="适合窗口" class="rounded p-1.5 hover:bg-white dark:hover:bg-zinc-700"><Maximize2 class="h-3.5 w-3.5" /></button>
                <button type="button" @click="zoomBy(0.25)" :disabled="!canZoom || zoom >= maxZoom - 0.001" title="放大" aria-label="放大" class="rounded p-1.5 hover:bg-white disabled:cursor-not-allowed disabled:opacity-35 dark:hover:bg-zinc-700"><Plus class="h-3.5 w-3.5" /></button>
              </div>
            </div>
          </div>
          <div ref="previewViewport" class="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-slate-100 dark:bg-black" :class="originalUrl ? (isScrubbing ? 'cursor-ew-resize' : (!canZoom ? 'cursor-default' : (zoom > 1 ? 'cursor-zoom-out' : 'cursor-zoom-in'))) : 'cursor-default'" @pointerdown="onPreviewPointerDown" @pointermove="onPreviewPointerMove" @pointerup="onPreviewPointerUp" @pointercancel="onPreviewPointerCancel">
            <template v-if="originalUrl">
              <div class="absolute inset-0 flex items-center justify-center overflow-hidden">
                <div class="preview-stage relative shrink-0" :style="imageStageStyle">
                  <img :src="previewImageUrl" :alt="mode === 'effect' && effectReady ? '综合效果' : '原图'" class="block h-full w-full object-contain" draggable="false" @load="onPreviewImageLoad" />
                </div>
              </div>
              <div v-if="showComparePreview" class="absolute inset-0 overflow-hidden" :style="{ clipPath: `inset(0 ${100 - split}% 0 0)` }">
                <div class="absolute inset-0 flex items-center justify-center overflow-hidden">
                  <div class="preview-stage relative shrink-0" :style="imageStageStyle">
                    <img :src="originalUrl" alt="原图" class="block h-full w-full object-contain" draggable="false" @load="onPreviewImageLoad" />
                  </div>
                </div>
              </div>
              <div v-if="showComparePreview" class="pointer-events-none absolute inset-y-0 z-10 w-px bg-white shadow" :style="{ left: `${split}%`, transform: 'translateX(-50%)' }"></div>
              <template v-if="showComparePreview">
                <span class="pointer-events-none absolute left-3 top-3 z-20 rounded bg-black/55 px-2 py-1 text-[11px] text-white">原图</span>
                <span class="pointer-events-none absolute right-3 top-3 z-20 rounded bg-black/55 px-2 py-1 text-[11px] text-white">综合效果</span>
                <input v-model.number="split" type="range" min="0" max="100" class="compare-split absolute bottom-4 z-20" aria-label="前后对比分割线" @pointerdown.stop @click.stop />
              </template>
              <span v-else class="pointer-events-none absolute left-3 top-3 z-20 rounded bg-black/55 px-2 py-1 text-[11px] text-white">{{ mode === "effect" && effectReady ? "综合效果" : "原图" }}</span>
            </template>
            <p v-if="!originalUrl && !selectedId" class="text-sm text-slate-400">选择照片开始预览</p>
          </div>
        </section>
        <section v-if="files.length && photoListLayout === 'horizontal'" class="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="flex items-center justify-between border-b border-slate-100 px-3 py-2 dark:border-zinc-800">
            <h2 class="text-xs font-semibold">照片列表</h2>
            <div class="flex shrink-0 items-center gap-0.5 rounded-lg border border-slate-200 bg-slate-50 p-1 dark:border-zinc-700 dark:bg-zinc-800" role="group" aria-label="照片列表布局">
              <button type="button" @click="photoListLayout = 'vertical'" :aria-pressed="isPhotoListLayout('vertical')" title="竖向列表" aria-label="竖向列表" class="flex h-7 w-7 items-center justify-center rounded-md p-1.5 transition" :class="isPhotoListLayout('vertical') ? 'bg-blue-600 text-white shadow-sm dark:bg-blue-500' : 'text-slate-500 hover:bg-white dark:text-zinc-400 dark:hover:bg-zinc-700'"><Rows2 class="h-3.5 w-3.5" /></button>
              <button type="button" @click="photoListLayout = 'horizontal'" :aria-pressed="isPhotoListLayout('horizontal')" title="横向列表" aria-label="横向列表" class="flex h-7 w-7 items-center justify-center rounded-md p-1.5 transition" :class="isPhotoListLayout('horizontal') ? 'bg-blue-600 text-white shadow-sm dark:bg-blue-500' : 'text-slate-500 hover:bg-white dark:text-zinc-400 dark:hover:bg-zinc-700'"><Columns2 class="h-3.5 w-3.5" /></button>
            </div>
          </div>
          <div class="photo-grid overflow-x-auto overflow-y-hidden p-2">
            <button v-for="file in files" :key="file.photo_id" @click="selectedId = file.photo_id" type="button" :title="file.name" :aria-label="file.name" class="photo-card min-w-0 shrink-0 text-left text-xs transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-blue-500" :class="selectedId === file.photo_id ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300' : 'hover:bg-slate-50 dark:hover:bg-zinc-800'">
              <span class="relative mb-1.5 block h-20 w-full overflow-hidden rounded-lg bg-slate-100 dark:bg-zinc-800">
                <span v-if="thumbnailState(file.photo_id) !== 'loaded'" class="absolute inset-0 flex items-center justify-center text-slate-400 dark:text-zinc-500"><ImageIcon class="h-6 w-6" /></span>
                <img v-if="thumbnailUrl(file)" :src="thumbnailUrl(file)" :alt="file.name" loading="lazy" decoding="async" class="absolute inset-0 block h-full w-full object-contain transition-opacity" :class="thumbnailState(file.photo_id) === 'loaded' ? 'opacity-100' : 'opacity-0'" @load="setThumbnailState(file.photo_id, 'loaded')" @error="setThumbnailState(file.photo_id, 'error')" />
              </span>
              <span class="block truncate" :title="file.name">{{ file.name }}</span>
              <span v-if="photoChangeSummary(file.photo_id)" class="mt-0.5 block truncate text-[10px] text-slate-500 dark:text-zinc-400">{{ photoChangeSummary(file.photo_id) }}</span>
            </button>
          </div>
        </section>
      </main>
      <aside class="min-h-0 space-y-4 overflow-y-auto">
        <section class="rounded-2xl border border-slate-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
          <div class="mb-3 flex items-center justify-between">
            <h2 class="text-sm font-semibold">基础参数</h2>
            <button type="button" @click="resetBasicParams" :disabled="!selectedId" title="重置当前照片基础参数" aria-label="重置当前照片基础参数" class="rounded px-2 py-1 text-[11px] text-slate-500 hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-zinc-800">重置</button>
          </div>
          <p class="mb-4 text-[11px] text-slate-500">仅调整当前照片；数值相对所选理光风格累加，修改后自动保存到 XMP。</p>
          <div class="space-y-3">
            <label v-for="item in basicParamControls" :key="item.key" class="block text-[11px]">
              <span class="flex justify-between"><span>{{ item.label }}</span><span class="font-mono text-blue-600">{{ formatParam(item.key, basicParams[item.key]) }}</span></span>
              <input v-model.number="basicParams[item.key]" class="app-range mt-1 w-full" :style="rangeChangeStyle(basicParams[item.key], item.min, item.max, 0)" type="range" :min="item.min" :max="item.max" :step="item.step" :disabled="!selectedId" :aria-label="item.label" />
            </label>
          </div>
        </section>
        <section class="rounded-2xl border border-slate-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
          <h2 class="mb-2 text-sm font-semibold">理光风格</h2>
          <p class="mb-3 text-[11px] text-slate-500">所选风格自动保存到同名 XMP，与去朦胧参数共享。</p>
          <div class="max-h-72 space-y-2 overflow-y-auto">
            <button @click="selectedPreset = ''" class="w-full rounded-xl border p-2.5 text-left text-xs" :class="!selectedPreset ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30' : 'border-slate-200 dark:border-zinc-700'"><span class="font-semibold">无</span><span class="mt-1 block text-[11px] text-slate-500">不应用理光风格，仅保留基础调整</span></button>
            <button v-for="preset in presets" :key="preset.id" @click="selectedPreset = preset.id" class="w-full rounded-xl border p-2.5 text-left text-xs" :class="selectedPreset === preset.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30' : 'border-slate-200 dark:border-zinc-700'"><span class="font-semibold">{{ preset.model }} · {{ preset.name }}</span><span class="mt-1 block text-[11px] text-slate-500">{{ preset.description }}</span></button>
          </div>
        </section>
        <p v-if="autoSaveError" class="text-[11px] text-rose-600">{{ autoSaveError }}</p>
        <p v-if="message" class="text-xs text-slate-500">{{ message }}</p>
        <p v-if="error" class="rounded-xl bg-rose-50 p-3 text-xs text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">{{ error }}</p>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.photo-list-panel {
  display: flex;
  flex: 1 1 0%;
  flex-direction: column;
}

.photo-grid {
  display: flex;
  gap: 0.5rem;
  overscroll-behavior-x: contain;
}

.photo-card {
  flex: 0 0 144px;
  width: 144px;
  padding: 0.5rem;
  border-right: 1px solid rgb(226 232 240);
}

.photo-card:last-child {
  border-right: 0;
}

:global(.dark) .photo-card {
  border-right-color: rgb(63 63 70);
}

:global(.dark) .photo-card:last-child {
  border-right: 0;
}

.photo-list-card + .photo-list-card {
  margin-top: 0.25rem;
}

.preview-stage {
  transform-origin: center center;
}

.compare-split {
  appearance: none;
  height: 1.5rem;
  left: -0.5rem;
  margin: 0;
  right: -0.5rem;
  width: auto;
  cursor: ew-resize;
  background: transparent;
  touch-action: none;
}

.compare-split::-webkit-slider-runnable-track {
  height: 1px;
  background: transparent;
}

.compare-split::-webkit-slider-thumb {
  appearance: none;
  width: 1rem;
  height: 1rem;
  margin-top: -0.4375rem;
  border: 2px solid white;
  border-radius: 9999px;
  background: rgb(37 99 235);
  box-shadow: 0 1px 4px rgb(0 0 0 / 45%);
}

.compare-split::-moz-range-track {
  height: 1px;
  background: transparent;
}

.compare-split::-moz-range-thumb {
  width: 1rem;
  height: 1rem;
  border: 2px solid white;
  border-radius: 9999px;
  background: rgb(37 99 235);
  box-shadow: 0 1px 4px rgb(0 0 0 / 45%);
}
</style>
