<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import { BASE_URL } from "../stores/api";
import {
  AlertCircle, CheckCircle2, ChevronDown, FolderOpen, ImagePlus, Images,
  LoaderCircle, Play, RotateCcw, ShieldCheck, SlidersHorizontal, Square,
  Maximize2, Minus, Plus,
} from "lucide-vue-next";

interface SessionFile { photo_id: string; name: string; extension: string }
interface JobFile { photo_id: string; name: string; status: string; output?: string; error?: string }
interface EnhanceJob {
  job_id: string; status: string; total: number; processed: number; success: number;
  failed: number; progress: number; current_file: string; output_dir: string; files: JobFile[];
}

const defaults = {
  strength: 0.45, naturalness: 0.70, fog_retention: 0.55,
  local_contrast: 0.25, color_protection: 0.80,
  highlight_protection: 0.75, shadow_protection: 0.75,
};
const params = ref({ ...defaults });
const sessionId = ref("");
const files = ref<SessionFile[]>([]);
const selectedId = ref("");
const outputDir = ref("");
const originalUrl = ref("");
const enhancedUrl = ref("");
const previewLoading = ref(false);
const sessionLoading = ref(false);
const previewError = ref("");
const imageSize = ref("");
const split = ref(50);
const previewViewport = ref<HTMLElement | null>(null);
const zoom = ref(1);
const lastZoom = ref(2);
const panX = ref(0);
const panY = ref(0);
const viewportWidth = ref(0);
const viewportHeight = ref(0);
const imageWidth = ref(0);
const imageHeight = ref(0);
const isScrubbing = ref(false);
let resizeObserver: ResizeObserver | undefined;
let pointerDownX = 0;
let pointerDownY = 0;
let pointerDownZoom = 1;
let pointerMoved = false;
const advancedOpen = ref(false);
const job = ref<EnhanceJob | null>(null);
const actionMessage = ref("");
const revealMenu = ref<{ left: number; top: number } | null>(null);
const revealBusy = ref(false);
let previewTimer: number | undefined;
let previewGeneration = 0;
let pollTimer: number | undefined;

const fitWidth = computed(() => {
  if (!imageWidth.value || !imageHeight.value || !viewportWidth.value || !viewportHeight.value) {
    return Math.max(1, viewportWidth.value);
  }
  const scale = Math.min(viewportWidth.value / imageWidth.value, viewportHeight.value / imageHeight.value);
  return Math.max(1, imageWidth.value * scale);
});
const fitHeight = computed(() => {
  if (!imageWidth.value || !imageHeight.value || !viewportWidth.value || !viewportHeight.value) {
    return Math.max(520, viewportHeight.value);
  }
  const scale = Math.min(viewportWidth.value / imageWidth.value, viewportHeight.value / imageHeight.value);
  return Math.max(1, imageHeight.value * scale);
});
const imageStageStyle = computed(() => ({
  width: `${fitWidth.value}px`,
  height: `${fitHeight.value}px`,
  transform: `translate(${panX.value}px, ${panY.value}px) scale(${zoom.value})`,
}));

const currentFile = computed(() => files.value.find((file) => file.photo_id === selectedId.value));
const isRunning = computed(() => job.value?.status === "queued" || job.value?.status === "running");
const failedFiles = computed(() => job.value?.files.filter((file) => file.status === "failed") ?? []);
const revealMenuLabel = computed(() => {
  const platform = `${navigator.platform || ""} ${navigator.userAgent || ""}`;
  if (/Mac/i.test(platform)) return "在 Finder 中显示";
  if (/Win/i.test(platform)) return "在文件资源管理器中显示";
  return "在文件管理器中显示";
});

function releaseUrl(url: string) {
  if (url) URL.revokeObjectURL(url);
}

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

function resetView() {
  if (zoom.value > 1.001) lastZoom.value = zoom.value;
  zoom.value = 1;
  panX.value = 0;
  panY.value = 0;
}

function setZoom(nextZoom: number, clientX?: number, clientY?: number) {
  const previousZoom = zoom.value;
  const targetZoom = clamp(nextZoom, 1, 4);
  if (Math.abs(targetZoom - previousZoom) < 0.001) return;

  let offsetX = 0;
  let offsetY = 0;
  if (previewViewport.value && clientX !== undefined && clientY !== undefined) {
    const rect = previewViewport.value.getBoundingClientRect();
    offsetX = clientX - (rect.left + rect.width / 2);
    offsetY = clientY - (rect.top + rect.height / 2);
  }
  const ratio = targetZoom / previousZoom;
  // Keep the image point under the pointer stationary while the stage scales.
  panX.value = offsetX - (offsetX - panX.value) * ratio;
  panY.value = offsetY - (offsetY - panY.value) * ratio;
  zoom.value = targetZoom;
  if (targetZoom > 1.001) lastZoom.value = targetZoom;
  else if (previousZoom > 1.001) lastZoom.value = previousZoom;
  clampPan();
}

function zoomBy(delta: number) {
  setZoom(zoom.value + delta);
}

function onPreviewPointerDown(event: PointerEvent) {
  const target = event.target as HTMLElement | null;
  if (target?.closest(".compare-split")) return;
  if (event.button !== 0 || !originalUrl.value) return;
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
  if (!pointerMoved) return;
  // Adobe-style scrubby zoom: horizontal movement maps smoothly and
  // exponentially to the 100%–400% range, anchored at the press location.
  setZoom(pointerDownZoom * Math.exp(deltaX * 0.005), pointerDownX, pointerDownY);
}

function finishPreviewPointer(event: PointerEvent, cancelled = false) {
  if (!isScrubbing.value) return;
  const wasMoved = pointerMoved;
  isScrubbing.value = false;
  if (event && previewViewport.value?.hasPointerCapture(event.pointerId)) {
    previewViewport.value.releasePointerCapture(event.pointerId);
  }
  if (cancelled || wasMoved) {
    if (zoom.value > 1.001) lastZoom.value = zoom.value;
    return;
  }
  if (pointerDownZoom > 1.001) {
    // Fit on click, retaining the zoom level for the next click-to-restore.
    lastZoom.value = pointerDownZoom;
    resetView();
  } else {
    // Restore the previous non-Fit zoom around the click location.
    setZoom(lastZoom.value, pointerDownX, pointerDownY);
  }
}

function onPreviewPointerUp(event: PointerEvent) {
  finishPreviewPointer(event);
}

function onPreviewPointerCancel(event: PointerEvent) {
  finishPreviewPointer(event, true);
}

function closeRevealMenu() {
  revealMenu.value = null;
}

function onPreviewClick(event: MouseEvent) {
  const target = event.target as HTMLElement | null;
  if (target?.closest(".reveal-context-menu")) return;
  closeRevealMenu();
}

function onPreviewContextMenu(event: MouseEvent) {
  const target = event.target as HTMLElement | null;
  if (!originalUrl.value || !currentFile.value || !sessionId.value || !selectedId.value || target?.closest(".compare-split")) return;
  event.preventDefault();
  event.stopPropagation();
  const viewport = previewViewport.value;
  if (!viewport) return;
  const rect = viewport.getBoundingClientRect();
  const menuWidth = 232;
  const menuHeight = 42;
  revealMenu.value = {
    left: clamp(event.clientX - rect.left, 8, Math.max(8, rect.width - menuWidth - 8)),
    top: clamp(event.clientY - rect.top, 8, Math.max(8, rect.height - menuHeight - 8)),
  };
}

async function revealOriginal() {
  if (revealBusy.value || !BASE_URL.value || !sessionId.value || !selectedId.value) return;
  revealBusy.value = true;
  closeRevealMenu();
  try {
    const response = await fetch(`${BASE_URL.value}/api/enhance/reveal`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId.value, photo_id: selectedId.value }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "无法显示原图");
    if (typeof data.path !== "string" || !data.path) throw new Error("无法显示原图");
    await revealItemInDir(data.path);
  } catch (error) {
    // Do not expose the local path in the status message, including opener errors.
    actionMessage.value = error instanceof Error && error.message === "无法显示原图"
      ? error.message
      : "无法在文件管理器中显示原图";
  } finally {
    revealBusy.value = false;
  }
}

function onWindowKeyDown(event: KeyboardEvent) {
  if (event.key === "Escape") closeRevealMenu();
}

function updateViewportSize() {
  if (!previewViewport.value) return;
  viewportWidth.value = previewViewport.value.clientWidth;
  viewportHeight.value = previewViewport.value.clientHeight;
  clampPan();
}

function onPreviewImageLoad(event: Event) {
  const image = event.currentTarget as HTMLImageElement;
  if (!image.naturalWidth || !image.naturalHeight) return;
  imageWidth.value = image.naturalWidth;
  imageHeight.value = image.naturalHeight;
  void nextTick(updateViewportSize);
}

async function createSession(payload: { paths?: string[]; input_dir?: string }) {
  if (!BASE_URL.value) return;
  sessionLoading.value = true;
  previewError.value = "";
  actionMessage.value = "";
  closeRevealMenu();
  try {
    const response = await fetch(`${BASE_URL.value}/api/enhance/session`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法读取照片");
    sessionId.value = data.session_id;
    files.value = data.files;
    resetView();
    selectedId.value = data.files[0]?.photo_id ?? "";
    outputDir.value = data.default_output_dir;
    job.value = null;
    actionMessage.value = `已载入 ${data.count} 张照片`;
  } catch (error) {
    previewError.value = error instanceof Error ? error.message : String(error);
  } finally {
    sessionLoading.value = false;
  }
}

async function chooseSingle() {
  const result = await open({ multiple: false, directory: false, title: "选择一张照片" });
  if (typeof result === "string") await createSession({ paths: [result] });
}

async function chooseMultiple() {
  const result = await open({ multiple: true, directory: false, title: "选择多张照片" });
  if (Array.isArray(result) && result.length) await createSession({ paths: result });
  else if (typeof result === "string") await createSession({ paths: [result] });
}

async function chooseFolder() {
  const result = await open({ multiple: false, directory: true, title: "选择照片文件夹" });
  if (typeof result === "string") await createSession({ input_dir: result });
}

async function chooseOutput() {
  const result = await open({ multiple: false, directory: true, title: "选择去朦胧输出目录" });
  if (typeof result === "string") outputDir.value = result;
}

async function fetchPreview(mode: "original" | "dehazed", generation: number) {
  const response = await fetch(`${BASE_URL.value}/api/enhance/preview`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId.value, photo_id: selectedId.value, params: params.value,
      max_edge: 1800, mode,
    }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || "预览生成失败");
  }
  const width = response.headers.get("X-Image-Width");
  const height = response.headers.get("X-Image-Height");
  if (width && height) {
    imageWidth.value = Number(width) || imageWidth.value;
    imageHeight.value = Number(height) || imageHeight.value;
    imageSize.value = `${width} × ${height} 预览`;
    await nextTick();
    updateViewportSize();
  }
  const url = URL.createObjectURL(await response.blob());
  if (generation !== previewGeneration) { releaseUrl(url); return; }
  if (mode === "original") { releaseUrl(originalUrl.value); originalUrl.value = url; }
  else { releaseUrl(enhancedUrl.value); enhancedUrl.value = url; }
}

async function refreshPreview(includeOriginal = false) {
  if (!sessionId.value || !selectedId.value || !BASE_URL.value) return;
  const generation = ++previewGeneration;
  previewLoading.value = true;
  previewError.value = "";
  try {
    const tasks = [fetchPreview("dehazed", generation)];
    if (includeOriginal || !originalUrl.value) tasks.push(fetchPreview("original", generation));
    await Promise.all(tasks);
  } catch (error) {
    if (generation === previewGeneration) previewError.value = error instanceof Error ? error.message : String(error);
  } finally {
    if (generation === previewGeneration) previewLoading.value = false;
  }
}

function schedulePreview() {
  window.clearTimeout(previewTimer);
  previewTimer = window.setTimeout(() => void refreshPreview(false), 180);
}

function resetParams() {
  params.value = { ...defaults };
  void refreshPreview(false);
}

async function pollJob() {
  if (!job.value || !BASE_URL.value) return;
  try {
    const response = await fetch(`${BASE_URL.value}/api/enhance/job/${job.value.job_id}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法读取任务进度");
    job.value = data;
    if (!["queued", "running"].includes(data.status)) {
      window.clearInterval(pollTimer);
      pollTimer = undefined;
      actionMessage.value = data.status === "cancelled"
        ? `任务已停止，已完成 ${data.success} 张`
        : `处理完成：成功 ${data.success} 张，失败 ${data.failed} 张`;
    }
  } catch (error) {
    actionMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function startBatch() {
  if (!sessionId.value || isRunning.value || !BASE_URL.value) return;
  actionMessage.value = "正在创建导出任务…";
  try {
    const response = await fetch(`${BASE_URL.value}/api/enhance/run`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId.value, output_dir: outputDir.value, params: params.value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "任务启动失败");
    job.value = data;
    actionMessage.value = "正在全分辨率处理；原图不会被修改";
    window.clearInterval(pollTimer);
    pollTimer = window.setInterval(() => void pollJob(), 700);
    await pollJob();
  } catch (error) {
    actionMessage.value = error instanceof Error ? error.message : String(error);
  }
}

async function cancelBatch() {
  if (!job.value || !BASE_URL.value) return;
  await fetch(`${BASE_URL.value}/api/enhance/cancel/${job.value.job_id}`, { method: "POST" });
  actionMessage.value = "停止请求已发送，当前照片完成后停止";
  await pollJob();
}

watch(selectedId, () => {
  closeRevealMenu();
  resetView();
  imageWidth.value = 0;
  imageHeight.value = 0;
  releaseUrl(originalUrl.value);
  originalUrl.value = "";
  void refreshPreview(true);
});
watch(params, schedulePreview, { deep: true });
watch([fitWidth, fitHeight], clampPan);
onMounted(() => {
  window.addEventListener("keydown", onWindowKeyDown);
  window.addEventListener("blur", closeRevealMenu);
  updateViewportSize();
  if (previewViewport.value) {
    resizeObserver = new ResizeObserver(updateViewportSize);
    resizeObserver.observe(previewViewport.value);
  }
});
onBeforeUnmount(() => {
  closeRevealMenu();
  window.removeEventListener("keydown", onWindowKeyDown);
  window.removeEventListener("blur", closeRevealMenu);
  releaseUrl(originalUrl.value); releaseUrl(enhancedUrl.value);
  window.clearTimeout(previewTimer); window.clearInterval(pollTimer);
  resizeObserver?.disconnect();
});
</script>

<template>
  <div class="h-full overflow-y-auto bg-slate-50 p-5 dark:bg-zinc-950">
    <div class="mx-auto grid max-w-[1500px] gap-4 xl:grid-cols-[270px_minmax(0,1fr)_310px]">
      <aside class="space-y-4">
        <section class="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="mb-3 flex items-center gap-2"><ImagePlus class="h-4 w-4 text-blue-600" /><h2 class="text-sm font-semibold">输入照片</h2></div>
          <div class="grid gap-2">
            <button @click="chooseSingle" class="rounded-xl border border-slate-200 px-3 py-2 text-left text-xs hover:border-blue-400 dark:border-zinc-700"><ImagePlus class="mr-2 inline h-3.5 w-3.5" />选择单张照片</button>
            <button @click="chooseMultiple" class="rounded-xl border border-slate-200 px-3 py-2 text-left text-xs hover:border-blue-400 dark:border-zinc-700"><Images class="mr-2 inline h-3.5 w-3.5" />选择多张照片</button>
            <button @click="chooseFolder" class="rounded-xl border border-slate-200 px-3 py-2 text-left text-xs hover:border-blue-400 dark:border-zinc-700"><FolderOpen class="mr-2 inline h-3.5 w-3.5" />选择照片文件夹</button>
          </div>
          <div class="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:bg-zinc-800 dark:text-zinc-400">
            <LoaderCircle v-if="sessionLoading" class="mr-1 inline h-3 w-3 animate-spin" />
            {{ files.length ? `当前共 ${files.length} 张` : "尚未选择照片" }}
          </div>
        </section>

        <section v-if="files.length" class="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="border-b border-slate-100 px-4 py-3 text-xs font-semibold dark:border-zinc-800">照片列表</div>
          <div class="max-h-80 overflow-y-auto p-2">
            <button v-for="file in files" :key="file.photo_id" @click="selectedId = file.photo_id"
              class="mb-1 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left text-xs"
              :class="selectedId === file.photo_id ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300' : 'hover:bg-slate-50 dark:hover:bg-zinc-800'">
              <Images class="h-3.5 w-3.5 shrink-0" /><span class="truncate">{{ file.name }}</span>
            </button>
          </div>
        </section>
      </aside>

      <main class="min-w-0">
        <section class="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="flex items-center justify-between border-b border-slate-100 px-4 py-3 dark:border-zinc-800">
            <div class="min-w-0"><div class="truncate text-sm font-semibold">{{ currentFile?.name || "去朦胧预览" }}</div><div class="mt-0.5 text-[11px] text-slate-400">{{ imageSize || "选择照片后生成参数化预览" }}</div></div>
            <div class="flex items-center gap-1 rounded-lg bg-slate-50 p-1 dark:bg-zinc-800">
              <button @click="zoomBy(-0.25)" :disabled="zoom <= 1" title="缩小" aria-label="缩小" class="rounded p-1.5 hover:bg-white disabled:cursor-not-allowed disabled:opacity-35 dark:hover:bg-zinc-700"><Minus class="h-3.5 w-3.5" /></button>
              <button @click="resetView" title="适合窗口" aria-label="适合窗口" class="min-w-[3.8rem] rounded px-1.5 py-1 text-[11px] font-medium tabular-nums hover:bg-white dark:hover:bg-zinc-700">{{ Math.round(zoom * 100) }}%</button>
              <button @click="resetView" title="适合窗口" aria-label="适合窗口" class="rounded p-1.5 hover:bg-white dark:hover:bg-zinc-700"><Maximize2 class="h-3.5 w-3.5" /></button>
              <button @click="zoomBy(0.25)" :disabled="zoom >= 4" title="放大" aria-label="放大" class="rounded p-1.5 hover:bg-white disabled:cursor-not-allowed disabled:opacity-35 dark:hover:bg-zinc-700"><Plus class="h-3.5 w-3.5" /></button>
            </div>
          </div>
          <div ref="previewViewport" class="relative flex min-h-[520px] items-center justify-center overflow-hidden bg-slate-100 dark:bg-black" :class="originalUrl ? (isScrubbing ? 'cursor-ew-resize' : (zoom > 1 ? 'cursor-zoom-out' : 'cursor-zoom-in')) : 'cursor-default'" @pointerdown="onPreviewPointerDown" @pointermove="onPreviewPointerMove" @pointerup="onPreviewPointerUp" @pointercancel="onPreviewPointerCancel" @click="onPreviewClick" @contextmenu="onPreviewContextMenu">
            <template v-if="originalUrl">
              <div class="absolute inset-0 flex items-center justify-center overflow-hidden">
                <div class="preview-stage relative shrink-0" :style="imageStageStyle">
                  <img :src="enhancedUrl || originalUrl" alt="去朦胧后" class="block h-full w-full object-contain" draggable="false" @load="onPreviewImageLoad" />
                </div>
              </div>
              <div v-if="enhancedUrl" class="absolute inset-0 overflow-hidden" :style="{ clipPath: `inset(0 ${100 - split}% 0 0)` }">
                <div class="absolute inset-0 flex items-center justify-center overflow-hidden">
                  <div class="preview-stage relative shrink-0" :style="imageStageStyle">
                    <img :src="originalUrl" alt="原图" class="block h-full w-full object-contain" draggable="false" @load="onPreviewImageLoad" />
                  </div>
                </div>
              </div>
              <div v-if="enhancedUrl" class="pointer-events-none absolute inset-y-0 z-10 w-px bg-white shadow" :style="{ left: `${split}%`, transform: 'translateX(-50%)' }"></div>
              <span class="pointer-events-none absolute left-3 top-3 z-20 rounded bg-black/55 px-2 py-1 text-[11px] text-white">原图</span>
              <span class="pointer-events-none absolute right-3 top-3 z-20 rounded bg-black/55 px-2 py-1 text-[11px] text-white">去朦胧后</span>
              <input v-if="enhancedUrl" v-model.number="split" type="range" min="0" max="100" class="compare-split absolute bottom-4 z-20" aria-label="前后对比分割线" @pointerdown.stop @click.stop />
            </template>
            <div v-if="revealMenu" class="reveal-context-menu absolute z-40 rounded-lg border border-slate-200 bg-white p-1 shadow-lg dark:border-zinc-700 dark:bg-zinc-900" :style="{ left: `${revealMenu.left}px`, top: `${revealMenu.top}px` }" @click.stop @contextmenu.prevent.stop>
              <button type="button" class="flex w-full items-center gap-2 whitespace-nowrap rounded-md px-3 py-2 text-left text-xs hover:bg-slate-100 disabled:cursor-wait disabled:opacity-60 dark:hover:bg-zinc-800" :disabled="revealBusy" @click="revealOriginal">
                <FolderOpen class="h-3.5 w-3.5 shrink-0" />{{ revealMenuLabel }}
              </button>
            </div>
            <div v-if="!originalUrl" class="text-center text-slate-400"><Images class="mx-auto mb-3 h-10 w-10" /><p class="text-sm">选择照片开始预览</p></div>
            <div v-if="previewLoading" class="absolute inset-0 flex items-center justify-center bg-white/45 backdrop-blur-[1px] dark:bg-black/35"><LoaderCircle class="h-7 w-7 animate-spin text-blue-600" /></div>
          </div>
          <div v-if="previewError" class="flex items-center gap-2 border-t border-rose-100 bg-rose-50 px-4 py-3 text-xs text-rose-600 dark:border-rose-950 dark:bg-rose-950/30 dark:text-rose-300"><AlertCircle class="h-4 w-4" />{{ previewError }}</div>
        </section>
      </main>

      <aside class="space-y-4">
        <section class="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="mb-4 flex items-center justify-between"><div class="flex items-center gap-2"><SlidersHorizontal class="h-4 w-4 text-blue-600" /><h2 class="text-sm font-semibold">去朦胧参数</h2></div><button @click="resetParams" title="重置参数" class="rounded p-1.5 hover:bg-slate-100 dark:hover:bg-zinc-800"><RotateCcw class="h-3.5 w-3.5" /></button></div>
          <label class="block text-xs"><span class="flex justify-between"><span>去朦胧强度</span><span class="font-mono text-blue-600">{{ Math.round(params.strength * 100) }}</span></span><input v-model.number="params.strength" class="mt-2 w-full accent-blue-600" type="range" min="0" max="1" step="0.01" /></label>
          <button @click="advancedOpen = !advancedOpen" class="mt-4 flex w-full items-center justify-between border-t border-slate-100 pt-3 text-xs font-medium dark:border-zinc-800">高级参数<ChevronDown class="h-3.5 w-3.5 transition" :class="advancedOpen ? 'rotate-180' : ''" /></button>
          <div v-if="advancedOpen" class="mt-3 space-y-3">
            <label v-for="item in [
              ['naturalness','自然度'],['fog_retention','雾气保留'],['local_contrast','局部对比度'],
              ['color_protection','色彩保护'],['highlight_protection','高光保护'],['shadow_protection','暗部保护']
            ]" :key="item[0]" class="block text-[11px]">
              <span class="flex justify-between"><span>{{ item[1] }}</span><span class="font-mono text-slate-400">{{ Math.round(params[item[0] as keyof typeof params] * 100) }}</span></span>
              <input v-model.number="params[item[0] as keyof typeof params]" class="mt-1 w-full accent-blue-600" type="range" min="0" max="1" step="0.01" />
            </label>
          </div>
        </section>

        <section class="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <h2 class="mb-3 text-sm font-semibold">导出</h2>
          <button @click="chooseOutput" class="w-full truncate rounded-xl border border-slate-200 px-3 py-2 text-left text-[11px] text-slate-500 hover:border-blue-400 dark:border-zinc-700" :title="outputDir"><FolderOpen class="mr-1.5 inline h-3.5 w-3.5" />{{ outputDir || "选择输出目录" }}</button>
          <div class="mt-3 space-y-1.5 rounded-xl bg-emerald-50 p-3 text-[11px] text-emerald-700 dark:bg-emerald-950/30 dark:text-emerald-300">
            <div><ShieldCheck class="mr-1 inline h-3.5 w-3.5" />原始照片始终保持不变</div><div>16-bit Linear / Demosaiced DNG</div><div>文件名增加 _dehaze 后缀</div>
          </div>
          <p class="mt-2 text-[10px] leading-4 text-slate-400">JPG、PNG 等普通图片会生成 RGB Linear DNG，不会被标记成相机传感器 RAW。</p>
          <button v-if="!isRunning" @click="startBatch" :disabled="!files.length" class="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-2.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40"><Play class="h-3.5 w-3.5" />{{ files.length > 1 ? `处理并导出 ${files.length} 张` : "处理并导出" }}</button>
          <button v-else @click="cancelBatch" class="mt-3 flex w-full items-center justify-center gap-2 rounded-xl bg-rose-600 px-4 py-2.5 text-xs font-semibold text-white"><Square class="h-3.5 w-3.5" />停止后续处理</button>
        </section>

        <section v-if="job" class="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
          <div class="mb-2 flex justify-between text-xs"><span>批量进度</span><span>{{ Math.round(job.progress * 100) }}%</span></div>
          <div class="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-zinc-800"><div class="h-full bg-blue-600 transition-all" :style="{ width: `${job.progress * 100}%` }"></div></div>
          <div class="mt-3 grid grid-cols-3 gap-2 text-center text-[11px]"><div class="rounded-lg bg-slate-50 p-2 dark:bg-zinc-800">{{ job.total }}<br><span class="text-slate-400">总数</span></div><div class="rounded-lg bg-emerald-50 p-2 text-emerald-700 dark:bg-emerald-950/30">{{ job.success }}<br><span>完成</span></div><div class="rounded-lg bg-rose-50 p-2 text-rose-600 dark:bg-rose-950/30">{{ job.failed }}<br><span>失败</span></div></div>
          <div v-if="job.current_file" class="mt-3 truncate text-[11px] text-slate-500"><LoaderCircle class="mr-1 inline h-3 w-3 animate-spin" />{{ job.current_file }}</div>
          <div v-if="failedFiles.length" class="mt-3 max-h-24 overflow-y-auto text-[10px] text-rose-600"><div v-for="file in failedFiles" :key="file.photo_id" :title="file.error">{{ file.name }}：{{ file.error }}</div></div>
        </section>
        <div v-if="actionMessage" class="flex items-start gap-2 px-1 text-[11px] text-slate-500"><CheckCircle2 class="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-500" />{{ actionMessage }}</div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.preview-stage {
  transform-origin: center center;
  will-change: transform;
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
