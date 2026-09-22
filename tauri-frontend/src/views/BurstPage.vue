<script setup lang="ts">
import { ref, computed, nextTick, watch, onMounted, onBeforeUnmount } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useSse } from "../composables/useSse";
import { BASE_URL, isServerReady } from "../stores/api";
import {
  FolderOpen, Play, Square, SlidersHorizontal, CheckCircle2,
  AlertCircle, Cpu, Trash2, FileCheck2, ChevronDown,
  Images, ScanSearch, Layers3, ShieldCheck, Clock3,
  Focus, ZoomIn, Trophy, BadgeCheck, FolderInput, LoaderCircle,
} from "lucide-vue-next";

type FilterPreset = "conservative" | "balanced" | "aggressive" | "custom";
type ReviewMode = "auto" | "review";
type WeightMode = "adaptive" | "custom";
type AllBlurryAction = "keep" | "review" | "reject";

interface BurstPreviewShot {
  photo_id: string;
  name: string;
  companion_count: number;
  kept: boolean;
  failed: boolean;
  rank: number | null;
  sharpness: number;
  exposure: number;
  aesthetic: number;
  score: number;
  sharpest: boolean;
  aesthetic_best: boolean;
  category?: "keep" | "review" | "defect";
  reject_reasons?: string[];
  absolute_sharpness?: number;
  face_count?: number;
  closed_face_count?: number;
}

interface BurstPreviewGroup {
  index: number;
  shot_count: number;
  confidence_margin: number;
  needs_review: boolean;
  weights: {
    aesthetic: number;
    sharpness: number;
    exposure: number;
  };
  weight_reason: string;
  sharpness_spread: number;
  exposure_spread: number;
  shots: BurstPreviewShot[];
}

const maxCpus = ref(
  typeof window !== "undefined" && window.navigator.hardwareConcurrency
    ? window.navigator.hardwareConcurrency : 8,
);
const defaultWorkers = Math.max(1, Math.round(maxCpus.value * 0.8));
const inputDir = ref("");
const gapSeconds = ref(1.5);
const maxHammingDistance = ref(12);
const keepCount = ref(1);
const maxWorkers = ref(defaultWorkers);
const useGpu = ref(true);
const gpuDetected = ref(false);
const gpuDeviceName = ref("");
const reviewSubdir = ref("审查_连拍淘汰");
const defectSubdir = ref("审查_明显废片");
const allBlurryAction = ref<AllBlurryAction>("keep");
const weightMode = ref<WeightMode>("adaptive");
const customWeights = ref({ sharpness: 0.9, aesthetic: 0.05, exposure: 0.05 });
const eyeDetection = ref(false);
const eyeModelReady = ref(false);
const preset = ref<FilterPreset>("balanced");
const reviewMode = ref<ReviewMode>("auto");
const logContainer = ref<HTMLElement | null>(null);
const settingsScroll = ref<HTMLElement | null>(null);
const isStickyActionPinned = ref(false);
const keepSliderPosition = ref(0);
const selectedGroupIndex = ref(0);
const selectedPhotoId = ref<string | null>(null);
const decisionPendingId = ref<string | null>(null);
const decisionFeedback = ref("");
const zoomViewport = ref<HTMLElement | null>(null);
const zoomImage = ref<HTMLImageElement | null>(null);
const reviewZoom = ref(1);
const reviewPan = ref({ x: 0, y: 0 });
let dragStart: { pointerX: number; pointerY: number; panX: number; panY: number } | null = null;
const presetOptions: Array<{
  value: Exclude<FilterPreset, "custom">;
  label: string;
  hint: string;
}> = [
  { value: "conservative", label: "保守", hint: "少整理" },
  { value: "balanced", label: "均衡", hint: "推荐" },
  { value: "aggressive", label: "激进", hint: "多整理" },
];
const keepCountTicks = [1, 2, 3, 5, 10, 20];

const {
  messages, lastMessage, progressPct, isDone, isRunning,
  error, resultData, start, cancel,
} = useSse("/api/burst/run");

const folderName = computed(() => {
  const clean = cleanPath(inputDir.value).replace(/[\\/]+$/, "");
  if (!clean) return "尚未选择照片目录";
  return clean.split(/[\\/]/).pop() || clean;
});

const isWorkersExceeded = computed(() => {
  const workers = Number(maxWorkers.value);
  return Number.isNaN(workers) || workers > maxCpus.value || workers < 1;
});

const processStage = computed(() => {
  if (isDone.value) return 4;
  if (!isRunning.value) return 0;
  const message = lastMessage.value;
  if (message.includes("处理连拍组")) return 3;
  if (message.includes("哈希") || message.includes("分析连拍组")) return 2;
  return 1;
});

const progressLabel = computed(() => {
  if (error.value) return "处理遇到问题";
  if (isDone.value) return "筛选已完成";
  if (isRunning.value) return lastMessage.value || "正在扫描照片目录…";
  return inputDir.value ? "目录已就绪，可以开始筛选" : "选择一个照片目录开始";
});

const previewGroups = computed<BurstPreviewGroup[]>(
  () => (resultData.value?.groups as BurstPreviewGroup[] | undefined) ?? [],
);

const selectedGroup = computed<BurstPreviewGroup | null>(
  () => previewGroups.value[selectedGroupIndex.value] ?? null,
);

const selectedPhoto = computed<BurstPreviewShot | null>(() => {
  const group = selectedGroup.value;
  if (!group) return null;
  return group.shots.find((shot) => shot.photo_id === selectedPhotoId.value)
    ?? group.shots.find((shot) => shot.kept)
    ?? group.shots[0]
    ?? null;
});

const reviewGroupCount = computed(
  () => previewGroups.value.filter((group) => group.needs_review).length,
);

function cleanPath(value: string): string {
  if (!value) return "";
  return value.trim().replace(/^["']|["']$/g, "").trim();
}

function applyPreset(value: Exclude<FilterPreset, "custom">) {
  preset.value = value;
  maxHammingDistance.value = {
    conservative: 8, balanced: 12, aggressive: 18,
  }[value];
}

function syncPresetFromHamming() {
  const value = Number(maxHammingDistance.value);
  if (value === 8) preset.value = "conservative";
  else if (value === 12) preset.value = "balanced";
  else if (value === 18) preset.value = "aggressive";
  else preset.value = "custom";
}

// 使用非线性刻度：低数值占据更多滑动距离，越接近 20 刻度越密。
function sliderPositionToKeepCount(position: number): number {
  const normalized = Math.min(1, Math.max(0, position / 1000));
  return Math.round(1 + 19 * normalized ** 2.2);
}

function keepCountToSliderPosition(count: number): number {
  const normalized = Math.min(1, Math.max(0, (count - 1) / 19));
  return Math.round(normalized ** (1 / 2.2) * 1000);
}

function updateKeepCount(event: Event) {
  const position = Number((event.target as HTMLInputElement).value);
  keepSliderPosition.value = position;
  keepCount.value = sliderPositionToKeepCount(position);
}

function setKeepCount(count: number) {
  keepCount.value = count;
  keepSliderPosition.value = keepCountToSliderPosition(count);
}

function snapKeepSliderPosition() {
  setKeepCount(keepCount.value);
}

function stepKeepCount(delta: number) {
  setKeepCount(Math.min(20, Math.max(1, keepCount.value + delta)));
}

function selectPreviewGroup(index: number) {
  selectedGroupIndex.value = index;
  const group = previewGroups.value[index];
  selectedPhotoId.value = group?.shots.find((shot) => shot.kept)?.photo_id
    ?? group?.shots[0]?.photo_id
    ?? null;
}

async function focusSelectedThumbnail() {
  await nextTick();
  if (!selectedPhotoId.value) return;
  document.querySelector<HTMLElement>(`[data-photo-id="${selectedPhotoId.value}"]`)
    ?.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" });
}

function selectAdjacentPhoto(delta: number) {
  const shots = selectedGroup.value?.shots ?? [];
  if (!shots.length) return;
  const currentIndex = Math.max(0, shots.findIndex((shot) => shot.photo_id === selectedPhotoId.value));
  const nextIndex = Math.min(shots.length - 1, Math.max(0, currentIndex + delta));
  selectedPhotoId.value = shots[nextIndex].photo_id;
  void focusSelectedThumbnail();
}

function handleReviewKeydown(event: KeyboardEvent) {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  const target = event.target as HTMLElement | null;
  if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
  if (!selectedGroup.value?.shots.length) return;
  event.preventDefault();
  selectAdjacentPhoto(event.key === "ArrowLeft" ? -1 : 1);
}

async function setPhotoDecision(shot: BurstPreviewShot, kept: boolean) {
  if (shot.kept === kept || decisionPendingId.value || !BASE_URL.value) return;
  const session = resultData.value?.preview_session;
  if (!session) return;

  decisionPendingId.value = shot.photo_id;
  decisionFeedback.value = "";
  try {
    const response = await fetch(`${BASE_URL.value}/api/burst/decision/${session}/${shot.photo_id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kept }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法移动照片");

    const wasKept = shot.kept;
    shot.kept = Boolean(data.kept);
    shot.category = data.category ?? (shot.kept ? "keep" : shot.category ?? "review");
    const changedFiles = Number(data.moved_files ?? shot.companion_count ?? 1);
    if (resultData.value && wasKept !== shot.kept) {
      const delta = shot.kept ? -changedFiles : changedFiles;
      resultData.value.moved = Math.max(0, Number(resultData.value.moved || 0) + delta);
    }
    decisionFeedback.value = `${shot.name}：${data.message}`;
    messages.value.push(`✓ ${decisionFeedback.value}`);
  } catch (err) {
    decisionFeedback.value = err instanceof Error ? err.message : String(err);
  } finally {
    decisionPendingId.value = null;
  }
}

function previewUrl(photoId: string, kind: "full" | "focus" | "review") {
  const session = resultData.value?.preview_session;
  if (!session || !BASE_URL.value) return "";
  return `${BASE_URL.value}/api/burst/preview/${session}/${photoId}?kind=${kind}`;
}

function clampReviewPan(x: number, y: number) {
  const viewport = zoomViewport.value;
  const image = zoomImage.value;
  if (!viewport || !image?.naturalWidth || !image.naturalHeight) return { x: 0, y: 0 };
  const maxX = Math.max(0, (image.naturalWidth * reviewZoom.value - viewport.clientWidth) / 2);
  const maxY = Math.max(0, (image.naturalHeight * reviewZoom.value - viewport.clientHeight) / 2);
  return {
    x: Math.min(maxX, Math.max(-maxX, x)),
    y: Math.min(maxY, Math.max(-maxY, y)),
  };
}

function resetReviewZoom() {
  reviewZoom.value = 1;
  reviewPan.value = { x: 0, y: 0 };
}

async function setReviewZoom(nextZoom: number) {
  reviewZoom.value = Math.min(4, Math.max(1, nextZoom));
  await nextTick();
  reviewPan.value = clampReviewPan(reviewPan.value.x, reviewPan.value.y);
}

function handleZoomWheel(event: WheelEvent) {
  event.preventDefault();
  void setReviewZoom(reviewZoom.value + (event.deltaY < 0 ? 0.25 : -0.25));
}

function startReviewDrag(event: PointerEvent) {
  if (event.button !== 0) return;
  dragStart = {
    pointerX: event.clientX,
    pointerY: event.clientY,
    panX: reviewPan.value.x,
    panY: reviewPan.value.y,
  };
  (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
}

function moveReviewDrag(event: PointerEvent) {
  if (!dragStart) return;
  reviewPan.value = clampReviewPan(
    dragStart.panX + event.clientX - dragStart.pointerX,
    dragStart.panY + event.clientY - dragStart.pointerY,
  );
}

function stopReviewDrag(event: PointerEvent) {
  dragStart = null;
  const target = event.currentTarget as HTMLElement;
  if (target.hasPointerCapture(event.pointerId)) target.releasePointerCapture(event.pointerId);
}

function scorePercent(value: number) {
  return `${Math.max(0, Math.min(100, value * 100))}%`;
}

function weightPercent(value: number | undefined) {
  return `${Math.round((value ?? 0) * 100)}%`;
}

function setCustomWeight(key: "sharpness" | "aesthetic" | "exposure", percent: number) {
  const next = Math.min(100, Math.max(0, Number(percent))) / 100;
  const keys = (["sharpness", "aesthetic", "exposure"] as const).filter((item) => item !== key);
  const remaining = 1 - next;
  const otherTotal = keys.reduce((total, item) => total + customWeights.value[item], 0);
  customWeights.value[key] = next;
  for (const item of keys) {
    customWeights.value[item] = otherTotal > 0
      ? remaining * customWeights.value[item] / otherTotal
      : remaining / keys.length;
  }
}

function applyClarityPreset() {
  customWeights.value = { sharpness: 0.9, aesthetic: 0.05, exposure: 0.05 };
  weightMode.value = "custom";
}

async function checkGpuAvailability() {
  if (!BASE_URL.value) return;
  try {
    const response = await fetch(`${BASE_URL.value}/api/models/status`);
    if (!response.ok) return;
    const data = await response.json();
    gpuDetected.value = Boolean(data.gpu_available);
    gpuDeviceName.value = data.gpu_name || "CPU 多核心并行计算";
    useGpu.value = gpuDetected.value;
    eyeModelReady.value = Boolean(data.face_landmarker_ready);
    if (!eyeModelReady.value) eyeDetection.value = false;
  } catch (err) {
    console.error("检测 GPU 状态失败:", err);
  }
}

async function selectDirectory() {
  try {
    const selected = await open({
      directory: true, multiple: false, title: "选择待筛选照片所在目录",
    });
    if (selected && typeof selected === "string") inputDir.value = cleanPath(selected);
  } catch (err) {
    console.error("选择目录失败:", err);
  }
}

async function handleStart() {
  const directory = cleanPath(inputDir.value);
  inputDir.value = directory;
  if (!directory) {
    alert("请先选择照片目录！");
    return;
  }
  if (isWorkersExceeded.value) {
    alert(`并发线程数无效或超过系统上限 (${maxCpus.value})！`);
    return;
  }
  await start({
    input_dir: directory,
    gap_seconds: Number(gapSeconds.value),
    max_hamming_distance: Number(maxHammingDistance.value),
    review_subdir: reviewSubdir.value,
    defect_subdir: defectSubdir.value,
    keep_count: Number(keepCount.value),
    max_workers: Number(maxWorkers.value),
    use_gpu: Boolean(useGpu.value),
    include_previews: reviewMode.value === "review",
    weight_mode: weightMode.value,
    custom_weights: weightMode.value === "custom" ? customWeights.value : null,
    all_blurry_action: allBlurryAction.value,
    eye_detection: eyeDetection.value && eyeModelReady.value,
  });
}

function clearLogs() {
  messages.value = [];
}

function updateStickyActionState() {
  const scroller = settingsScroll.value;
  isStickyActionPinned.value = Boolean(scroller && scroller.scrollTop > 1);
}

onMounted(() => {
  checkGpuAvailability();
  window.addEventListener("keydown", handleReviewKeydown);
  window.addEventListener("imprint:model-status-changed", checkGpuAvailability);
});
onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleReviewKeydown);
  window.removeEventListener("imprint:model-status-changed", checkGpuAvailability);
});
watch(() => isServerReady.value, (ready) => ready && checkGpuAvailability());
watch(() => messages.value.length, async () => {
  await nextTick();
  if (logContainer.value) logContainer.value.scrollTop = logContainer.value.scrollHeight;
});
watch(() => resultData.value, (result) => {
  if (result?.groups?.length) selectPreviewGroup(0);
});
watch(selectedPhotoId, resetReviewZoom);
</script>

<template>
  <div class="workspace-readable h-full min-h-0 bg-[#f5f7fa] dark:bg-zinc-950">
    <div class="workspace-wide-frame wide-workspace-grid grid h-full min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_clamp(360px,24vw,430px)]">
      <aside class="wide-workspace-sidebar hidden min-h-0 flex-col gap-5 border-r border-slate-200 bg-[#f8fafc] px-5 py-6 dark:border-zinc-800 dark:bg-zinc-950">
        <header>
          <div class="mb-1 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-blue-600 dark:text-blue-400">
            <ScanSearch class="h-3.5 w-3.5" />
            Batch culling workspace
          </div>
          <h2 class="text-2xl font-bold tracking-tight text-slate-950 dark:text-white">连拍优选</h2>
          <p class="mt-1 text-xs leading-5 text-slate-500 dark:text-zinc-400">自动识别连拍序列，挑出最佳画面</p>
        </header>

        <button type="button" @click="selectDirectory"
          class="group flex w-full items-start gap-3 rounded-xl border border-slate-200 bg-white p-3.5 text-left shadow-[0_1px_2px_rgba(15,23,42,0.03)] transition hover:border-blue-300 hover:shadow-[0_8px_24px_rgba(37,99,235,0.08)] dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-blue-700">
          <span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-400">
            <FolderOpen class="h-4 w-4" />
          </span>
          <span class="min-w-0 flex-1">
            <span class="block truncate text-xs font-semibold text-slate-900 dark:text-zinc-100">{{ folderName }}</span>
            <span v-if="inputDir" class="mt-1 block break-all text-[10px] leading-4 text-slate-400 dark:text-zinc-500">{{ inputDir }}</span>
            <span v-else class="mt-1 block text-[10px] leading-4 text-slate-400 dark:text-zinc-500">RAW、JPG、HEIC、HIF 等常见格式</span>
            <span class="mt-2 block text-[10px] font-medium text-blue-600 dark:text-blue-400">{{ inputDir ? "更换目录" : "浏览选择" }}</span>
          </span>
        </button>

        <section class="rounded-xl border border-slate-200 bg-white p-3.5 dark:border-zinc-800 dark:bg-zinc-900">
          <div class="mb-3 flex items-center gap-2 text-xs font-semibold text-slate-800 dark:text-zinc-200">
            <ShieldCheck class="h-4 w-4 text-blue-600 dark:text-blue-400" />任务概览
          </div>
          <div class="space-y-2.5 text-[11px] text-slate-500 dark:text-zinc-400">
            <div class="flex items-center gap-2"><CheckCircle2 class="h-3.5 w-3.5 shrink-0 text-emerald-500" />本地处理，不上传照片</div>
            <div class="flex items-center gap-2"><ShieldCheck class="h-3.5 w-3.5 shrink-0 text-blue-500" />原图不修改、不删除</div>
          </div>
        </section>
      </aside>

      <section class="flex min-w-0 min-h-0 flex-col border-r border-slate-200 bg-[#f8fafc] dark:border-zinc-800 dark:bg-zinc-950">
        <div class="workspace-main-scroll flex-1 px-7 py-6">
          <div class="mx-auto flex min-h-full w-full max-w-6xl flex-col gap-5">
            <p class="wide-workspace-description hidden text-sm text-slate-500 dark:text-zinc-400">
              自动识别连拍序列，从清晰度、曝光与审美表现中挑出最佳画面
            </p>
            <header class="wide-workspace-header">
              <div class="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-blue-600 dark:text-blue-400">
                <ScanSearch class="h-3.5 w-3.5" />
                Batch culling workspace
              </div>
              <h2 class="text-[28px] font-bold tracking-tight text-slate-950 dark:text-white">连拍优选</h2>
              <p class="mt-1 text-sm text-slate-500 dark:text-zinc-400">
                自动识别连拍序列，从清晰度、曝光与审美表现中挑出最佳画面
              </p>
            </header>

            <button type="button" @click="selectDirectory"
              class="wide-workspace-directory group flex w-full items-center gap-4 rounded-xl border border-slate-200 bg-white px-5 py-4 text-left shadow-[0_1px_2px_rgba(15,23,42,0.03)] transition hover:border-blue-300 hover:shadow-[0_8px_24px_rgba(37,99,235,0.08)] dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-blue-700">
              <span class="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-400">
                <FolderOpen class="h-5 w-5" />
              </span>
              <span class="min-w-0 flex-1">
                <span class="block truncate text-[15px] font-semibold text-slate-900 dark:text-zinc-100">{{ folderName }}</span>
                <span v-if="inputDir" class="mt-0.5 block truncate text-xs text-slate-400 dark:text-zinc-500">{{ inputDir }}</span>
                <span v-else class="mt-0.5 block text-xs text-slate-400 dark:text-zinc-500">RAW、JPG、HEIC、HIF 等常见摄影格式</span>
              </span>
              <span class="shrink-0 text-xs font-medium text-blue-600 dark:text-blue-400">
                {{ inputDir ? "更换目录" : "浏览选择" }}
              </span>
            </button>

            <div class="wide-workspace-flow order-3 flex h-[260px] shrink-0 flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-900 shadow-[0_12px_36px_rgba(15,23,42,0.12)] dark:border-zinc-800">
              <div class="flex shrink-0 items-center justify-between border-b border-white/10 px-5 py-3">
                <div class="flex items-center gap-2 text-sm font-medium text-white">
                  <Images class="h-4 w-4 text-blue-400" /> 批处理流程
                </div>
                <span class="text-[11px] text-slate-400">本地处理 · 原图内容不修改</span>
              </div>
              <div class="wide-workspace-flow-grid grid min-h-0 flex-1 grid-cols-2 gap-px bg-white/10 md:grid-cols-4">
                <div v-for="(stage, index) in ['扫描文件', '识别连拍', '质量评估', '整理结果']"
                  :key="stage" class="relative bg-slate-900 px-5 py-6">
                  <div class="mb-3 flex h-8 w-8 items-center justify-center rounded-full border text-xs font-semibold"
                    :class="processStage > index
                      ? 'border-blue-500 bg-blue-500 text-white'
                      : processStage === index && isRunning
                        ? 'border-blue-400 bg-blue-500/15 text-blue-300 ring-4 ring-blue-500/10'
                        : 'border-slate-700 bg-slate-800 text-slate-400'">
                    <CheckCircle2 v-if="processStage > index" class="h-4 w-4" />
                    <span v-else>0{{ index + 1 }}</span>
                  </div>
                  <div class="text-sm font-medium text-slate-100">{{ stage }}</div>
                  <div class="mt-1 text-[11px] leading-5 text-slate-500">
                    {{ ['读取格式与 EXIF', '按时间和画面归组', '清晰度 · 曝光 · 审美', '保留优选并移动淘汰项'][index] }}
                  </div>
                  <div v-if="index < 3" class="wide-workspace-connector absolute right-0 top-10 hidden h-px w-5 translate-x-1/2 bg-slate-700 md:block"></div>
                </div>
              </div>
              <div class="shrink-0 border-t border-white/10 bg-slate-950/50 px-5 py-3">
                <div class="flex items-center gap-3">
                  <span class="h-2 w-2 shrink-0 rounded-full"
                    :class="error ? 'bg-rose-500' : isRunning ? 'animate-pulse bg-blue-400' : isDone ? 'bg-emerald-400' : 'bg-slate-600'"></span>
                  <span class="min-w-0 flex-1 truncate text-xs text-slate-300">{{ progressLabel }}</span>
                  <span v-if="progressPct !== null" class="text-[11px] tabular-nums text-blue-300">{{ Math.round(progressPct * 100) }}%</span>
                </div>
                <div v-if="isRunning" class="mt-3 h-1 overflow-hidden rounded-full bg-slate-800">
                  <div class="h-full rounded-full bg-blue-500 transition-all duration-500"
                    :class="progressPct === null ? 'w-1/3 animate-pulse' : ''"
                    :style="progressPct !== null ? { width: `${Math.max(3, progressPct * 100)}%` } : undefined"></div>
                </div>
              </div>
            </div>

            <div v-if="isDone && resultData" class="order-2 rounded-xl border border-blue-200 bg-blue-50/70 p-5 dark:border-blue-900 dark:bg-blue-950/25">
              <div class="mb-4 flex items-center gap-2 font-semibold text-blue-950 dark:text-blue-100">
                <CheckCircle2 class="h-5 w-5 text-blue-600 dark:text-blue-400" /> 本次筛选已完成
              </div>
              <div class="grid grid-cols-2 divide-x divide-blue-200 md:grid-cols-4 dark:divide-blue-900">
                <div class="px-4 first:pl-0"><div class="text-2xl font-bold tabular-nums text-slate-950 dark:text-white">{{ resultData.total }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">扫描文件</div></div>
                <div class="px-4"><div class="text-2xl font-bold tabular-nums text-slate-950 dark:text-white">{{ resultData.burst_groups }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">连拍组</div></div>
                <div class="px-4"><div class="text-2xl font-bold tabular-nums text-blue-700 dark:text-blue-300">{{ resultData.moved }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">共移出原目录</div><div v-if="resultData.defect_moved" class="mt-1 text-[10px] text-rose-500">明显废片 {{ resultData.defect_moved }}</div></div>
                <div class="px-4"><div class="text-2xl font-bold tabular-nums text-slate-950 dark:text-white">{{ resultData.skipped_single }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">单张跳过</div></div>
              </div>
              <div v-if="resultData.review_dir" class="mt-4 flex items-center gap-2 border-t border-blue-200 pt-3 text-xs text-blue-800 dark:border-blue-900 dark:text-blue-300">
                <FileCheck2 class="h-4 w-4 shrink-0" /><span class="truncate">审查目录：{{ resultData.review_dir }}</span>
              </div>
              <div v-if="resultData.defect_dir" class="mt-2 flex items-center gap-2 text-xs text-rose-700 dark:text-rose-300">
                <Trash2 class="h-4 w-4 shrink-0" /><span class="truncate">明显废片目录：{{ resultData.defect_dir }}</span>
              </div>
            </div>

            <div v-if="previewGroups.length" class="order-1 overflow-hidden rounded-xl border border-slate-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
              <div class="flex flex-wrap items-center gap-3 border-b border-slate-200 px-5 py-4 dark:border-zinc-800">
                <div class="flex items-center gap-2 font-semibold text-slate-900 dark:text-zinc-100">
                  <Images class="h-5 w-5 text-blue-600 dark:text-blue-400" />连拍组复核
                </div>
                <span class="text-xs text-slate-400">优先展示 {{ reviewGroupCount }} 个低置信度组</span>
                <span class="ml-auto text-xs text-slate-400">本次显示 {{ previewGroups.length }} / {{ resultData.burst_groups }} 组</span>
              </div>

              <div class="border-b border-slate-200 bg-slate-50 px-4 py-3 dark:border-zinc-800 dark:bg-zinc-950/50">
                <div class="flex gap-2 overflow-x-auto pb-1">
                  <button
                    v-for="(group, index) in previewGroups"
                    :key="group.index"
                    type="button"
                    @click="selectPreviewGroup(index)"
                    class="flex shrink-0 items-center gap-2 rounded-md border px-3 py-2 text-xs font-medium transition"
                    :class="selectedGroupIndex === index
                      ? 'border-blue-500 bg-blue-600 text-white'
                      : group.needs_review
                        ? 'border-amber-200 bg-amber-50 text-amber-800 hover:border-amber-300 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-300'
                        : 'border-slate-200 bg-white text-slate-600 hover:border-blue-300 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300'"
                  >
                    <span>第 {{ group.index }} 组</span>
                    <span class="opacity-70">{{ group.shot_count }} 张</span>
                    <span v-if="group.needs_review" class="h-1.5 w-1.5 rounded-full" :class="selectedGroupIndex === index ? 'bg-amber-300' : 'bg-amber-500'"></span>
                  </button>
                </div>
              </div>

              <div v-if="selectedGroup" class="p-5">
                <div class="mb-4 flex flex-wrap items-center gap-3">
                  <div>
                    <div class="text-sm font-semibold text-slate-900 dark:text-zinc-100">第 {{ selectedGroup.index }} 组 · {{ selectedGroup.shot_count }} 次快门</div>
                    <div class="mt-1 text-xs text-slate-400">点击缩略图检查完整画面与局部清晰度</div>
                  </div>
                  <span class="ml-auto rounded-md border border-slate-200 px-2 py-1 text-[11px] text-slate-400 dark:border-zinc-700">← → 切换照片</span>
                  <span class="rounded-md px-2 py-1 text-xs font-medium"
                    :class="selectedGroup.needs_review ? 'bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300' : 'bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300'">
                    {{ selectedGroup.needs_review ? '建议复核' : '选择明确' }}
                  </span>
                </div>

                <div class="mb-4 flex flex-wrap items-center gap-2 rounded-lg border border-blue-100 bg-blue-50/70 px-3.5 py-3 text-xs text-blue-900 dark:border-blue-900/70 dark:bg-blue-950/25 dark:text-blue-200">
                  <span class="font-semibold">实际权重</span>
                  <span class="text-blue-700/80 dark:text-blue-300/80">{{ selectedGroup.weight_reason }}</span>
                  <div class="ml-auto flex flex-wrap gap-1.5 tabular-nums">
                    <span class="rounded bg-white/80 px-2 py-1 dark:bg-zinc-900/60">审美 {{ weightPercent(selectedGroup.weights?.aesthetic) }}</span>
                    <span class="rounded bg-white/80 px-2 py-1 dark:bg-zinc-900/60">清晰 {{ weightPercent(selectedGroup.weights?.sharpness) }}</span>
                    <span class="rounded bg-white/80 px-2 py-1 dark:bg-zinc-900/60">曝光 {{ weightPercent(selectedGroup.weights?.exposure) }}</span>
                  </div>
                </div>

                <div class="flex gap-3 overflow-x-auto pb-3">
                  <div
                    v-for="shot in selectedGroup.shots"
                    :key="shot.photo_id"
                    :data-photo-id="shot.photo_id"
                    role="button"
                    tabindex="0"
                    @click="selectedPhotoId = shot.photo_id"
                    @keydown.enter.prevent="selectedPhotoId = shot.photo_id"
                    class="group/photo relative w-40 shrink-0 overflow-hidden rounded-lg border-2 bg-slate-100 text-left transition dark:bg-zinc-800"
                    :class="selectedPhoto?.photo_id === shot.photo_id
                      ? 'border-blue-500 shadow-[0_6px_18px_rgba(37,99,235,0.16)]'
                      : shot.kept ? 'border-blue-200 hover:border-blue-400 dark:border-blue-900' : 'border-transparent opacity-70 hover:opacity-100'"
                  >
                    <div class="relative aspect-[3/2] overflow-hidden bg-slate-200 dark:bg-zinc-800">
                      <img :src="previewUrl(shot.photo_id, 'full')" :alt="shot.name" loading="lazy" class="h-full w-full object-cover" />
                      <button type="button" @click.stop="setPhotoDecision(shot, !shot.kept)" :disabled="decisionPendingId !== null"
                        :title="shot.kept ? '点击移入审查目录' : '点击恢复到原目录'"
                        class="absolute left-2 top-2 flex items-center gap-1 rounded px-1.5 py-1 text-[10px] font-semibold text-white transition hover:ring-2 hover:ring-white/70 disabled:opacity-60"
                        :class="shot.kept ? 'bg-blue-600' : 'bg-slate-950/70'">
                        <LoaderCircle v-if="decisionPendingId === shot.photo_id" class="h-3 w-3 animate-spin" />
                        <BadgeCheck v-else-if="shot.kept" class="h-3 w-3" />
                        {{ shot.kept ? '保留' : shot.category === 'defect' ? '废片' : '审查' }}
                      </button>
                      <span class="absolute bottom-2 right-2 rounded bg-slate-950/75 px-1.5 py-1 text-[10px] font-semibold tabular-nums text-white">{{ Math.round(shot.score * 100) }}</span>
                    </div>
                    <div class="truncate px-2.5 py-2 text-[11px] text-slate-600 dark:text-zinc-300">{{ shot.name }}</div>
                    <div class="grid grid-cols-2 border-t border-slate-200 dark:border-zinc-700">
                      <button type="button" @click.stop="setPhotoDecision(shot, true)"
                        :disabled="shot.kept || decisionPendingId !== null"
                        class="flex items-center justify-center gap-1 border-r border-slate-200 px-1 py-2 text-[10px] font-medium transition dark:border-zinc-700"
                        :class="shot.kept ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/40 dark:text-blue-300' : 'text-slate-500 hover:bg-blue-50 hover:text-blue-700 dark:text-zinc-400 dark:hover:bg-blue-950/30'">
                        <LoaderCircle v-if="decisionPendingId === shot.photo_id" class="h-3 w-3 animate-spin" />
                        <BadgeCheck v-else class="h-3 w-3" />保留
                      </button>
                      <button type="button" @click.stop="setPhotoDecision(shot, false)"
                        :disabled="!shot.kept || decisionPendingId !== null"
                        class="flex items-center justify-center gap-1 px-1 py-2 text-[10px] font-medium transition"
                        :class="!shot.kept ? 'bg-slate-200 text-slate-700 dark:bg-zinc-700 dark:text-zinc-200' : 'text-slate-500 hover:bg-slate-200 hover:text-slate-800 dark:text-zinc-400 dark:hover:bg-zinc-700'">
                        <LoaderCircle v-if="decisionPendingId === shot.photo_id" class="h-3 w-3 animate-spin" />
                        <FolderInput v-else class="h-3 w-3" />审查
                      </button>
                    </div>
                  </div>
                </div>

                <div v-if="selectedPhoto" class="mt-2 grid grid-cols-1 gap-4 border-t border-slate-200 pt-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(220px,0.65fr)] dark:border-zinc-800">
                  <div class="grid min-w-0 grid-cols-2 gap-3">
                    <div class="overflow-hidden rounded-lg bg-slate-950">
                      <div class="flex items-center gap-2 border-b border-white/10 px-3 py-2 text-xs text-slate-300"><Images class="h-3.5 w-3.5 text-blue-400" />完整画面</div>
                      <div class="flex aspect-[3/2] items-center justify-center"><img :src="previewUrl(selectedPhoto.photo_id, 'review')" :alt="selectedPhoto.name" class="max-h-full max-w-full object-contain" /></div>
                    </div>
                    <div class="overflow-hidden rounded-lg bg-slate-950">
                      <div class="flex items-center gap-2 border-b border-white/10 px-3 py-1.5 text-xs text-slate-300">
                        <ZoomIn class="h-3.5 w-3.5 text-blue-400" />1:1 细节检查
                        <div class="ml-auto flex items-center gap-1">
                          <button type="button" @click="setReviewZoom(reviewZoom - 0.25)" :disabled="reviewZoom <= 1" class="flex h-6 w-6 items-center justify-center rounded bg-white/5 text-sm hover:bg-white/10 disabled:opacity-30" aria-label="缩小">−</button>
                          <button type="button" @click="resetReviewZoom" class="min-w-12 rounded bg-white/5 px-1.5 py-1 text-[10px] tabular-nums hover:bg-white/10">{{ Math.round(reviewZoom * 100) }}%</button>
                          <button type="button" @click="setReviewZoom(reviewZoom + 0.25)" :disabled="reviewZoom >= 4" class="flex h-6 w-6 items-center justify-center rounded bg-white/5 text-sm hover:bg-white/10 disabled:opacity-30" aria-label="放大">+</button>
                        </div>
                      </div>
                      <div ref="zoomViewport"
                        class="relative aspect-[3/2] touch-none select-none overflow-hidden bg-slate-950 cursor-grab active:cursor-grabbing"
                        @wheel="handleZoomWheel"
                        @pointerdown="startReviewDrag"
                        @pointermove="moveReviewDrag"
                        @pointerup="stopReviewDrag"
                        @pointercancel="stopReviewDrag"
                        @dblclick="resetReviewZoom"
                      >
                        <img ref="zoomImage" :src="previewUrl(selectedPhoto.photo_id, 'review')" :alt="`${selectedPhoto.name} 1:1 细节`" draggable="false"
                          @load="reviewPan = clampReviewPan(reviewPan.x, reviewPan.y)"
                          class="pointer-events-none absolute max-w-none"
                          :style="{
                            left: `calc(50% + ${reviewPan.x}px)`,
                            top: `calc(50% + ${reviewPan.y}px)`,
                            transform: `translate(-50%, -50%) scale(${reviewZoom})`,
                            transformOrigin: 'center center',
                          }"
                        />
                        <div class="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 rounded bg-black/55 px-2 py-1 text-[10px] text-white/75">按住拖动 · 滚轮缩放 · 双击复位</div>
                      </div>
                    </div>
                  </div>

                  <div class="flex flex-col rounded-lg border border-slate-200 p-4 dark:border-zinc-700">
                    <div class="mb-3 flex items-start justify-between gap-3">
                      <div class="min-w-0"><div class="truncate text-sm font-semibold text-slate-900 dark:text-zinc-100">{{ selectedPhoto.name }}</div><div class="mt-1 text-xs text-slate-400">综合排名 #{{ selectedPhoto.rank ?? '—' }}</div></div>
                      <span class="shrink-0 rounded-md px-2 py-1 text-xs font-semibold" :class="selectedPhoto.kept ? 'bg-blue-50 text-blue-700 dark:bg-blue-950/50 dark:text-blue-300' : selectedPhoto.category === 'defect' ? 'bg-rose-50 text-rose-700 dark:bg-rose-950/40 dark:text-rose-300' : 'bg-slate-100 text-slate-600 dark:bg-zinc-800 dark:text-zinc-300'">{{ selectedPhoto.kept ? '已保留' : selectedPhoto.category === 'defect' ? '已移入明显废片' : '已移入审查' }}</span>
                    </div>
                    <div class="space-y-3">
                      <div v-for="metric in [
                        { label: '清晰度', value: selectedPhoto.sharpness },
                        { label: '审美', value: selectedPhoto.aesthetic },
                        { label: '曝光', value: selectedPhoto.exposure },
                      ]" :key="metric.label">
                        <div class="mb-1 flex justify-between text-xs text-slate-500 dark:text-zinc-400"><span>{{ metric.label }}</span><span class="tabular-nums">{{ Math.round(metric.value * 100) }}</span></div>
                        <div class="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-zinc-800"><div class="h-full rounded-full bg-blue-500" :style="{ width: scorePercent(metric.value) }"></div></div>
                      </div>
                    </div>
                    <div class="mt-4 flex flex-wrap gap-2 border-t border-slate-200 pt-3 dark:border-zinc-700">
                      <span v-if="selectedPhoto.sharpest" class="flex items-center gap-1 rounded bg-blue-50 px-2 py-1 text-[10px] font-medium text-blue-700 dark:bg-blue-950/40 dark:text-blue-300"><Focus class="h-3 w-3" />本组最清晰</span>
                      <span v-if="selectedPhoto.aesthetic_best" class="flex items-center gap-1 rounded bg-violet-50 px-2 py-1 text-[10px] font-medium text-violet-700 dark:bg-violet-950/40 dark:text-violet-300"><Trophy class="h-3 w-3" />审美最高</span>
                      <span v-if="selectedPhoto.reject_reasons?.includes('severe_blur')" class="rounded bg-rose-50 px-2 py-1 text-[10px] text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">严重虚焦</span>
                      <span v-if="selectedPhoto.reject_reasons?.includes('closed_eyes')" class="rounded bg-rose-50 px-2 py-1 text-[10px] text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">检测到闭眼</span>
                      <span v-if="selectedPhoto.companion_count > 1" class="rounded bg-slate-100 px-2 py-1 text-[10px] text-slate-600 dark:bg-zinc-800 dark:text-zinc-300">含 {{ selectedPhoto.companion_count }} 个伴生文件</span>
                    </div>
                    <div class="mt-auto grid grid-cols-2 gap-2 pt-4">
                      <button type="button" @click="setPhotoDecision(selectedPhoto, true)"
                        :disabled="selectedPhoto.kept || decisionPendingId !== null"
                        class="flex items-center justify-center gap-1.5 rounded-lg border border-blue-200 px-3 py-2.5 text-xs font-semibold text-blue-700 transition hover:bg-blue-50 disabled:cursor-default disabled:bg-blue-50 disabled:opacity-55 dark:border-blue-900 dark:text-blue-300 dark:hover:bg-blue-950/40 dark:disabled:bg-blue-950/30">
                        <BadgeCheck class="h-3.5 w-3.5" />保留在原目录
                      </button>
                      <button type="button" @click="setPhotoDecision(selectedPhoto, false)"
                        :disabled="!selectedPhoto.kept || decisionPendingId !== null"
                        class="flex items-center justify-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2.5 text-xs font-semibold text-slate-600 transition hover:bg-slate-100 disabled:cursor-default disabled:bg-slate-100 disabled:opacity-55 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:disabled:bg-zinc-800">
                        <LoaderCircle v-if="decisionPendingId === selectedPhoto.photo_id" class="h-3.5 w-3.5 animate-spin" />
                        <FolderInput v-else class="h-3.5 w-3.5" />移入审查目录
                      </button>
                    </div>
                  </div>
                </div>
                <div v-if="decisionFeedback" class="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500 dark:bg-zinc-800 dark:text-zinc-300">{{ decisionFeedback }}</div>
              </div>
            </div>

            <div v-if="error" class="order-4 flex items-center gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300">
              <AlertCircle class="h-5 w-5 shrink-0" /> {{ error }}
            </div>

            <details class="order-5 group rounded-xl border border-slate-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
              <summary class="flex min-w-0 cursor-pointer list-none items-center gap-3 px-5 py-3.5 text-sm font-medium text-slate-700 dark:text-zinc-300">
                <Clock3 class="h-4 w-4 text-slate-400" />处理记录
                <span class="text-xs font-normal text-slate-400">{{ messages.length ? `${messages.length} 条` : '暂无记录' }}</span>
                <ChevronDown class="ml-auto h-4 w-4 text-slate-400 transition group-open:rotate-180" />
                <button v-if="messages.length" type="button" @click.stop="clearLogs"
                  class="flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200">
                  <Trash2 class="h-3.5 w-3.5" />清空
                </button>
              </summary>
              <div class="border-t border-slate-200 dark:border-zinc-800">
                <div ref="logContainer" class="max-h-44 min-h-24 overflow-y-auto bg-slate-950 px-5 py-4 font-mono text-[11px] leading-5 text-slate-300 select-text">
                  <div v-if="!messages.length" class="py-4 text-center text-slate-600">处理进度和异常信息会显示在这里</div>
                  <div v-for="(message, index) in messages" :key="index" class="break-all">{{ message }}</div>
                </div>
              </div>
            </details>
          </div>
        </div>
        <footer class="flex h-11 shrink-0 items-center gap-3 border-t border-slate-200 bg-white px-7 text-xs text-slate-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
          <span class="h-2 w-2 rounded-full" :class="isServerReady ? 'bg-emerald-500' : 'bg-amber-500'"></span>
          <span>{{ isServerReady ? '本地引擎就绪' : '正在连接本地引擎' }}</span>
          <span class="text-slate-300 dark:text-zinc-700">|</span>
          <span>{{ inputDir ? `已选择 ${folderName}` : '尚未选择目录' }}</span>
        </footer>
      </section>

      <aside ref="settingsScroll" class="settings-scroll min-h-0 bg-white px-7 py-6 dark:bg-zinc-900"
        @scroll.passive="updateStickyActionState">
        <div class="mb-6 flex items-center gap-2">
          <SlidersHorizontal class="h-5 w-5 text-blue-600 dark:text-blue-400" />
          <div><h3 class="font-semibold text-slate-950 dark:text-white">筛选方案</h3><p class="mt-0.5 text-xs text-slate-400">控制归组范围与保留数量</p></div>
        </div>
        <div class="sticky-action sticky top-0 z-20 -mx-2 mb-6 rounded-xl border border-blue-100 bg-white p-2 shadow-[0_8px_24px_rgba(15,23,42,0.08)] dark:border-blue-950 dark:bg-zinc-900"
          :class="{ 'is-pinned': isStickyActionPinned }">
          <span v-if="isStickyActionPinned" class="sticky-glass" aria-hidden="true"></span>
          <button v-if="!isRunning" type="button" @click="handleStart"
            :disabled="!inputDir || isWorkersExceeded || !isServerReady"
            class="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40">
            <Play class="h-4 w-4 fill-white" />开始筛选
          </button>
          <button v-else type="button" @click="cancel"
            class="flex w-full items-center justify-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-5 py-3 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300">
            <Square class="h-4 w-4 fill-current" />停止接收进度
          </button>
        </div>
        <div class="space-y-7">
          <div>
            <label class="mb-3 block text-xs font-semibold text-slate-700 dark:text-zinc-300">处理方式</label>
            <div class="space-y-2">
              <button type="button" @click="reviewMode = 'auto'"
                class="flex w-full items-start gap-3 rounded-lg border p-3 text-left transition"
                :class="reviewMode === 'auto' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/35' : 'border-slate-200 hover:border-blue-300 dark:border-zinc-700 dark:hover:border-blue-700'">
                <span class="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border"
                  :class="reviewMode === 'auto' ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-300 dark:border-zinc-600'">
                  <CheckCircle2 v-if="reviewMode === 'auto'" class="h-3.5 w-3.5" />
                </span>
                <span><span class="block text-xs font-semibold text-slate-900 dark:text-zinc-100">全自动整理</span><span class="mt-1 block text-[10px] leading-4 text-slate-400">程序完成筛选与移动，只展示结果汇总</span></span>
              </button>
              <button type="button" @click="reviewMode = 'review'"
                class="flex w-full items-start gap-3 rounded-lg border p-3 text-left transition"
                :class="reviewMode === 'review' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/35' : 'border-slate-200 hover:border-blue-300 dark:border-zinc-700 dark:hover:border-blue-700'">
                <span class="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border"
                  :class="reviewMode === 'review' ? 'border-blue-600 bg-blue-600 text-white' : 'border-slate-300 dark:border-zinc-600'">
                  <CheckCircle2 v-if="reviewMode === 'review'" class="h-3.5 w-3.5" />
                </span>
                <span><span class="block text-xs font-semibold text-slate-900 dark:text-zinc-100">整理后查看连拍组</span><span class="mt-1 block text-[10px] leading-4 text-slate-400">仍由程序自动完成，结束后提供可选复核视图</span></span>
              </button>
            </div>
          </div>

          <div>
            <label class="mb-3 block text-xs font-semibold text-slate-700 dark:text-zinc-300">筛选强度</label>
            <div class="grid grid-cols-3 overflow-hidden rounded-lg border border-slate-200 bg-slate-50 p-1 dark:border-zinc-700 dark:bg-zinc-800">
              <button v-for="option in presetOptions" :key="option.value" type="button"
                @click="applyPreset(option.value)"
                class="rounded-md px-2 py-2 text-center transition"
                :class="preset === option.value
                  ? 'bg-white text-blue-700 shadow-sm ring-1 ring-slate-200 dark:bg-zinc-700 dark:text-blue-300 dark:ring-zinc-600'
                  : 'text-slate-500 hover:text-slate-800 dark:text-zinc-400 dark:hover:text-zinc-200'">
                <span class="block text-xs font-semibold">{{ option.label }}</span>
                <span class="mt-0.5 block text-[10px]" :class="preset === option.value ? 'text-blue-500' : 'text-slate-400'">{{ option.hint }}</span>
              </button>
            </div>
            <p v-if="preset === 'custom'" class="mt-2 text-[11px] text-blue-600 dark:text-blue-400">正在使用自定义相似度参数</p>
          </div>

          <div>
            <label class="mb-3 block text-xs font-semibold text-slate-700 dark:text-zinc-300">整组全糊时</label>
            <div class="grid grid-cols-3 gap-1 rounded-lg border border-slate-200 bg-slate-50 p-1 dark:border-zinc-700 dark:bg-zinc-800">
              <button v-for="option in [
                { value: 'keep', label: '保守', hint: '仍保留最佳' },
                { value: 'review', label: '复核', hint: '保留并提醒' },
                { value: 'reject', label: '激进', hint: '整组移出' },
              ]" :key="option.value" type="button" @click="allBlurryAction = option.value as AllBlurryAction"
                class="rounded-md px-1 py-2 text-center transition"
                :class="allBlurryAction === option.value ? 'bg-white text-blue-700 shadow-sm ring-1 ring-slate-200 dark:bg-zinc-700 dark:text-blue-300 dark:ring-zinc-600' : 'text-slate-500 dark:text-zinc-400'">
                <span class="block text-xs font-semibold">{{ option.label }}</span>
                <span class="mt-0.5 block text-[9px]">{{ option.hint }}</span>
              </button>
            </div>
            <p class="mt-2 text-[10px] leading-4 text-slate-400">“移出”表示移动到明显废片目录，不会真正删除。</p>
          </div>

          <div>
            <div class="mb-3 flex items-center justify-between">
              <label for="keep-count-slider" class="text-xs font-semibold text-slate-700 dark:text-zinc-300">每组保留</label>
              <output for="keep-count-slider" class="rounded-md bg-blue-50 px-2 py-1 text-xs font-bold tabular-nums text-blue-700 dark:bg-blue-950/60 dark:text-blue-300">
                {{ keepCount }} 张
              </output>
            </div>
            <div class="rounded-lg border border-slate-200 bg-slate-50 px-3 pb-2 pt-4 dark:border-zinc-700 dark:bg-zinc-800/70">
              <input
                id="keep-count-slider"
                :value="keepSliderPosition"
                @input="updateKeepCount"
                @change="snapKeepSliderPosition"
                @keydown.left.prevent="stepKeepCount(-1)"
                @keydown.down.prevent="stepKeepCount(-1)"
                @keydown.right.prevent="stepKeepCount(1)"
                @keydown.up.prevent="stepKeepCount(1)"
                type="range"
                min="0"
                max="1000"
                step="1"
                class="app-range block w-full"
                :style="{ '--range-progress': `${keepSliderPosition / 10}%` }"
                aria-label="每组保留照片数量，1 到 20 张"
                :aria-valuetext="`${keepCount} 张`"
              />
              <div class="relative mx-[9px] mt-2 h-6">
                <button
                  v-for="count in keepCountTicks"
                  :key="count"
                  type="button"
                  @click="setKeepCount(count)"
                  class="absolute top-0 -translate-x-1/2 text-[10px] tabular-nums transition first:translate-x-0 last:-translate-x-full"
                  :class="keepCount === count ? 'font-bold text-blue-600 dark:text-blue-300' : 'text-slate-400 hover:text-slate-700 dark:text-zinc-500 dark:hover:text-zinc-300'"
                  :style="{ left: `${keepCountToSliderPosition(count) / 10}%` }"
                >
                  {{ count }}
                </button>
              </div>
            </div>
            <p class="mt-2 text-[10px] leading-4 text-slate-400">低数量区间刻度更宽，便于精细选择；越往右增长越快。</p>
          </div>

          <div class="rounded-lg border border-slate-200 p-4 dark:border-zinc-700">
            <div class="flex items-center gap-3">
              <span class="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-400"><Layers3 class="h-4 w-4" /></span>
              <div class="min-w-0"><div class="text-xs font-semibold text-slate-800 dark:text-zinc-200">当前美学模型</div><div class="mt-0.5 truncate text-[11px] text-slate-400">使用“模型管理”中的活跃模型</div></div>
            </div>
            <div class="mt-3 grid grid-cols-2 gap-1 rounded-lg bg-slate-50 p-1 dark:bg-zinc-800">
              <button type="button" @click="weightMode = 'adaptive'" class="rounded-md px-2 py-2 text-[11px] font-semibold" :class="weightMode === 'adaptive' ? 'bg-white text-blue-700 shadow-sm dark:bg-zinc-700 dark:text-blue-300' : 'text-slate-500'">智能自适应</button>
              <button type="button" @click="weightMode = 'custom'" class="rounded-md px-2 py-2 text-[11px] font-semibold" :class="weightMode === 'custom' ? 'bg-white text-blue-700 shadow-sm dark:bg-zinc-700 dark:text-blue-300' : 'text-slate-500'">自定义权重</button>
            </div>
            <div v-if="weightMode === 'custom'" class="mt-3 space-y-3 border-t border-slate-100 pt-3 dark:border-zinc-800">
              <div v-for="item in [
                { key: 'sharpness', label: '清晰度' },
                { key: 'aesthetic', label: '审美' },
                { key: 'exposure', label: '曝光' },
              ]" :key="item.key">
                <div class="mb-1 flex justify-between text-[10px] text-slate-500"><span>{{ item.label }}</span><span class="tabular-nums">{{ Math.round(customWeights[item.key as keyof typeof customWeights] * 100) }}%</span></div>
                <input type="range" min="0" max="100" step="1" :value="Math.round(customWeights[item.key as keyof typeof customWeights] * 100)" @input="setCustomWeight(item.key as 'sharpness' | 'aesthetic' | 'exposure', Number(($event.target as HTMLInputElement).value))" class="app-range block w-full" :style="{ '--range-progress': `${customWeights[item.key as keyof typeof customWeights] * 100}%` }" />
              </div>
              <button type="button" @click="applyClarityPreset" class="w-full rounded-md border border-blue-200 px-2 py-2 text-[10px] font-semibold text-blue-700 hover:bg-blue-50 dark:border-blue-900 dark:text-blue-300">应用“清晰优先” 90 / 5 / 5</button>
            </div>
          </div>

          <details class="group rounded-lg border border-slate-200 dark:border-zinc-700">
            <summary class="flex cursor-pointer list-none items-center gap-2 px-4 py-3.5 text-xs font-semibold text-slate-700 dark:text-zinc-300">
              <SlidersHorizontal class="h-4 w-4 text-slate-400" />高级设置
              <ChevronDown class="ml-auto h-4 w-4 text-slate-400 transition group-open:rotate-180" />
            </summary>
            <div class="space-y-4 border-t border-slate-200 px-4 py-4 dark:border-zinc-700">
              <label class="block"><span class="mb-1.5 block text-[11px] text-slate-500 dark:text-zinc-400">连拍时间间隔（秒）</span><input v-model.number="gapSeconds" type="number" min="0.1" max="10" step="0.1" class="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-500/10 dark:border-zinc-700 dark:bg-zinc-800" /></label>
              <label class="block"><span class="mb-1.5 block text-[11px] text-slate-500 dark:text-zinc-400">构图相似度（汉明距离）</span><input v-model.number="maxHammingDistance" @input="syncPresetFromHamming" type="number" min="1" max="64" class="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-500/10 dark:border-zinc-700 dark:bg-zinc-800" /></label>
              <label class="block"><span class="mb-1.5 flex justify-between text-[11px] text-slate-500 dark:text-zinc-400"><span>工作线程</span><span>上限 {{ maxCpus }}</span></span><input v-model.number="maxWorkers" type="number" min="1" :max="maxCpus" class="w-full rounded-lg border bg-slate-50 px-3 py-2 text-sm outline-none focus:ring-2 dark:bg-zinc-800" :class="isWorkersExceeded ? 'border-rose-400 focus:ring-rose-500/10' : 'border-slate-200 focus:border-blue-400 focus:ring-blue-500/10 dark:border-zinc-700'" /></label>
              <label class="block"><span class="mb-1.5 block text-[11px] text-slate-500 dark:text-zinc-400">审查目录名称</span><input v-model="reviewSubdir" type="text" class="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-500/10 dark:border-zinc-700 dark:bg-zinc-800" /></label>
              <label class="block"><span class="mb-1.5 block text-[11px] text-slate-500 dark:text-zinc-400">明显废片目录名称</span><input v-model="defectSubdir" type="text" class="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-500/10 dark:border-zinc-700 dark:bg-zinc-800" /></label>
              <label class="flex cursor-pointer items-start gap-3 rounded-lg bg-slate-50 p-3 dark:bg-zinc-800" :class="!eyeModelReady ? 'opacity-60' : ''">
                <input v-model="eyeDetection" type="checkbox" :disabled="!eyeModelReady" class="mt-0.5 h-4 w-4 rounded accent-blue-600" />
                <span><span class="block text-xs font-medium text-slate-700 dark:text-zinc-300">人像闭眼检测</span><span class="mt-1 block text-[10px] leading-4 text-slate-400">{{ eyeModelReady ? '明确闭眼的照片移入明显废片' : '请先在模型管理中下载人像模型' }}</span></span>
              </label>
              <label class="flex cursor-pointer items-start gap-3 rounded-lg bg-slate-50 p-3 dark:bg-zinc-800">
                <input v-model="useGpu" type="checkbox" :disabled="!gpuDetected" class="mt-0.5 h-4 w-4 rounded accent-blue-600" />
                <span class="min-w-0"><span class="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-zinc-300"><Cpu class="h-3.5 w-3.5 text-blue-500" />硬件加速</span><span class="mt-1 block break-words text-[10px] leading-4 text-slate-400">{{ gpuDeviceName || '正在检测设备…' }}</span></span>
              </label>
            </div>
          </details>

          <div class="rounded-lg bg-blue-50 p-3.5 text-[11px] leading-5 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300">
            <div class="flex gap-2"><ShieldCheck class="mt-0.5 h-4 w-4 shrink-0" /><span>淘汰照片会移入审查目录，不会删除文件或修改原图内容。</span></div>
          </div>

        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.settings-scroll {
  overflow-y: scroll;
  scrollbar-gutter: stable;
}

@media (min-aspect-ratio: 3 / 2) {
  .wide-workspace-grid {
    grid-template-columns: 270px minmax(0, 1fr) clamp(360px, 24vw, 430px);
  }

  .wide-workspace-sidebar {
    display: flex;
  }

  .wide-workspace-description {
    display: block;
  }

  .wide-workspace-header,
  .wide-workspace-directory {
    display: none;
  }

  .wide-workspace-flow {
    height: 390px;
  }

  .wide-workspace-flow-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .wide-workspace-connector {
    display: none;
  }
}

.sticky-action.is-pinned {
  background: rgba(255, 255, 255, 0.48);
}

.sticky-glass {
  position: absolute;
  z-index: 0;
  top: -240px;
  right: -28px;
  bottom: -30px;
  left: -28px;
  display: block;
  pointer-events: none;
  background: linear-gradient(to bottom, rgba(255, 255, 255, 0.42) 0%, rgba(255, 255, 255, 0.42) calc(100% - 24px), rgba(255, 255, 255, 0.16) 100%);
  -webkit-backdrop-filter: blur(28px) saturate(1.35);
  backdrop-filter: blur(28px) saturate(1.35);
  transform: translateZ(0);
  will-change: backdrop-filter;
}

.sticky-action > button {
  position: relative;
  z-index: 1;
}

html.dark .sticky-action.is-pinned {
  /* 黑色毛玻璃：压暗亮色内容，但保留底下界面的轮廓和色彩。 */
  background: rgba(18, 18, 21, 0.78);
  border-color: rgba(63, 63, 70, 0.72);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.38);
}

html.dark .sticky-glass {
  background: linear-gradient(
    to bottom,
    rgba(18, 18, 21, 0.78) 0%,
    rgba(18, 18, 21, 0.78) calc(100% - 24px),
    rgba(18, 18, 21, 0.62) 100%
  );
  -webkit-backdrop-filter: blur(32px) saturate(0.9) brightness(0.72);
  backdrop-filter: blur(32px) saturate(0.9) brightness(0.72);
}

</style>
