<script setup lang="ts">
import { ref, computed, nextTick, watch, onMounted } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useSse } from "../composables/useSse";
import { BASE_URL, isServerReady } from "../stores/api";
import {
  FolderOpen, Play, Square, SlidersHorizontal, CheckCircle2,
  AlertCircle, Cpu, Trash2, FileCheck2, ChevronDown,
  Images, ScanSearch, Layers3, ShieldCheck, Clock3,
} from "lucide-vue-next";

type FilterPreset = "conservative" | "balanced" | "aggressive" | "custom";

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
const preset = ref<FilterPreset>("balanced");
const logContainer = ref<HTMLElement | null>(null);
const keepSliderPosition = ref(0);
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

async function checkGpuAvailability() {
  if (!BASE_URL.value) return;
  try {
    const response = await fetch(`${BASE_URL.value}/api/models/status`);
    if (!response.ok) return;
    const data = await response.json();
    gpuDetected.value = Boolean(data.gpu_available);
    gpuDeviceName.value = data.gpu_name || "CPU 多核心并行计算";
    useGpu.value = gpuDetected.value;
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
    keep_count: Number(keepCount.value),
    max_workers: Number(maxWorkers.value),
    use_gpu: Boolean(useGpu.value),
  });
}

function clearLogs() {
  messages.value = [];
}

onMounted(checkGpuAvailability);
watch(() => isServerReady.value, (ready) => ready && checkGpuAvailability());
watch(() => messages.value.length, async () => {
  await nextTick();
  if (logContainer.value) logContainer.value.scrollTop = logContainer.value.scrollHeight;
});
</script>

<template>
  <div class="workspace-readable h-full min-h-0 bg-[#f5f7fa] dark:bg-zinc-950">
    <div class="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_clamp(360px,24vw,430px)]">
      <section class="flex min-h-0 flex-col border-r border-slate-200 bg-[#f8fafc] dark:border-zinc-800 dark:bg-zinc-950">
        <div class="flex-1 overflow-y-auto px-7 py-6">
          <div class="mx-auto flex min-h-full w-full max-w-6xl flex-col gap-5">
            <header>
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
              class="group flex w-full items-center gap-4 rounded-xl border border-slate-200 bg-white px-5 py-4 text-left shadow-[0_1px_2px_rgba(15,23,42,0.03)] transition hover:border-blue-300 hover:shadow-[0_8px_24px_rgba(37,99,235,0.08)] dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-blue-700">
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

            <div class="flex min-h-[260px] flex-1 flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-900 shadow-[0_12px_36px_rgba(15,23,42,0.12)] dark:border-zinc-800">
              <div class="flex items-center justify-between border-b border-white/10 px-5 py-3">
                <div class="flex items-center gap-2 text-sm font-medium text-white">
                  <Images class="h-4 w-4 text-blue-400" /> 批处理流程
                </div>
                <span class="text-[11px] text-slate-400">本地处理 · 原图内容不修改</span>
              </div>
              <div class="grid flex-1 grid-cols-2 gap-px bg-white/10 md:grid-cols-4">
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
                  <div v-if="index < 3" class="absolute right-0 top-10 hidden h-px w-5 translate-x-1/2 bg-slate-700 md:block"></div>
                </div>
              </div>
              <div class="border-t border-white/10 bg-slate-950/50 px-5 py-3">
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

            <div v-if="isDone && resultData" class="rounded-xl border border-blue-200 bg-blue-50/70 p-5 dark:border-blue-900 dark:bg-blue-950/25">
              <div class="mb-4 flex items-center gap-2 font-semibold text-blue-950 dark:text-blue-100">
                <CheckCircle2 class="h-5 w-5 text-blue-600 dark:text-blue-400" /> 本次筛选已完成
              </div>
              <div class="grid grid-cols-2 divide-x divide-blue-200 md:grid-cols-4 dark:divide-blue-900">
                <div class="px-4 first:pl-0"><div class="text-2xl font-bold tabular-nums text-slate-950 dark:text-white">{{ resultData.total }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">扫描文件</div></div>
                <div class="px-4"><div class="text-2xl font-bold tabular-nums text-slate-950 dark:text-white">{{ resultData.burst_groups }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">连拍组</div></div>
                <div class="px-4"><div class="text-2xl font-bold tabular-nums text-blue-700 dark:text-blue-300">{{ resultData.moved }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">移入审查</div></div>
                <div class="px-4"><div class="text-2xl font-bold tabular-nums text-slate-950 dark:text-white">{{ resultData.skipped_single }}</div><div class="mt-1 text-xs text-slate-500 dark:text-zinc-400">单张跳过</div></div>
              </div>
              <div v-if="resultData.review_dir" class="mt-4 flex items-center gap-2 border-t border-blue-200 pt-3 text-xs text-blue-800 dark:border-blue-900 dark:text-blue-300">
                <FileCheck2 class="h-4 w-4 shrink-0" /><span class="truncate">审查目录：{{ resultData.review_dir }}</span>
              </div>
            </div>

            <div v-if="error" class="flex items-center gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300">
              <AlertCircle class="h-5 w-5 shrink-0" /> {{ error }}
            </div>

            <details class="group rounded-xl border border-slate-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
              <div class="flex items-center">
                <summary class="flex min-w-0 flex-1 cursor-pointer list-none items-center gap-3 px-5 py-3.5 text-sm font-medium text-slate-700 dark:text-zinc-300">
                  <Clock3 class="h-4 w-4 text-slate-400" />处理记录
                  <span class="text-xs font-normal text-slate-400">{{ messages.length ? `${messages.length} 条` : '暂无记录' }}</span>
                  <ChevronDown class="ml-auto h-4 w-4 text-slate-400 transition group-open:rotate-180" />
                </summary>
                <button v-if="messages.length" type="button" @click.stop="clearLogs"
                  class="mr-4 flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200">
                  <Trash2 class="h-3.5 w-3.5" />清空
                </button>
              </div>
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

      <aside class="settings-scroll min-h-0 bg-white px-7 py-6 dark:bg-zinc-900">
        <div class="mb-6 flex items-center gap-2">
          <SlidersHorizontal class="h-5 w-5 text-blue-600 dark:text-blue-400" />
          <div><h3 class="font-semibold text-slate-950 dark:text-white">筛选方案</h3><p class="mt-0.5 text-xs text-slate-400">控制归组范围与保留数量</p></div>
        </div>
        <div class="space-y-7">
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
                class="keep-slider block w-full"
                :style="{ '--slider-progress': `${keepSliderPosition / 10}%` }"
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
              <label class="flex cursor-pointer items-start gap-3 rounded-lg bg-slate-50 p-3 dark:bg-zinc-800">
                <input v-model="useGpu" type="checkbox" :disabled="!gpuDetected" class="mt-0.5 h-4 w-4 rounded accent-blue-600" />
                <span class="min-w-0"><span class="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-zinc-300"><Cpu class="h-3.5 w-3.5 text-blue-500" />硬件加速</span><span class="mt-1 block break-words text-[10px] leading-4 text-slate-400">{{ gpuDeviceName || '正在检测设备…' }}</span></span>
              </label>
            </div>
          </details>

          <div class="rounded-lg bg-blue-50 p-3.5 text-[11px] leading-5 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300">
            <div class="flex gap-2"><ShieldCheck class="mt-0.5 h-4 w-4 shrink-0" /><span>淘汰照片会移入审查目录，不会删除文件或修改原图内容。</span></div>
          </div>

          <button v-if="!isRunning" type="button" @click="handleStart"
            :disabled="!inputDir || isWorkersExceeded || !isServerReady"
            class="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-[0_8px_20px_rgba(37,99,235,0.22)] transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none">
            <Play class="h-4 w-4 fill-white" />开始筛选
          </button>
          <button v-else type="button" @click="cancel"
            class="flex w-full items-center justify-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-5 py-3 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300">
            <Square class="h-4 w-4 fill-current" />停止接收进度
          </button>
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

.keep-slider {
  --slider-progress: 0%;
  height: 18px;
  appearance: none;
  -webkit-appearance: none;
  cursor: pointer;
  background: transparent;
}

.keep-slider::-webkit-slider-runnable-track {
  height: 4px;
  border-radius: 999px;
  background: linear-gradient(
    to right,
    #2563eb 0,
    #2563eb var(--slider-progress),
    #cbd5e1 var(--slider-progress),
    #cbd5e1 100%
  );
}

.keep-slider::-webkit-slider-thumb {
  width: 18px;
  height: 18px;
  margin-top: -7px;
  appearance: none;
  -webkit-appearance: none;
  border: 3px solid #2563eb;
  border-radius: 999px;
  background: #ffffff;
  box-shadow: 0 1px 4px rgb(15 23 42 / 0.22);
}

.keep-slider::-moz-range-track {
  height: 4px;
  border-radius: 999px;
  background: #cbd5e1;
}

.keep-slider::-moz-range-progress {
  height: 4px;
  border-radius: 999px;
  background: #2563eb;
}

.keep-slider::-moz-range-thumb {
  width: 13px;
  height: 13px;
  border: 3px solid #2563eb;
  border-radius: 999px;
  background: #ffffff;
  box-shadow: 0 1px 4px rgb(15 23 42 / 0.22);
}

:global(.dark) .keep-slider::-webkit-slider-runnable-track {
  background: linear-gradient(
    to right,
    #3b82f6 0,
    #3b82f6 var(--slider-progress),
    #52525b var(--slider-progress),
    #52525b 100%
  );
}

:global(.dark) .keep-slider::-webkit-slider-thumb {
  border-color: #60a5fa;
  background: #18181b;
}
</style>
