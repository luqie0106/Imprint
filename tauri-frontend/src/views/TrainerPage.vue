<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useSse } from "../composables/useSse";
import {
  BrainCircuit, FolderOpen, Play, Square, SlidersHorizontal,
  CheckCircle2, AlertCircle, Trash2, ChevronDown, Database,
  ScanSearch, Sparkles, Flame, ShieldCheck, Clock3,
} from "lucide-vue-next";

const photosDir = ref("");
const modelType = ref<"standard" | "l14">("standard");
const epochs = ref(15);
const epochSliderPosition = ref(epochToSliderPosition(epochs.value));
const lr = ref(0.001);
const logContainer = ref<HTMLElement | null>(null);
const epochTicks = [1, 5, 15, 30, 60, 100];

const {
  messages, lastMessage, progressPct, isDone, isRunning,
  error, start, cancel,
} = useSse("/api/trainer/run");

const folderName = computed(() => {
  const clean = cleanPath(photosDir.value).replace(/[\\/]+$/, "");
  if (!clean) return "尚未选择样本目录";
  return clean.split(/[\\/]/).pop() || clean;
});

const trainingStage = computed(() => {
  if (isDone.value) return 4;
  if (!isRunning.value) return 0;
  const message = lastMessage.value;
  if (message.includes("ONNX") || message.includes("熔铸")) return 3;
  if (message.includes("Epoch") || message.includes("微调训练")) return 2;
  if (message.includes("特征") || message.includes("底座")) return 1;
  return 0;
});

const progressLabel = computed(() => {
  if (error.value) return "训练遇到问题";
  if (isDone.value) return "个人偏好模型已生成";
  if (isRunning.value) return lastMessage.value || "正在扫描训练样本…";
  return photosDir.value ? "样本目录已就绪，可以开始训练" : "选择包含 like 和 dislike 的样本目录";
});

function cleanPath(value: string): string {
  if (!value) return "";
  return value.trim().replace(/^["']|["']$/g, "").trim();
}

// 展开常用的低轮数区间，高轮数区间逐渐压缩。
function sliderPositionToEpoch(position: number): number {
  const normalized = Math.min(1, Math.max(0, position / 1000));
  return Math.round(1 + 99 * normalized ** 2);
}

function epochToSliderPosition(value: number): number {
  const normalized = Math.min(1, Math.max(0, (value - 1) / 99));
  return Math.round(Math.sqrt(normalized) * 1000);
}

function updateEpochs(event: Event) {
  const position = Number((event.target as HTMLInputElement).value);
  epochSliderPosition.value = position;
  epochs.value = sliderPositionToEpoch(position);
}

function setEpochs(value: number) {
  epochs.value = value;
  epochSliderPosition.value = epochToSliderPosition(value);
}

function snapEpochSlider() {
  setEpochs(epochs.value);
}

function stepEpochs(delta: number) {
  setEpochs(Math.min(100, Math.max(1, epochs.value + delta)));
}

async function selectDirectory() {
  try {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择包含 like/ 与 dislike/ 的样本照片目录",
    });
    if (selected && typeof selected === "string") photosDir.value = cleanPath(selected);
  } catch (err) {
    console.error("选择目录失败:", err);
  }
}

async function handleStart() {
  const directory = cleanPath(photosDir.value);
  photosDir.value = directory;
  if (!directory) {
    alert("请先选择训练样本照片目录！");
    return;
  }
  await start({
    photos_dir: directory,
    model_type: modelType.value,
    epochs: Number(epochs.value),
    lr: Number(lr.value),
  });
}

function clearLogs() {
  messages.value = [];
}

watch(() => messages.value.length, async () => {
  await nextTick();
  if (logContainer.value) logContainer.value.scrollTop = logContainer.value.scrollHeight;
});
</script>

<template>
  <div class="workspace-readable h-full min-h-0 bg-[#f5f7fa] dark:bg-zinc-950">
    <div class="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_clamp(360px,24vw,430px)]">
      <section class="flex min-h-0 flex-col border-r border-slate-200 bg-[#f8fafc] dark:border-zinc-800 dark:bg-zinc-950">
        <div class="workspace-main-scroll flex-1 px-7 py-6">
          <div class="mx-auto flex min-h-full w-full max-w-6xl flex-col gap-5">
            <header>
              <div class="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-blue-600 dark:text-blue-400">
                <BrainCircuit class="h-3.5 w-3.5" /> Preference training studio
              </div>
              <h2 class="text-[28px] font-bold tracking-tight text-slate-950 dark:text-white">偏好训练</h2>
              <p class="mt-1 text-sm text-slate-500 dark:text-zinc-400">用喜欢与不喜欢的样片，训练符合个人摄影审美的本地模型</p>
            </header>

            <button type="button" @click="selectDirectory"
              class="group flex w-full items-center gap-4 rounded-xl border border-slate-200 bg-white px-5 py-4 text-left shadow-[0_1px_2px_rgba(15,23,42,0.03)] transition hover:border-blue-300 hover:shadow-[0_8px_24px_rgba(37,99,235,0.08)] dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-blue-700">
              <span class="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600 dark:bg-blue-950/60 dark:text-blue-400"><FolderOpen class="h-5 w-5" /></span>
              <span class="min-w-0 flex-1">
                <span class="block truncate text-[15px] font-semibold text-slate-900 dark:text-zinc-100">{{ folderName }}</span>
                <span v-if="photosDir" class="mt-0.5 block truncate text-xs text-slate-400 dark:text-zinc-500">{{ photosDir }}</span>
                <span v-else class="mt-0.5 block text-xs text-slate-400 dark:text-zinc-500">目录内需要包含 like 和 dislike 两个子文件夹</span>
              </span>
              <span class="shrink-0 text-xs font-medium text-blue-600 dark:text-blue-400">{{ photosDir ? "更换目录" : "浏览选择" }}</span>
            </button>

            <div class="grid grid-cols-1 gap-3 md:grid-cols-2">
              <div class="rounded-xl border border-blue-200 bg-blue-50/70 p-5 dark:border-blue-900 dark:bg-blue-950/25">
                <div class="flex items-start gap-3">
                  <span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-600 text-white"><Sparkles class="h-4 w-4" /></span>
                  <div><div class="text-sm font-semibold text-blue-950 dark:text-blue-100">like / 喜欢</div><p class="mt-1 text-xs leading-5 text-blue-800/70 dark:text-blue-300/70">放入构图、色彩和氛围符合您审美的照片。</p></div>
                </div>
              </div>
              <div class="rounded-xl border border-slate-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
                <div class="flex items-start gap-3">
                  <span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-800 text-white dark:bg-zinc-700"><Database class="h-4 w-4" /></span>
                  <div><div class="text-sm font-semibold text-slate-900 dark:text-zinc-100">dislike / 不喜欢</div><p class="mt-1 text-xs leading-5 text-slate-500 dark:text-zinc-400">放入不希望模型优先选择的反例照片。</p></div>
                </div>
              </div>
            </div>

            <div class="flex min-h-[280px] flex-1 flex-col overflow-hidden rounded-xl border border-slate-200 bg-slate-900 shadow-[0_12px_36px_rgba(15,23,42,0.12)] dark:border-zinc-800">
              <div class="flex items-center justify-between border-b border-white/10 px-5 py-3">
                <div class="flex items-center gap-2 text-sm font-medium text-white"><ScanSearch class="h-4 w-4 text-blue-400" />训练流程</div>
                <span class="text-[11px] text-slate-400">纯本地 · 自动生成 ONNX</span>
              </div>
              <div class="grid flex-1 grid-cols-2 gap-px bg-white/10 md:grid-cols-4">
                <div v-for="(stage, index) in ['扫描样本', '提取特征', '偏好训练', '生成模型']" :key="stage" class="relative bg-slate-900 px-5 py-6">
                  <div class="mb-3 flex h-8 w-8 items-center justify-center rounded-full border text-xs font-semibold"
                    :class="trainingStage > index
                      ? 'border-blue-500 bg-blue-500 text-white'
                      : trainingStage === index && isRunning
                        ? 'border-blue-400 bg-blue-500/15 text-blue-300 ring-4 ring-blue-500/10'
                        : 'border-slate-700 bg-slate-800 text-slate-400'">
                    <CheckCircle2 v-if="trainingStage > index" class="h-4 w-4" /><span v-else>0{{ index + 1 }}</span>
                  </div>
                  <div class="text-sm font-medium text-slate-100">{{ stage }}</div>
                  <div class="mt-1 text-[11px] leading-5 text-slate-500">{{ ['检查样本分布', 'CLIP 视觉向量', '学习审美差异', '熔铸 ONNX 文件'][index] }}</div>
                  <div v-if="index < 3" class="absolute right-0 top-10 hidden h-px w-5 translate-x-1/2 bg-slate-700 md:block"></div>
                </div>
              </div>
              <div class="border-t border-white/10 bg-slate-950/50 px-5 py-3">
                <div class="flex items-center gap-3">
                  <span class="h-2 w-2 shrink-0 rounded-full" :class="error ? 'bg-rose-500' : isRunning ? 'animate-pulse bg-blue-400' : isDone ? 'bg-emerald-400' : 'bg-slate-600'"></span>
                  <span class="min-w-0 flex-1 truncate text-xs text-slate-300">{{ progressLabel }}</span>
                  <span v-if="progressPct !== null" class="text-[11px] tabular-nums text-blue-300">{{ Math.round(progressPct * 100) }}%</span>
                </div>
                <div v-if="isRunning || progressPct !== null" class="mt-3 h-1 overflow-hidden rounded-full bg-slate-800">
                  <div class="h-full rounded-full bg-blue-500 transition-all duration-500" :class="progressPct === null ? 'w-1/3 animate-pulse' : ''" :style="progressPct !== null ? { width: `${Math.max(3, progressPct * 100)}%` } : undefined"></div>
                </div>
              </div>
            </div>

            <div v-if="isDone" class="rounded-xl border border-blue-200 bg-blue-50/70 p-5 dark:border-blue-900 dark:bg-blue-950/25">
              <div class="flex items-center gap-3"><span class="flex h-10 w-10 items-center justify-center rounded-full bg-blue-600 text-white"><CheckCircle2 class="h-5 w-5" /></span><div><div class="font-semibold text-blue-950 dark:text-blue-100">个人偏好模型训练完成</div><p class="mt-1 text-xs text-blue-800/70 dark:text-blue-300/70">模型已保存到本地，可前往“模型管理”切换使用。</p></div></div>
            </div>

            <div v-if="error" class="flex items-center gap-3 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300"><AlertCircle class="h-5 w-5 shrink-0" />{{ error }}</div>

            <details class="group rounded-xl border border-slate-200 bg-white dark:border-zinc-800 dark:bg-zinc-900">
              <summary class="flex min-w-0 cursor-pointer list-none items-center gap-3 px-5 py-3.5 text-sm font-medium text-slate-700 dark:text-zinc-300">
                <Clock3 class="h-4 w-4 text-slate-400" />训练记录
                <span class="text-xs font-normal text-slate-400">{{ messages.length ? `${messages.length} 条` : '暂无记录' }}</span>
                <ChevronDown class="ml-auto h-4 w-4 text-slate-400 transition group-open:rotate-180" />
                <button v-if="messages.length" type="button" @click.stop="clearLogs"
                  class="flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1.5 text-xs text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200">
                  <Trash2 class="h-3.5 w-3.5" />清空
                </button>
              </summary>
              <div class="border-t border-slate-200 dark:border-zinc-800">
                <div ref="logContainer" class="max-h-48 min-h-24 overflow-y-auto bg-slate-950 px-5 py-4 font-mono text-[11px] leading-5 text-slate-300 select-text">
                  <div v-if="!messages.length" class="py-4 text-center text-slate-600">训练进度和指标会显示在这里</div>
                  <div v-for="(message, index) in messages" :key="index" class="break-all">{{ message }}</div>
                </div>
              </div>
            </details>
          </div>
        </div>
        <footer class="flex h-11 shrink-0 items-center gap-3 border-t border-slate-200 bg-white px-7 text-xs text-slate-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
          <span class="h-2 w-2 rounded-full" :class="isRunning ? 'animate-pulse bg-blue-500' : isDone ? 'bg-emerald-500' : 'bg-slate-400'"></span>
          <span>{{ isRunning ? '正在训练' : isDone ? '训练完成' : '训练器就绪' }}</span>
          <span class="text-slate-300 dark:text-zinc-700">|</span><span>{{ photosDir ? `已选择 ${folderName}` : '尚未选择样本目录' }}</span>
        </footer>
      </section>

      <aside class="trainer-settings-scroll min-h-0 bg-white px-7 py-6 dark:bg-zinc-900">
        <div class="mb-6 flex items-center gap-2">
          <SlidersHorizontal class="h-5 w-5 text-blue-600 dark:text-blue-400" />
          <div><h3 class="font-semibold text-slate-950 dark:text-white">训练设置</h3><p class="mt-0.5 text-xs text-slate-400">选择底座与训练强度</p></div>
        </div>

        <div class="space-y-7">
          <div>
            <label class="mb-3 block text-xs font-semibold text-slate-700 dark:text-zinc-300">视觉底座</label>
            <div class="space-y-2">
              <button type="button" @click="modelType = 'standard'" class="flex w-full items-center gap-3 rounded-lg border p-3 text-left transition"
                :class="modelType === 'standard' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/35' : 'border-slate-200 hover:border-blue-300 dark:border-zinc-700 dark:hover:border-blue-700'">
                <span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg" :class="modelType === 'standard' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-500 dark:bg-zinc-800 dark:text-zinc-400'"><BrainCircuit class="h-4 w-4" /></span>
                <span><span class="block text-xs font-semibold text-slate-900 dark:text-zinc-100">标准 · ViT-B/32</span><span class="mt-0.5 block text-[10px] text-slate-400">训练更快 · 512 维</span></span>
                <CheckCircle2 v-if="modelType === 'standard'" class="ml-auto h-4 w-4 text-blue-600 dark:text-blue-400" />
              </button>
              <button type="button" @click="modelType = 'l14'" class="flex w-full items-center gap-3 rounded-lg border p-3 text-left transition"
                :class="modelType === 'l14' ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/35' : 'border-slate-200 hover:border-blue-300 dark:border-zinc-700 dark:hover:border-blue-700'">
                <span class="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg" :class="modelType === 'l14' ? 'bg-blue-600 text-white' : 'bg-slate-100 text-slate-500 dark:bg-zinc-800 dark:text-zinc-400'"><Sparkles class="h-4 w-4" /></span>
                <span><span class="block text-xs font-semibold text-slate-900 dark:text-zinc-100">专业 · ViT-L/14</span><span class="mt-0.5 block text-[10px] text-slate-400">精度更高 · 768 维</span></span>
                <CheckCircle2 v-if="modelType === 'l14'" class="ml-auto h-4 w-4 text-blue-600 dark:text-blue-400" />
              </button>
            </div>
          </div>

          <div>
            <div class="mb-3 flex items-center justify-between"><label for="epochs" class="text-xs font-semibold text-slate-700 dark:text-zinc-300">训练轮数</label><output for="epochs" class="rounded-md bg-blue-50 px-2 py-1 text-xs font-bold tabular-nums text-blue-700 dark:bg-blue-950/60 dark:text-blue-300">{{ epochs }} 轮</output></div>
            <div class="rounded-lg border border-slate-200 bg-slate-50 px-3 pb-2 pt-4 dark:border-zinc-700 dark:bg-zinc-800/70">
              <input
                id="epochs"
                :value="epochSliderPosition"
                @input="updateEpochs"
                @change="snapEpochSlider"
                @keydown.left.prevent="stepEpochs(-1)"
                @keydown.down.prevent="stepEpochs(-1)"
                @keydown.right.prevent="stepEpochs(1)"
                @keydown.up.prevent="stepEpochs(1)"
                type="range"
                min="0"
                max="1000"
                step="1"
                class="epoch-slider block w-full"
                :style="{ '--slider-progress': `${epochSliderPosition / 10}%` }"
                aria-label="训练轮数，1 到 100 轮"
                :aria-valuetext="`${epochs} 轮`"
              />
              <div class="relative mx-[9px] mt-2 h-6">
                <button
                  v-for="tick in epochTicks"
                  :key="tick"
                  type="button"
                  @click="setEpochs(tick)"
                  class="absolute top-0 -translate-x-1/2 whitespace-nowrap text-[10px] tabular-nums transition first:translate-x-0 last:-translate-x-full"
                  :class="tick === 15
                    ? 'font-semibold text-blue-600 dark:text-blue-300'
                    : epochs === tick
                      ? 'font-bold text-slate-700 dark:text-zinc-200'
                      : 'text-slate-400 hover:text-slate-700 dark:text-zinc-500 dark:hover:text-zinc-300'"
                  :style="{ left: `${epochToSliderPosition(tick) / 10}%` }"
                >
                  {{ tick }}<span v-if="tick === 15" class="ml-0.5">推荐</span>
                </button>
              </div>
            </div>
            <p class="mt-2 text-[10px] leading-4 text-slate-400">常用轮数区间更宽，超过 30 轮后刻度逐渐变密。</p>
          </div>

          <details class="group rounded-lg border border-slate-200 dark:border-zinc-700">
            <summary class="flex cursor-pointer list-none items-center gap-2 px-4 py-3.5 text-xs font-semibold text-slate-700 dark:text-zinc-300">
              <SlidersHorizontal class="h-4 w-4 text-slate-400" />高级设置
              <ChevronDown class="ml-auto h-4 w-4 text-slate-400 transition group-open:rotate-180" />
            </summary>
            <div class="border-t border-slate-200 px-4 py-4 dark:border-zinc-700">
              <label class="block"><span class="mb-1.5 block text-[11px] text-slate-500 dark:text-zinc-400">学习率</span><input v-model.number="lr" type="number" min="0.00001" max="0.1" step="0.0001" class="w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm outline-none focus:border-blue-400 focus:ring-2 focus:ring-blue-500/10 dark:border-zinc-700 dark:bg-zinc-800" /></label>
              <p class="mt-2 text-[10px] leading-4 text-slate-400">默认 0.001。数值过高可能导致训练结果不稳定。</p>
            </div>
          </details>

          <div class="rounded-lg bg-blue-50 p-3.5 text-[11px] leading-5 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300">
            <div class="flex gap-2"><ShieldCheck class="mt-0.5 h-4 w-4 shrink-0" /><span>样片与训练过程全部保留在本机，不会上传照片。</span></div>
          </div>

          <button v-if="!isRunning" type="button" @click="handleStart" :disabled="!photosDir"
            class="flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-3 text-sm font-semibold text-white shadow-[0_8px_20px_rgba(37,99,235,0.22)] transition hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none">
            <Play class="h-4 w-4 fill-white" />开始训练
          </button>
          <button v-else type="button" @click="cancel"
            class="flex w-full items-center justify-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-5 py-3 text-sm font-semibold text-rose-700 transition hover:bg-rose-100 dark:border-rose-900 dark:bg-rose-950/30 dark:text-rose-300">
            <Square class="h-4 w-4 fill-current" />停止接收进度
          </button>

          <div class="flex gap-2 rounded-lg bg-slate-50 p-3 text-[10px] leading-4 text-slate-500 dark:bg-zinc-800 dark:text-zinc-400">
            <Flame class="mt-0.5 h-3.5 w-3.5 shrink-0 text-blue-500" />训练完成后会自动生成可硬件加速的 ONNX 模型。
          </div>
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.trainer-settings-scroll {
  overflow-y: scroll;
  scrollbar-gutter: stable;
}

.epoch-slider {
  --slider-progress: 0%;
  height: 18px;
  appearance: none;
  -webkit-appearance: none;
  cursor: pointer;
  background: transparent;
}

.epoch-slider::-webkit-slider-runnable-track {
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

.epoch-slider::-webkit-slider-thumb {
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

.epoch-slider::-moz-range-track {
  height: 4px;
  border-radius: 999px;
  background: #cbd5e1;
}

.epoch-slider::-moz-range-progress {
  height: 4px;
  border-radius: 999px;
  background: #2563eb;
}

.epoch-slider::-moz-range-thumb {
  width: 13px;
  height: 13px;
  border: 3px solid #2563eb;
  border-radius: 999px;
  background: #ffffff;
  box-shadow: 0 1px 4px rgb(15 23 42 / 0.22);
}

:global(.dark) .epoch-slider::-webkit-slider-runnable-track {
  background: linear-gradient(
    to right,
    #3b82f6 0,
    #3b82f6 var(--slider-progress),
    #52525b var(--slider-progress),
    #52525b 100%
  );
}

:global(.dark) .epoch-slider::-webkit-slider-thumb {
  border-color: #60a5fa;
  background: #18181b;
}
</style>
