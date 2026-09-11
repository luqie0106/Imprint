<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { BASE_URL, isServerReady } from "../stores/api";
import { useSse } from "../composables/useSse";
import {
  Sparkles, DownloadCloud, RefreshCw, Zap, Flame, ChevronDown,
  Gauge, CheckCircle2, CircleAlert, Database, ShieldCheck,
} from "lucide-vue-next";

type ModelMode = "standard" | "standard_l14" | "custom" | "custom_l14";

interface ModelStatusResponse {
  mode: ModelMode;
  clip_b32_ready: boolean;
  clip_l14_ready: boolean;
  standard_onnx_ready: boolean;
  standard_l14_onnx_ready: boolean;
  custom_onnx_ready: boolean;
  custom_l14_onnx_ready: boolean;
  mlp_ready: boolean;
  mlp_path: string;
  mlp_l14_ready: boolean;
  mlp_l14_path: string;
}

const status = ref<ModelStatusResponse>({
  mode: "standard",
  clip_b32_ready: false,
  clip_l14_ready: false,
  standard_onnx_ready: false,
  standard_l14_onnx_ready: false,
  custom_onnx_ready: false,
  custom_l14_onnx_ready: false,
  mlp_ready: false,
  mlp_path: "",
  mlp_l14_ready: false,
  mlp_l14_path: "",
});

const modelOptions: Array<{
  mode: ModelMode;
  name: string;
  shortName: string;
  family: string;
  description: string;
  readyKey: keyof ModelStatusResponse;
}> = [
  {
    mode: "standard",
    name: "官方通用模型",
    shortName: "标准",
    family: "ViT-B/32",
    description: "速度与画质均衡，适合日常连拍筛选",
    readyKey: "standard_onnx_ready",
  },
  {
    mode: "standard_l14",
    name: "Aesthetic 3 专业模型",
    shortName: "专业",
    family: "ViT-L/14",
    description: "细节感知更强，适合高质量摄影工作流",
    readyKey: "standard_l14_onnx_ready",
  },
  {
    mode: "custom",
    name: "个人偏好模型",
    shortName: "个人",
    family: "ViT-B/32",
    description: "根据您的 like / dislike 样片学习审美",
    readyKey: "custom_onnx_ready",
  },
  {
    mode: "custom_l14",
    name: "个人偏好专业模型",
    shortName: "个人专业",
    family: "ViT-L/14",
    description: "更高精度的个人摄影偏好模型",
    readyKey: "custom_l14_onnx_ready",
  },
];

const isRefreshing = ref(false);
const useMirror = ref(true);

const {
  messages: downloadMessages,
  progressPct: downloadPct,
  isRunning: isDownloading,
  error: downloadError,
  start: startDownload,
} = useSse("/api/models/download");

const {
  messages: fuseMessages,
  isRunning: isFusing,
  isDone: fuseDone,
  error: fuseError,
  start: startFuse,
} = useSse("/api/models/fuse-onnx");

const activeModel = computed(
  () => modelOptions.find((model) => model.mode === status.value.mode) ?? modelOptions[0],
);

const fuseTargets = computed(() => [
  {
    type: "b32" as const,
    name: "个人模型 · ViT-B/32",
    path: status.value.mlp_path,
    available: status.value.mlp_ready,
    ready: status.value.custom_onnx_ready,
  },
  {
    type: "l14" as const,
    name: "个人模型 · ViT-L/14",
    path: status.value.mlp_l14_path,
    available: status.value.mlp_l14_ready,
    ready: status.value.custom_l14_onnx_ready,
  },
].filter((target) => target.available));

function modelIsReady(model: typeof modelOptions[number]) {
  return Boolean(status.value[model.readyKey]);
}

async function fetchStatus() {
  if (!BASE_URL.value) return;
  isRefreshing.value = true;
  try {
    const response = await fetch(`${BASE_URL.value}/api/models/status`);
    if (response.ok) status.value = await response.json();
  } catch (err) {
    console.error("获取模型状态失败:", err);
  } finally {
    isRefreshing.value = false;
  }
}

async function setMode(mode: ModelMode) {
  try {
    const response = await fetch(`${BASE_URL.value}/api/models/set-mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (response.ok) status.value.mode = mode;
  } catch (err) {
    console.error("切换模型模式失败:", err);
  }
}

async function triggerDownload(model: "clip_b32" | "clip_l14") {
  await startDownload({ model, use_mirror: useMirror.value });
  await fetchStatus();
}

async function triggerFuse(modelType: "b32" | "l14") {
  await startFuse({ model_type: modelType });
  await fetchStatus();
}

watch(() => isServerReady.value, (ready) => ready && fetchStatus(), { immediate: true });
watch(() => BASE_URL.value, (url) => url && fetchStatus());
onMounted(fetchStatus);
</script>

<template>
  <div class="workspace-readable h-full min-h-0 bg-[#f5f7fa] dark:bg-zinc-950">
    <div class="grid h-full min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,1fr)_clamp(360px,24vw,430px)]">
      <section class="flex min-h-0 flex-col border-r border-slate-200 bg-[#f8fafc] dark:border-zinc-800 dark:bg-zinc-950">
        <div class="flex-1 overflow-y-auto px-7 py-6">
          <div class="mx-auto flex min-h-full w-full max-w-6xl flex-col gap-5">
            <header class="flex items-start justify-between gap-4">
              <div>
                <div class="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-blue-600 dark:text-blue-400">
                  <Sparkles class="h-3.5 w-3.5" /> Aesthetic model library
                </div>
                <h2 class="text-[28px] font-bold tracking-tight text-slate-950 dark:text-white">模型管理</h2>
                <p class="mt-1 text-sm text-slate-500 dark:text-zinc-400">选择连拍筛选使用的审美模型，并管理本地模型资源</p>
              </div>
              <button @click="fetchStatus" :disabled="isRefreshing"
                class="flex shrink-0 items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-600 transition hover:border-blue-300 hover:text-blue-700 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300">
                <RefreshCw class="h-3.5 w-3.5" :class="{ 'animate-spin': isRefreshing }" />刷新状态
              </button>
            </header>

            <div class="rounded-xl border border-slate-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
              <div class="mb-4 flex items-center justify-between">
                <div class="flex items-center gap-2 text-sm font-semibold text-slate-800 dark:text-zinc-200">
                  <Zap class="h-4 w-4 text-blue-600 dark:text-blue-400" />筛选模型
                </div>
                <span class="text-[11px] text-slate-400">点击即可切换</span>
              </div>
              <div class="grid grid-cols-1 gap-3 md:grid-cols-2">
                <button v-for="model in modelOptions" :key="model.mode" type="button" @click="setMode(model.mode)"
                  class="group relative flex min-h-[116px] flex-col rounded-lg border p-4 text-left transition"
                  :class="status.mode === model.mode
                    ? 'border-blue-500 bg-blue-50/70 shadow-[0_6px_18px_rgba(37,99,235,0.08)] ring-2 ring-blue-500/10 dark:bg-blue-950/30'
                    : 'border-slate-200 bg-slate-50/60 hover:border-blue-300 hover:bg-white dark:border-zinc-700 dark:bg-zinc-800/50 dark:hover:border-blue-700 dark:hover:bg-zinc-800'">
                  <div class="flex w-full items-start justify-between gap-3">
                    <span class="rounded-md px-2 py-1 text-[10px] font-bold"
                      :class="status.mode === model.mode ? 'bg-blue-600 text-white' : 'bg-slate-200 text-slate-600 dark:bg-zinc-700 dark:text-zinc-300'">{{ model.shortName }}</span>
                    <span class="flex items-center gap-1 text-[10px] font-medium"
                      :class="modelIsReady(model) ? 'text-emerald-600 dark:text-emerald-400' : 'text-amber-600 dark:text-amber-400'">
                      <span class="h-1.5 w-1.5 rounded-full" :class="modelIsReady(model) ? 'bg-emerald-500' : 'bg-amber-500'"></span>
                      {{ modelIsReady(model) ? '可用' : '未就绪' }}
                    </span>
                  </div>
                  <div class="mt-3 flex items-baseline gap-2">
                    <span class="text-sm font-semibold text-slate-900 dark:text-white">{{ model.name }}</span>
                    <span class="font-mono text-[10px] text-slate-400">{{ model.family }}</span>
                  </div>
                  <p class="mt-1 text-[11px] leading-5 text-slate-500 dark:text-zinc-400">{{ model.description }}</p>
                  <CheckCircle2 v-if="status.mode === model.mode" class="absolute bottom-3 right-3 h-4 w-4 text-blue-600 dark:text-blue-400" />
                </button>
              </div>
            </div>

            <div class="overflow-hidden rounded-xl border border-slate-200 bg-slate-900 dark:border-zinc-800">
              <div class="flex items-center justify-between border-b border-white/10 px-5 py-4">
                <div class="flex items-center gap-3">
                  <span class="flex h-9 w-9 items-center justify-center rounded-lg bg-blue-500/15 text-blue-300"><Gauge class="h-4 w-4" /></span>
                  <div><div class="text-sm font-semibold text-white">当前启用</div><div class="mt-0.5 text-[11px] text-slate-400">连拍筛选会使用此模型进行审美评分</div></div>
                </div>
                <div class="text-right"><div class="text-sm font-semibold text-blue-300">{{ activeModel.name }}</div><div class="mt-0.5 font-mono text-[10px] text-slate-500">{{ activeModel.family }}</div></div>
              </div>
              <div class="flex items-center gap-2 px-5 py-3 text-[11px] text-slate-400">
                <ShieldCheck class="h-4 w-4 text-emerald-400" />所有模型均保存在本地，照片不会上传到云端
              </div>
            </div>

            <div v-if="fuseTargets.length" class="rounded-xl border border-slate-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
              <div class="mb-2 flex items-center gap-2 text-sm font-semibold text-slate-800 dark:text-zinc-200">
                <Flame class="h-4 w-4 text-blue-600 dark:text-blue-400" />部署个人模型
              </div>
              <p class="mb-4 text-xs text-slate-500 dark:text-zinc-400">将训练权重熔铸为可硬件加速的 ONNX 推理模型。</p>
              <div class="space-y-2">
                <div v-for="target in fuseTargets" :key="target.type" class="flex items-center gap-4 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-zinc-700 dark:bg-zinc-800/60">
                  <Database class="h-4 w-4 shrink-0 text-slate-400" />
                  <div class="min-w-0 flex-1"><div class="text-xs font-semibold text-slate-800 dark:text-zinc-200">{{ target.name }}</div><div class="mt-0.5 truncate font-mono text-[10px] text-slate-400">{{ target.path }}</div></div>
                  <button @click="triggerFuse(target.type)" :disabled="isFusing || target.ready"
                    class="shrink-0 rounded-md bg-blue-600 px-3 py-2 text-[11px] font-semibold text-white transition hover:bg-blue-700 disabled:bg-slate-300 dark:disabled:bg-zinc-700">
                    {{ target.ready ? '已部署' : isFusing ? '处理中…' : '部署模型' }}
                  </button>
                </div>
              </div>
              <details v-if="fuseMessages.length || fuseError" class="group mt-3 rounded-lg border border-slate-200 dark:border-zinc-700">
                <summary class="flex cursor-pointer list-none items-center px-3 py-2 text-[11px] text-slate-500 dark:text-zinc-400">部署记录<ChevronDown class="ml-auto h-3.5 w-3.5 transition group-open:rotate-180" /></summary>
                <div class="max-h-32 overflow-y-auto border-t border-slate-200 bg-slate-950 p-3 font-mono text-[10px] leading-5 text-slate-300 dark:border-zinc-700"><div v-for="(message, index) in fuseMessages" :key="index">{{ message }}</div><div v-if="fuseError" class="text-rose-400">{{ fuseError }}</div><div v-if="fuseDone && !fuseError" class="text-emerald-400">模型部署完成</div></div>
              </details>
            </div>
          </div>
        </div>
        <footer class="flex h-11 shrink-0 items-center gap-3 border-t border-slate-200 bg-white px-7 text-xs text-slate-500 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-400">
          <span class="h-2 w-2 rounded-full" :class="isServerReady ? 'bg-emerald-500' : 'bg-amber-500'"></span>
          <span>{{ isServerReady ? '模型服务就绪' : '正在连接模型服务' }}</span>
          <span class="text-slate-300 dark:text-zinc-700">|</span><span>当前：{{ activeModel.name }}</span>
        </footer>
      </section>

      <aside class="models-settings-scroll min-h-0 bg-white px-7 py-6 dark:bg-zinc-900">
        <div class="mb-6 flex items-center gap-2">
          <DownloadCloud class="h-5 w-5 text-blue-600 dark:text-blue-400" />
          <div><h3 class="font-semibold text-slate-950 dark:text-white">模型资源</h3><p class="mt-0.5 text-xs text-slate-400">下载与校验本地视觉底座</p></div>
        </div>

        <label class="mb-5 flex cursor-pointer items-start gap-3 rounded-lg bg-blue-50 p-3.5 dark:bg-blue-950/35">
          <input v-model="useMirror" type="checkbox" class="mt-0.5 h-4 w-4 accent-blue-600" />
          <span><span class="block text-xs font-semibold text-blue-900 dark:text-blue-200">国内镜像加速</span><span class="mt-1 block text-[10px] leading-4 text-blue-700/70 dark:text-blue-300/70">通过 HuggingFace 镜像下载模型资源</span></span>
        </label>

        <div class="space-y-3">
          <div class="rounded-lg border border-slate-200 p-4 dark:border-zinc-700">
            <div class="flex items-start justify-between gap-2"><div><div class="text-sm font-semibold text-slate-900 dark:text-zinc-100">CLIP ViT-B/32</div><div class="mt-1 text-[11px] text-slate-400">标准底座 · 约 335 MB</div></div><span class="mt-1 h-2 w-2 rounded-full" :class="status.clip_b32_ready ? 'bg-emerald-500' : 'bg-slate-300 dark:bg-zinc-600'"></span></div>
            <button @click="triggerDownload('clip_b32')" :disabled="isDownloading"
              class="mt-4 flex w-full items-center justify-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
              <DownloadCloud class="h-3.5 w-3.5" />{{ status.clip_b32_ready ? '重新下载 / 校验' : '下载标准底座' }}
            </button>
          </div>
          <div class="rounded-lg border border-slate-200 p-4 dark:border-zinc-700">
            <div class="flex items-start justify-between gap-2"><div><div class="text-sm font-semibold text-slate-900 dark:text-zinc-100">CLIP ViT-L/14</div><div class="mt-1 text-[11px] text-slate-400">专业底座 · 约 900 MB</div></div><span class="mt-1 h-2 w-2 rounded-full" :class="status.clip_l14_ready ? 'bg-emerald-500' : 'bg-slate-300 dark:bg-zinc-600'"></span></div>
            <button @click="triggerDownload('clip_l14')" :disabled="isDownloading"
              class="mt-4 flex w-full items-center justify-center gap-1.5 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300">
              <DownloadCloud class="h-3.5 w-3.5" />{{ status.clip_l14_ready ? '重新下载 / 校验' : '下载专业底座' }}
            </button>
          </div>
        </div>

        <div v-if="isDownloading || downloadMessages.length || downloadError" class="mt-5 rounded-lg border border-slate-200 p-4 dark:border-zinc-700">
          <div class="flex items-center justify-between text-xs font-semibold text-slate-700 dark:text-zinc-300"><span>下载进度</span><span v-if="downloadPct !== null" class="tabular-nums text-blue-600 dark:text-blue-400">{{ Math.round(downloadPct * 100) }}%</span></div>
          <div class="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-zinc-700"><div class="h-full rounded-full bg-blue-600 transition-all" :class="downloadPct === null && isDownloading ? 'w-1/3 animate-pulse' : ''" :style="downloadPct !== null ? { width: `${downloadPct * 100}%` } : undefined"></div></div>
          <details v-if="downloadMessages.length || downloadError" class="group mt-3">
            <summary class="flex cursor-pointer list-none items-center text-[10px] text-slate-400">查看下载记录<ChevronDown class="ml-auto h-3.5 w-3.5 transition group-open:rotate-180" /></summary>
            <div class="mt-2 max-h-36 overflow-y-auto rounded-md bg-slate-950 p-3 font-mono text-[10px] leading-5 text-slate-300"><div v-for="(message, index) in downloadMessages" :key="index">{{ message }}</div><div v-if="downloadError" class="text-rose-400">{{ downloadError }}</div></div>
          </details>
        </div>

        <div class="mt-5 flex gap-2 rounded-lg bg-slate-50 p-3 text-[10px] leading-4 text-slate-500 dark:bg-zinc-800 dark:text-zinc-400">
          <CircleAlert class="mt-0.5 h-3.5 w-3.5 shrink-0 text-blue-500" />下载时请保持应用运行。已下载的模型可完全离线使用。
        </div>
      </aside>
    </div>
  </div>
</template>

<style scoped>
.models-settings-scroll {
  overflow-y: scroll;
  scrollbar-gutter: stable;
}
</style>
