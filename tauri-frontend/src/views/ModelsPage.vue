<script setup lang="ts">
import { ref, onMounted, watch } from "vue";
import { BASE_URL, isServerReady } from "../stores/api";
import { useSse } from "../composables/useSse";
import {
  Sparkles,
  DownloadCloud,
  RefreshCw,
  Zap,
  Flame,
} from "lucide-vue-next";

interface ModelStatusResponse {
  mode: "standard" | "standard_l14" | "custom" | "custom_l14";
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

const isRefreshing = ref(false);
const useMirror = ref(true);

const {
  messages: downloadMessages,
  progressPct: downloadPct,
  isRunning: isDownloading,
  start: startDownload,
} = useSse("/api/models/download");

const {
  messages: fuseMessages,
  isRunning: isFusing,
  isDone: fuseDone,
  error: fuseError,
  start: startFuse,
} = useSse("/api/models/fuse-onnx");

async function triggerFuse(modelType: "b32" | "l14") {
  await startFuse({ model_type: modelType });
  await fetchStatus();
}

async function fetchStatus() {
  if (!BASE_URL.value) return;
  isRefreshing.value = true;
  try {
    const resp = await fetch(`${BASE_URL.value}/api/models/status`);
    if (resp.ok) {
      status.value = await resp.json();
    }
  } catch (err) {
    console.error("获取模型状态失败:", err);
  } finally {
    isRefreshing.value = false;
  }
}

watch(
  () => isServerReady.value,
  (ready) => {
    if (ready) {
      fetchStatus();
    }
  },
  { immediate: true }
);

watch(
  () => BASE_URL.value,
  (url) => {
    if (url) {
      fetchStatus();
    }
  }
);

async function setMode(mode: "standard" | "standard_l14" | "custom" | "custom_l14") {
  try {
    const resp = await fetch(`${BASE_URL.value}/api/models/set-mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (resp.ok) {
      status.value.mode = mode;
    }
  } catch (err) {
    console.error("切换模型模式失败:", err);
  }
}

async function triggerDownload(model: "clip_b32" | "clip_l14") {
  await startDownload({
    model,
    use_mirror: useMirror.value,
  });
  await fetchStatus();
}

onMounted(() => {
  fetchStatus();
});
</script>

<template>
  <div class="h-full flex flex-col gap-5 p-6 overflow-y-auto">
    <!-- 头部说明与刷新 -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
          <Sparkles class="w-6 h-6 text-amber-500" />
          AI 美学模型管理与底座库
        </h2>
        <p class="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
          管理官方 ViT-B/32 与 ViT-L/14 基础模型、ONNX 硬件加速模型及个人专属微调模型
        </p>
      </div>

      <button
        @click="fetchStatus"
        :disabled="isRefreshing"
        class="px-4 py-2 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-700 dark:text-zinc-300 text-xs font-medium transition flex items-center gap-1.5 shadow-xs cursor-pointer"
      >
        <RefreshCw class="w-3.5 h-3.5" :class="{ 'animate-spin': isRefreshing }" />
        刷新状态
      </button>
    </div>

    <!-- 当前激活模型模式选择卡片 -->
    <div
      class="bg-white/70 dark:bg-zinc-900/85 backdrop-blur-md rounded-2xl p-5 shadow-sm border border-zinc-200/60 dark:border-zinc-800/80 flex flex-col gap-4"
    >
      <div class="flex items-center gap-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
        <Zap class="w-4 h-4 text-indigo-500" />
        当前活跃打分模型选择
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
        <!-- 模式 1: Standard B/32 -->
        <div
          @click="setMode('standard')"
          class="p-4 rounded-xl border transition cursor-pointer flex flex-col gap-1.5"
          :class="
            status.mode === 'standard'
              ? 'border-indigo-500 bg-indigo-50/60 dark:bg-indigo-950/60 ring-2 ring-indigo-500/20'
              : 'border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 hover:border-zinc-300 dark:hover:border-zinc-700'
          "
        >
          <div class="flex items-center justify-between">
            <span class="font-semibold text-sm text-zinc-900 dark:text-zinc-100">
              官方通用标准模型 (ViT-B/32)
            </span>
            <span
              class="text-[11px] px-2 py-0.5 rounded-full font-medium"
              :class="status.standard_onnx_ready ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60' : 'bg-amber-100 text-amber-700 dark:bg-amber-950/80 dark:text-amber-300 dark:border dark:border-amber-800/60'"
            >
              {{ status.standard_onnx_ready ? '就绪' : '未就绪' }}
            </span>
          </div>
          <p class="text-xs text-zinc-500 dark:text-zinc-400">
            兼顾极速推理与通用大众审美评分（推荐大多数场景日常使用）
          </p>
        </div>

        <!-- 模式 2: Standard L/14 -->
        <div
          @click="setMode('standard_l14')"
          class="p-4 rounded-xl border transition cursor-pointer flex flex-col gap-1.5"
          :class="
            status.mode === 'standard_l14'
              ? 'border-indigo-500 bg-indigo-50/60 dark:bg-indigo-950/60 ring-2 ring-indigo-500/20'
              : 'border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 hover:border-zinc-300 dark:hover:border-zinc-700'
          "
        >
          <div class="flex items-center justify-between">
            <span class="font-semibold text-sm text-zinc-900 dark:text-zinc-100">
              Aesthetic 3 专业大模型 (ViT-L/14)
            </span>
            <span
              class="text-[11px] px-2 py-0.5 rounded-full font-medium"
              :class="status.standard_l14_onnx_ready ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60' : 'bg-amber-100 text-amber-700 dark:bg-amber-950/80 dark:text-amber-300 dark:border dark:border-amber-800/60'"
            >
              {{ status.standard_l14_onnx_ready ? '就绪' : '未就绪' }}
            </span>
          </div>
          <p class="text-xs text-zinc-500 dark:text-zinc-400">
            LAION-AI 官方专业摄影级高分辨率大底座，细节感知力更强
          </p>
        </div>

        <!-- 模式 3: Custom B/32 -->
        <div
          @click="setMode('custom')"
          class="p-4 rounded-xl border transition cursor-pointer flex flex-col gap-1.5"
          :class="
            status.mode === 'custom'
              ? 'border-indigo-500 bg-indigo-50/60 dark:bg-indigo-950/60 ring-2 ring-indigo-500/20'
              : 'border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 hover:border-zinc-300 dark:hover:border-zinc-700'
          "
        >
          <div class="flex items-center justify-between">
            <span class="font-semibold text-sm text-zinc-900 dark:text-zinc-100">
              个人专属训练模型 (ViT-B/32)
            </span>
            <span
              class="text-[11px] px-2 py-0.5 rounded-full font-medium"
              :class="status.custom_onnx_ready ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60' : 'bg-amber-100 text-amber-700 dark:bg-amber-950/80 dark:text-amber-300 dark:border dark:border-amber-800/60'"
            >
              {{ status.custom_onnx_ready ? '就绪' : '未训练' }}
            </span>
          </div>
          <p class="text-xs text-zinc-500 dark:text-zinc-400">
            由您个人标记的 like/dislike 样本训练出的专属审美风格
          </p>
        </div>

        <!-- 模式 4: Custom L/14 -->
        <div
          @click="setMode('custom_l14')"
          class="p-4 rounded-xl border transition cursor-pointer flex flex-col gap-1.5"
          :class="
            status.mode === 'custom_l14'
              ? 'border-indigo-500 bg-indigo-50/60 dark:bg-indigo-950/60 ring-2 ring-indigo-500/20'
              : 'border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 hover:border-zinc-300 dark:hover:border-zinc-700'
          "
        >
          <div class="flex items-center justify-between">
            <span class="font-semibold text-sm text-zinc-900 dark:text-zinc-100">
              个人专属训练模型 (ViT-L/14)
            </span>
            <span
              class="text-[11px] px-2 py-0.5 rounded-full font-medium"
              :class="status.custom_l14_onnx_ready ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60' : 'bg-amber-100 text-amber-700 dark:bg-amber-950/80 dark:text-amber-300 dark:border dark:border-amber-800/60'"
            >
              {{ status.custom_l14_onnx_ready ? '就绪' : '未训练' }}
            </span>
          </div>
          <p class="text-xs text-zinc-500 dark:text-zinc-400">
            基于 ViT-L/14 专业大底座微调的最高精度个人偏好模型
          </p>
        </div>
      </div>
    </div>

    <!-- 个人 PTH 权重熔铸为 ONNX -->
    <div
      v-if="status.mlp_ready || status.mlp_l14_ready"
      class="bg-white/70 dark:bg-zinc-900/85 backdrop-blur-md rounded-2xl p-5 shadow-sm border border-zinc-200/60 dark:border-zinc-800/80 flex flex-col gap-4"
    >
      <div class="flex items-center gap-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
        <Flame class="w-4 h-4 text-orange-500" />
        个人偏好权重熔铸为 ONNX 推理模型
      </div>
      <p class="text-xs text-zinc-500 dark:text-zinc-400">
        检测到本地已有训练好的 .pth 权重文件，需要熔铸为 ONNX 格式才能被筛选引擎加速调用。
      </p>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- B/32 熔铸 -->
        <div
          v-if="status.mlp_ready"
          class="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 flex flex-col gap-3"
        >
          <div class="flex items-center justify-between">
            <span class="font-medium text-sm text-zinc-800 dark:text-zinc-200">个人模型 (ViT-B/32)</span>
            <span
              class="text-[11px] px-2 py-0.5 rounded-full font-medium"
              :class="status.custom_onnx_ready
                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60'
                : 'bg-orange-100 text-orange-700 dark:bg-orange-950/80 dark:text-orange-300 dark:border dark:border-orange-800/60'"
            >
              {{ status.custom_onnx_ready ? 'ONNX 就绪' : '待熔铸' }}
            </span>
          </div>
          <p class="text-xs text-zinc-500 dark:text-zinc-400 font-mono break-all">{{ status.mlp_path }}</p>
          <button
            @click="triggerFuse('b32')"
            :disabled="isFusing || status.custom_onnx_ready"
            class="px-4 py-2 rounded-xl bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white text-xs font-medium transition flex items-center justify-center gap-2 cursor-pointer shadow-xs"
          >
            <Flame class="w-3.5 h-3.5" />
            {{ status.custom_onnx_ready ? '已熔铸（可重新熔铸）' : '立即熔铸 → ONNX' }}
          </button>
        </div>

        <!-- L/14 熔铸 -->
        <div
          v-if="status.mlp_l14_ready"
          class="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 flex flex-col gap-3"
        >
          <div class="flex items-center justify-between">
            <span class="font-medium text-sm text-zinc-800 dark:text-zinc-200">个人模型 (ViT-L/14)</span>
            <span
              class="text-[11px] px-2 py-0.5 rounded-full font-medium"
              :class="status.custom_l14_onnx_ready
                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60'
                : 'bg-orange-100 text-orange-700 dark:bg-orange-950/80 dark:text-orange-300 dark:border dark:border-orange-800/60'"
            >
              {{ status.custom_l14_onnx_ready ? 'ONNX 就绪' : '待熔铸' }}
            </span>
          </div>
          <p class="text-xs text-zinc-500 dark:text-zinc-400 font-mono break-all">{{ status.mlp_l14_path }}</p>
          <button
            @click="triggerFuse('l14')"
            :disabled="isFusing || status.custom_l14_onnx_ready"
            class="px-4 py-2 rounded-xl bg-orange-500 hover:bg-orange-600 disabled:opacity-50 text-white text-xs font-medium transition flex items-center justify-center gap-2 cursor-pointer shadow-xs"
          >
            <Flame class="w-3.5 h-3.5" />
            {{ status.custom_l14_onnx_ready ? '已熔铸（可重新熔铸）' : '立即熔铸 → ONNX' }}
          </button>
        </div>
      </div>

      <!-- 熔铸进度 -->
      <div v-if="isFusing || fuseMessages.length > 0" class="flex flex-col gap-2">
        <div class="text-xs font-mono bg-zinc-900 dark:bg-zinc-950 text-zinc-300 border border-zinc-800 p-3 rounded-xl max-h-32 overflow-y-auto">
          <div v-for="(msg, idx) in fuseMessages" :key="idx">{{ msg }}</div>
        </div>
        <p v-if="fuseError" class="text-xs text-rose-500">{{ fuseError }}</p>
        <p v-if="fuseDone && !fuseError" class="text-xs text-emerald-600 dark:text-emerald-400 font-medium">
          ✅ 熔铸完成，ONNX 模型已就绪
        </p>
      </div>
    </div>

    <!-- 视觉底座模型下载与同步卡片 -->
    <div
      class="bg-white/70 dark:bg-zinc-900/85 backdrop-blur-md rounded-2xl p-5 shadow-sm border border-zinc-200/60 dark:border-zinc-800/80 flex flex-col gap-4"
    >
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
          <DownloadCloud class="w-4 h-4 text-indigo-500" />
          CLIP 视觉底座下载与离线同步
        </div>

        <!-- 镜像源开关 -->
        <label class="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400 cursor-pointer">
          <input
            v-model="useMirror"
            type="checkbox"
            class="w-3.5 h-3.5 rounded text-indigo-600 focus:ring-indigo-500 dark:border-zinc-700 dark:bg-zinc-900"
          />
          启用国内 HuggingFace 镜像源加速
        </label>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <!-- 下载项 1: ViT-B/32 -->
        <div class="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 flex flex-col justify-between gap-3">
          <div>
            <div class="flex items-center justify-between">
              <span class="font-medium text-sm text-zinc-800 dark:text-zinc-200">
                CLIP ViT-B/32 基础底座
              </span>
              <span
                class="text-[11px] px-2 py-0.5 rounded-full font-medium"
                :class="status.clip_b32_ready ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60' : 'bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 dark:border dark:border-zinc-700/60'"
              >
                {{ status.clip_b32_ready ? '已就绪' : '未下载 (~335MB)' }}
              </span>
            </div>
            <p class="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
              用于个人偏好模型微调与特征提取的基础视觉模型
            </p>
          </div>

          <button
            @click="triggerDownload('clip_b32')"
            :disabled="isDownloading"
            class="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-medium transition flex items-center justify-center gap-2 cursor-pointer shadow-xs"
          >
            <DownloadCloud class="w-3.5 h-3.5" />
            {{ status.clip_b32_ready ? '重新下载/校验' : '一键下载 B/32 底座' }}
          </button>
        </div>

        <!-- 下载项 2: ViT-L/14 -->
        <div class="p-4 rounded-xl border border-zinc-200 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-950/60 flex flex-col justify-between gap-3">
          <div>
            <div class="flex items-center justify-between">
              <span class="font-medium text-sm text-zinc-800 dark:text-zinc-200">
                CLIP ViT-L/14 专业大底座
              </span>
              <span
                class="text-[11px] px-2 py-0.5 rounded-full font-medium"
                :class="status.clip_l14_ready ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300 dark:border dark:border-emerald-800/60' : 'bg-zinc-200 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400 dark:border dark:border-zinc-700/60'"
              >
                {{ status.clip_l14_ready ? '已就绪' : '未下载 (~900MB)' }}
              </span>
            </div>
            <p class="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
              LAION Aesthetic 3 官方专业级大底座模型
            </p>
          </div>

          <button
            @click="triggerDownload('clip_l14')"
            :disabled="isDownloading"
            class="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-medium transition flex items-center justify-center gap-2 cursor-pointer shadow-xs"
          >
            <DownloadCloud class="w-3.5 h-3.5" />
            {{ status.clip_l14_ready ? '重新下载/校验' : '一键下载 L/14 底座' }}
          </button>
        </div>
      </div>

      <!-- 下载进度条与状态显示 -->
      <div v-if="isDownloading || downloadMessages.length > 0" class="pt-2 flex flex-col gap-2">
        <div class="flex items-center justify-between text-xs text-zinc-600 dark:text-zinc-400">
          <span>下载进度日志:</span>
          <span v-if="typeof downloadPct === 'number'" class="font-mono font-medium text-indigo-600 dark:text-indigo-400">
            {{ Math.round(downloadPct * 100) }}%
          </span>
        </div>

        <progress
          class="w-full h-2 rounded-full overflow-hidden [&::-webkit-progress-bar]:bg-zinc-200 dark:[&::-webkit-progress-bar]:bg-zinc-700 [&::-webkit-progress-value]:bg-indigo-600 [&::-moz-progress-bar]:bg-indigo-600"
          :value="downloadPct ?? 0"
          max="1"
        ></progress>

        <div class="text-xs font-mono bg-zinc-900 dark:bg-zinc-950 text-zinc-300 border border-zinc-800 p-3 rounded-xl max-h-32 overflow-y-auto">
          <div v-for="(msg, idx) in downloadMessages" :key="idx">
            {{ msg }}
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
