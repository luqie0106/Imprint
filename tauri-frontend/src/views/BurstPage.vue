<script setup lang="ts">
import { ref, computed, nextTick, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useSse } from "../composables/useSse";
import {
  Folder,
  Play,
  Square,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Trash2,
  FileCheck,
} from "lucide-vue-next";

const maxCpus = ref(typeof window !== "undefined" && window.navigator.hardwareConcurrency ? window.navigator.hardwareConcurrency : 8);
const defaultWorkers = Math.max(1, Math.round(maxCpus.value * 0.8));

const inputDir = ref("");
const gapSeconds = ref(1.5);
const maxHammingDistance = ref(12);
const keepCount = ref(1);
const maxWorkers = ref(defaultWorkers);
const useGpu = ref(false);
const reviewSubdir = ref("审查_连拍淘汰");

const isWorkersExceeded = computed(() => {
  const w = Number(maxWorkers.value);
  return isNaN(w) || w > maxCpus.value || w < 1;
});

const logContainer = ref<HTMLElement | null>(null);

const { messages, isDone, isRunning, error, resultData, start, cancel } =
  useSse("/api/burst/run");

async function selectDirectory() {
  try {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择待筛选照片所在目录",
    });
    if (selected && typeof selected === "string") {
      inputDir.value = selected;
    }
  } catch (err) {
    console.error("选择目录失败:", err);
  }
}

async function handleStart() {
  if (!inputDir.value) {
    alert("请先选择照片目录！");
    return;
  }
  if (isWorkersExceeded.value) {
    alert(`并发线程数设置无效或超过系统最大 CPU 核心数 (${maxCpus.value})！`);
    return;
  }

  await start({
    input_dir: inputDir.value,
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

// 自动滚动日志终端到底部
watch(
  () => messages.value.length,
  async () => {
    await nextTick();
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight;
    }
  }
);
</script>

<template>
  <div class="h-full flex flex-col gap-5 p-6 overflow-y-auto">
    <!-- 头部说明 -->
    <div>
      <h2 class="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100">
        连拍照片智能优选
      </h2>
      <p class="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
        基于亚秒级 EXIF 拍摄时间、感知哈希及美学质量评分，自动归组并淘汰模糊/废片
      </p>
    </div>

    <!-- 目录选择卡片 -->
    <div
      class="bg-white/70 dark:bg-zinc-900/85 backdrop-blur-md rounded-2xl p-5 shadow-sm border border-zinc-200/60 dark:border-zinc-800/80 flex flex-col gap-3"
    >
      <label class="text-sm font-medium text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
        <Folder class="w-4 h-4 text-indigo-500" />
        照片目录路径
      </label>
      <div class="flex gap-3">
        <input
          v-model="inputDir"
          type="text"
          placeholder="请选择包含 RAW / JPG / HIF 等格式的照片目录..."
          class="flex-1 px-4 py-2.5 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-sm text-zinc-900 dark:text-zinc-100 placeholder-zinc-400 dark:placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
        />
        <button
          @click="selectDirectory"
          class="px-5 py-2.5 rounded-xl bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 text-zinc-800 dark:text-zinc-200 text-sm font-medium transition flex items-center gap-2 shadow-xs cursor-pointer"
        >
          <Folder class="w-4 h-4" />
          浏览选择
        </button>
      </div>
    </div>

    <!-- 参数配置卡片 -->
    <div
      class="bg-white/70 dark:bg-zinc-900/85 backdrop-blur-md rounded-2xl p-5 shadow-sm border border-zinc-200/60 dark:border-zinc-800/80 flex flex-col gap-4"
    >
      <div class="flex items-center gap-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
        <Sliders class="w-4 h-4 text-indigo-500" />
        优选参数设置
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <!-- 时间间隔 -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs text-zinc-500 dark:text-zinc-400">连拍时间间隔阈值 (秒)</label>
          <input
            v-model.number="gapSeconds"
            type="number"
            step="0.1"
            min="0.1"
            max="10"
            class="px-3.5 py-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
          <div class="text-[11px] text-zinc-400 dark:text-zinc-400 mt-0.5">
            相邻快门小于此间隔归为同一连拍组（默认 1.5s）
          </div>
        </div>

        <!-- 感知哈希 -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-center justify-between">
            <label class="text-xs text-zinc-500 dark:text-zinc-400">最大汉明距离 (图像相似度)</label>
            <span class="text-[11px] text-zinc-400 dark:text-zinc-400">范围: 1~64</span>
          </div>
          <input
            v-model.number="maxHammingDistance"
            type="number"
            min="1"
            max="64"
            class="px-3.5 py-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
          <div class="text-[11px] text-zinc-500 dark:text-zinc-400 leading-relaxed space-y-0.5 mt-0.5 bg-zinc-100/60 dark:bg-zinc-950/60 p-2 rounded-lg border border-zinc-200/50 dark:border-zinc-800/80">
            <div>• <span class="text-zinc-700 dark:text-zinc-200 font-medium">6 ~ 8</span>：定点摆拍（要求构图严格一致）</div>
            <div>• <span class="text-indigo-600 dark:text-indigo-400 font-medium">12</span>：常规手持 / 追焦（默认推荐）</div>
            <div>• <span class="text-zinc-700 dark:text-zinc-200 font-medium">16 ~ 20</span>：大幅甩镜头 / 奔跑运动抓拍</div>
          </div>
        </div>

        <!-- 每组保留数 -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs text-zinc-500 dark:text-zinc-400">每组连拍保留张数</label>
          <input
            v-model.number="keepCount"
            type="number"
            min="1"
            max="10"
            class="px-3.5 py-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
          <div class="text-[11px] text-zinc-400 dark:text-zinc-400 mt-0.5">
            每组按画质与美学打分优选保留的前 N 张
          </div>
        </div>

        <!-- 工作线程 -->
        <div class="flex flex-col gap-1.5">
          <div class="flex items-center justify-between">
            <label
              class="text-xs transition-colors"
              :class="isWorkersExceeded ? 'text-rose-600 dark:text-rose-400 font-medium' : 'text-zinc-500 dark:text-zinc-400'"
            >
              并发工作线程数
            </label>
            <span class="text-[11px] text-zinc-400 dark:text-zinc-400">
              (默认80%: {{ defaultWorkers }}，CPU上限: {{ maxCpus }})
            </span>
          </div>
          <input
            v-model.number="maxWorkers"
            type="number"
            min="1"
            class="px-3.5 py-2 rounded-xl border transition-all text-sm focus:outline-none"
            :class="
              isWorkersExceeded
                ? 'border-rose-500 text-rose-600 dark:text-rose-400 bg-rose-50/50 dark:bg-rose-950/50 ring-2 ring-rose-500/20'
                : 'border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-indigo-500/40'
            "
          />
          <span v-if="isWorkersExceeded" class="text-[11px] text-rose-500 dark:text-rose-400 font-medium">
            ⚠️ 超过系统 CPU 逻辑核心数 ({{ maxCpus }}) 或小于 1
          </span>
        </div>

        <!-- 审查子目录名称 -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs text-zinc-500 dark:text-zinc-400">淘汰废片审查目录名称</label>
          <input
            v-model="reviewSubdir"
            type="text"
            class="px-3.5 py-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
        </div>

        <!-- 硬件加速开关 -->
        <div class="flex items-center gap-3 pt-5">
          <input
            id="useGpuCheckbox"
            v-model="useGpu"
            type="checkbox"
            class="w-4 h-4 rounded text-indigo-600 focus:ring-indigo-500 dark:border-zinc-700 dark:bg-zinc-900 cursor-pointer"
          />
          <label
            for="useGpuCheckbox"
            class="text-sm font-medium text-zinc-800 dark:text-zinc-200 flex items-center gap-1.5 cursor-pointer"
          >
            <Cpu class="w-4 h-4 text-emerald-500" />
            启用 GPU 硬件加速推理
          </label>
        </div>
      </div>

      <!-- 操作按钮栏 -->
      <div class="flex items-center justify-between pt-2 border-t border-zinc-200/50 dark:border-zinc-800/80">
        <div class="flex items-center gap-3">
          <button
            v-if="!isRunning"
            @click="handleStart"
            :disabled="!inputDir || isWorkersExceeded"
            class="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-medium transition flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <Play class="w-4 h-4 fill-white" />
            开始连拍筛选
          </button>
          <button
            v-else
            @click="cancel"
            class="px-6 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium transition flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <Square class="w-4 h-4 fill-white" />
            停止筛选
          </button>
        </div>

        <button
          v-if="messages.length > 0"
          @click="clearLogs"
          class="px-3 py-1.5 text-xs text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200 transition flex items-center gap-1.5 cursor-pointer"
        >
          <Trash2 class="w-3.5 h-3.5" />
          清空日志
        </button>
      </div>
    </div>

    <!-- 结果完成卡片 -->
    <div
      v-if="isDone && resultData"
      class="bg-emerald-50/80 dark:bg-emerald-950/40 backdrop-blur-md rounded-2xl p-5 border border-emerald-300 dark:border-emerald-800/60 flex flex-col gap-3 shadow-sm animate-fade-in"
    >
      <div class="flex items-center gap-2 text-emerald-800 dark:text-emerald-300 font-semibold text-base">
        <CheckCircle2 class="w-5 h-5" />
        连拍优选任务已圆满完成！
      </div>

      <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm mt-1">
        <div class="bg-white/60 dark:bg-zinc-900/60 border border-transparent dark:border-zinc-800/60 p-3 rounded-xl">
          <div class="text-xs text-zinc-500 dark:text-zinc-400">总处理照片</div>
          <div class="text-lg font-bold text-zinc-800 dark:text-zinc-100">
            {{ resultData.total }}
          </div>
        </div>

        <div class="bg-white/60 dark:bg-zinc-900/60 border border-transparent dark:border-zinc-800/60 p-3 rounded-xl">
          <div class="text-xs text-zinc-500 dark:text-zinc-400">识别连拍组</div>
          <div class="text-lg font-bold text-indigo-600 dark:text-indigo-400">
            {{ resultData.burst_groups }} 组
          </div>
        </div>

        <div class="bg-white/60 dark:bg-zinc-900/60 border border-transparent dark:border-zinc-800/60 p-3 rounded-xl">
          <div class="text-xs text-zinc-500 dark:text-zinc-400">淘汰移入审查</div>
          <div class="text-lg font-bold text-amber-600 dark:text-amber-400">
            {{ resultData.moved }} 张
          </div>
        </div>

        <div class="bg-white/60 dark:bg-zinc-900/60 border border-transparent dark:border-zinc-800/60 p-3 rounded-xl">
          <div class="text-xs text-zinc-500 dark:text-zinc-400">单张快门跳过</div>
          <div class="text-lg font-bold text-zinc-700 dark:text-zinc-300">
            {{ resultData.skipped_single }} 张
          </div>
        </div>
      </div>

      <div v-if="resultData.review_dir" class="text-xs text-emerald-700 dark:text-emerald-400 mt-1 flex items-center gap-1.5">
        <FileCheck class="w-4 h-4 shrink-0" />
        <span>淘汰照片已安全移至: <span class="font-mono">{{ resultData.review_dir }}</span></span>
      </div>
    </div>

    <!-- 异常提示卡片 -->
    <div
      v-if="error"
      class="bg-rose-50/80 dark:bg-rose-950/40 backdrop-blur-md rounded-2xl p-4 border border-rose-300 dark:border-rose-800/60 flex items-center gap-3 text-sm text-rose-800 dark:text-rose-300 shadow-sm"
    >
      <AlertCircle class="w-5 h-5 shrink-0 text-rose-600 dark:text-rose-400" />
      <div>{{ error }}</div>
    </div>

    <!-- 实时执行日志终端 -->
    <div
      class="flex-1 min-h-[220px] bg-zinc-900 dark:bg-zinc-950/95 text-zinc-200 font-mono text-xs rounded-2xl p-4 shadow-sm border border-zinc-800 flex flex-col overflow-hidden"
    >
      <div class="flex items-center justify-between pb-2 border-b border-zinc-800 text-zinc-400 text-xs">
        <span class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" :class="isRunning ? 'bg-emerald-400 animate-ping' : 'bg-zinc-600'"></span>
          实时处理日志
        </span>
        <span v-if="isRunning" class="text-emerald-400 text-[11px]">处理中...</span>
      </div>
      <div ref="logContainer" class="flex-1 overflow-y-auto mt-2 space-y-1 select-text">
        <div v-if="messages.length === 0" class="text-zinc-600 dark:text-zinc-500 py-4 text-center">
          准备就绪。选择照片目录并点击“开始连拍筛选”即可在此处查看实时进度。
        </div>
        <div v-for="(msg, idx) in messages" :key="idx" class="leading-relaxed break-all">
          {{ msg }}
        </div>
      </div>
    </div>
  </div>
</template>
