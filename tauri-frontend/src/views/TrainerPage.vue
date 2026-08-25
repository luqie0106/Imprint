<script setup lang="ts">
import { ref, nextTick, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { useSse } from "../composables/useSse";
import {
  BrainCircuit,
  Folder,
  Play,
  Square,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Trash2,
} from "lucide-vue-next";

const photosDir = ref("");
const modelType = ref<"standard" | "l14">("standard");
const epochs = ref(15);
const lr = ref(0.001);

const logContainer = ref<HTMLElement | null>(null);

const { messages, isDone, isRunning, error, start, cancel } =
  useSse("/api/trainer/run");

async function selectDirectory() {
  try {
    const selected = await open({
      directory: true,
      multiple: false,
      title: "选择包含 like/ 与 dislike/ 的样本照片目录",
    });
    if (selected && typeof selected === "string") {
      photosDir.value = selected;
    }
  } catch (err) {
    console.error("选择目录失败:", err);
  }
}

async function handleStart() {
  if (!photosDir.value) {
    alert("请先选择训练样本照片目录！");
    return;
  }

  await start({
    photos_dir: photosDir.value,
    model_type: modelType.value,
    epochs: Number(epochs.value),
    lr: Number(lr.value),
  });
}

function clearLogs() {
  messages.value = [];
}

// 自动滚动日志到底部
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
      <h2 class="text-2xl font-bold tracking-tight text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
        <BrainCircuit class="w-6 h-6 text-indigo-500" />
        个人审美偏好模型训练器
      </h2>
      <p class="text-sm text-zinc-500 dark:text-zinc-400 mt-1">
        通过标注的喜欢 (like) 与不喜欢 (dislike) 照片微调个人专属视觉审美打分网络，并一键熔铸为 ONNX 硬件加速模型
      </p>
    </div>

    <!-- 样本目录选择卡片 -->
    <div
      class="bg-white/70 dark:bg-zinc-900/85 backdrop-blur-md rounded-2xl p-5 shadow-sm border border-zinc-200/60 dark:border-zinc-800/80 flex flex-col gap-3"
    >
      <label class="text-sm font-medium text-zinc-700 dark:text-zinc-300 flex items-center gap-2">
        <Folder class="w-4 h-4 text-indigo-500" />
        标注样本数据集目录
      </label>
      <div class="flex gap-3">
        <input
          v-model="photosDir"
          type="text"
          placeholder="请选择包含 like/ 与 dislike/ 两个子文件夹的样本目录..."
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
      <p class="text-xs text-zinc-500 dark:text-zinc-400">
        💡 提示：在所选目录下创建 <code class="px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 font-mono text-zinc-800 dark:text-zinc-200">like</code> 与 <code class="px-1.5 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 font-mono text-zinc-800 dark:text-zinc-200">dislike</code> 文件夹，分别放入您满意和不满意的样张照片即可（支持 RAW / JPG / HIF 等全格式）。
      </p>
    </div>

    <!-- 训练超参数配置卡片 -->
    <div
      class="bg-white/70 dark:bg-zinc-900/85 backdrop-blur-md rounded-2xl p-5 shadow-sm border border-zinc-200/60 dark:border-zinc-800/80 flex flex-col gap-4"
    >
      <div class="flex items-center gap-2 text-sm font-medium text-zinc-700 dark:text-zinc-300">
        <Sliders class="w-4 h-4 text-indigo-500" />
        深度学习微调超参数
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
        <!-- 基础视觉模型底座选择 -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs text-zinc-500 dark:text-zinc-400">视觉特征提取底座</label>
          <select
            v-model="modelType"
            class="px-3.5 py-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40 cursor-pointer"
          >
            <option value="standard">CLIP ViT-B/32 (标准极速 · 512维)</option>
            <option value="l14">CLIP ViT-L/14 (专业高精 · 768维)</option>
          </select>
        </div>

        <!-- 训练轮数 -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs text-zinc-500 dark:text-zinc-400">训练轮数 (Epochs)</label>
          <input
            v-model.number="epochs"
            type="number"
            min="1"
            max="100"
            class="px-3.5 py-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
        </div>

        <!-- 学习率 -->
        <div class="flex flex-col gap-1.5">
          <label class="text-xs text-zinc-500 dark:text-zinc-400">学习率 (Learning Rate)</label>
          <input
            v-model.number="lr"
            type="number"
            step="0.0001"
            min="0.00001"
            max="0.1"
            class="px-3.5 py-2 rounded-xl border border-zinc-300 dark:border-zinc-800 bg-white/50 dark:bg-zinc-950/70 text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/40"
          />
        </div>
      </div>

      <!-- 操作按钮栏 -->
      <div class="flex items-center justify-between pt-2 border-t border-zinc-200/50 dark:border-zinc-800/80">
        <div class="flex items-center gap-3">
          <button
            v-if="!isRunning"
            @click="handleStart"
            :disabled="!photosDir"
            class="px-6 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-sm font-medium transition flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <Play class="w-4 h-4 fill-white" />
            开始微调训练与 ONNX 熔铸
          </button>
          <button
            v-else
            @click="cancel"
            class="px-6 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium transition flex items-center gap-2 shadow-sm cursor-pointer"
          >
            <Square class="w-4 h-4 fill-white" />
            中止训练
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

    <!-- 训练完成卡片 -->
    <div
      v-if="isDone"
      class="bg-emerald-50/80 dark:bg-emerald-950/40 backdrop-blur-md rounded-2xl p-5 border border-emerald-300 dark:border-emerald-800/60 flex flex-col gap-2 shadow-sm animate-fade-in"
    >
      <div class="flex items-center gap-2 text-emerald-800 dark:text-emerald-300 font-semibold text-base">
        <CheckCircle2 class="w-5 h-5" />
        专属审美偏好模型微调与 ONNX 熔铸成功！
      </div>
      <p class="text-xs text-emerald-700 dark:text-emerald-400">
        模型已自动保存并加载至 models/ 目录，您可以在“模型管理”中切换至“个人专属模型”即刻体验！
      </p>
    </div>

    <!-- 异常提示卡片 -->
    <div
      v-if="error"
      class="bg-rose-50/80 dark:bg-rose-950/40 backdrop-blur-md rounded-2xl p-4 border border-rose-300 dark:border-rose-800/60 flex items-center gap-3 text-sm text-rose-800 dark:text-rose-300 shadow-sm"
    >
      <AlertCircle class="w-5 h-5 shrink-0 text-rose-600 dark:text-rose-400" />
      <div>{{ error }}</div>
    </div>

    <!-- 实时训练日志终端 -->
    <div
      class="flex-1 min-h-[220px] bg-zinc-900 dark:bg-zinc-950/95 text-zinc-200 font-mono text-xs rounded-2xl p-4 shadow-sm border border-zinc-800 flex flex-col overflow-hidden"
    >
      <div class="flex items-center justify-between pb-2 border-b border-zinc-800 text-zinc-400 text-xs">
        <span class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" :class="isRunning ? 'bg-emerald-400 animate-ping' : 'bg-zinc-600'"></span>
          训练与熔铸实时控制台输出
        </span>
        <span v-if="isRunning" class="text-emerald-400 text-[11px]">正在微调中...</span>
      </div>
      <div ref="logContainer" class="flex-1 overflow-y-auto mt-2 space-y-1 select-text">
        <div v-if="messages.length === 0" class="text-zinc-600 dark:text-zinc-500 py-4 text-center">
          准备就绪。选择包含 like 与 dislike 的样本目录后点击“开始微调训练与 ONNX 熔铸”。
        </div>
        <div v-for="(msg, idx) in messages" :key="idx" class="leading-relaxed break-all">
          {{ msg }}
        </div>
      </div>
    </div>
  </div>
</template>
