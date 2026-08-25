<script setup lang="ts">
import { ref, onMounted } from "vue";
import {
  initApiConnection,
  isServerReady,
  isConnecting,
  serverError,
  apiPort,
} from "./stores/api";
import BurstPage from "./views/BurstPage.vue";
import ModelsPage from "./views/ModelsPage.vue";
import TrainerPage from "./views/TrainerPage.vue";
import appLogo from "./assets/logo.png";
import {
  Images,
  Sparkles,
  BrainCircuit,
  RefreshCw,
  AlertTriangle,
  Server,
} from "lucide-vue-next";

type TabType = "burst" | "models" | "trainer";
const activeTab = ref<TabType>("burst");

async function retryConnection() {
  await initApiConnection();
}

onMounted(() => {
  initApiConnection();
});
</script>

<template>
  <div class="w-full h-full flex flex-col bg-transparent dark:bg-zinc-950/92 dark:backdrop-blur-2xl text-zinc-800 dark:text-zinc-100 select-none">
    <!-- 原生标题栏 / 顶部导航区域 (支持拖拽窗口) -->
    <header
      data-tauri-drag-region
      class="h-14 shrink-0 px-5 flex items-center justify-between border-b border-zinc-200/40 dark:border-zinc-800/80 bg-white/40 dark:bg-zinc-900/80 backdrop-blur-md"
    >
      <!-- Logo 与应用标题 -->
      <div class="flex items-center gap-3" data-tauri-drag-region>
        <img
          :src="appLogo"
          alt="PhotoSort"
          class="w-8 h-8 rounded-xl object-contain shadow-xs"
        />
        <div>
          <h1 class="text-sm font-bold tracking-tight text-zinc-900 dark:text-zinc-100 flex items-center gap-2">
            PhotoSort
            <span class="text-[10px] font-medium px-1.5 py-0.5 rounded-md bg-indigo-100 dark:bg-indigo-950/60 text-indigo-700 dark:text-indigo-300">
              Tauri 2.0
            </span>
          </h1>
        </div>
      </div>

      <!-- 居中 Tab 导航切换栏 -->
      <nav class="flex items-center bg-zinc-200/50 dark:bg-zinc-900/80 p-1 rounded-xl gap-1 border border-transparent dark:border-zinc-800/60">
        <button
          @click="activeTab = 'burst'"
          class="px-4 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5 cursor-pointer"
          :class="
            activeTab === 'burst'
              ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-xs font-semibold'
              : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100'
          "
        >
          <Images class="w-3.5 h-3.5" />
          连拍优选
        </button>

        <button
          @click="activeTab = 'models'"
          class="px-4 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5 cursor-pointer"
          :class="
            activeTab === 'models'
              ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-xs font-semibold'
              : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100'
          "
        >
          <Sparkles class="w-3.5 h-3.5" />
          模型管理
        </button>

        <button
          @click="activeTab = 'trainer'"
          class="px-4 py-1.5 rounded-lg text-xs font-medium transition flex items-center gap-1.5 cursor-pointer"
          :class="
            activeTab === 'trainer'
              ? 'bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-100 shadow-xs font-semibold'
              : 'text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-100'
          "
        >
          <BrainCircuit class="w-3.5 h-3.5" />
          偏好训练
        </button>
      </nav>

      <!-- 右侧后端状态指标 -->
      <div class="flex items-center gap-3 text-xs" data-tauri-drag-region>
        <div
          v-if="isServerReady"
          class="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-medium px-2.5 py-1 rounded-full bg-emerald-50/80 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900"
        >
          <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          <span>Sidecar 就绪 (端口: {{ apiPort }})</span>
        </div>

        <div
          v-else-if="isConnecting"
          class="flex items-center gap-1.5 text-amber-600 dark:text-amber-400 font-medium px-2.5 py-1 rounded-full bg-amber-50/80 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900"
        >
          <RefreshCw class="w-3 h-3 animate-spin" />
          <span>正在连接后端服务...</span>
        </div>

        <div
          v-else
          class="flex items-center gap-1.5 text-rose-600 dark:text-rose-400 font-medium px-2.5 py-1 rounded-full bg-rose-50/80 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900"
        >
          <AlertTriangle class="w-3 h-3" />
          <span>后端离线</span>
        </div>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="flex-1 overflow-hidden relative">
      <!-- 异常友好提示界面（非白屏） -->
      <div
        v-if="!isServerReady && !isConnecting"
        class="absolute inset-0 z-50 flex flex-col items-center justify-center p-8 bg-white/80 dark:bg-zinc-900/80 backdrop-blur-xl gap-4 text-center"
      >
        <div class="w-14 h-14 rounded-2xl bg-rose-100 dark:bg-rose-950/60 text-rose-600 dark:text-rose-400 flex items-center justify-center shadow-sm">
          <Server class="w-7 h-7" />
        </div>
        <h3 class="text-lg font-bold text-zinc-900 dark:text-zinc-100">
          无法连接到 Python FastAPI 后端 Sidecar
        </h3>
        <p class="text-sm text-zinc-500 dark:text-zinc-400 max-w-md">
          {{ serverError || "未能与本地 Python 服务建立通信。请确保已安装所需依赖并正确配置 Conda py311 环境。" }}
        </p>
        <button
          @click="retryConnection"
          class="px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-medium transition flex items-center gap-2 shadow-sm cursor-pointer"
        >
          <RefreshCw class="w-3.5 h-3.5" />
          重新连接后端
        </button>
      </div>

      <!-- 启动等待中界面 -->
      <div
        v-else-if="!isServerReady && isConnecting"
        class="absolute inset-0 z-50 flex flex-col items-center justify-center p-8 bg-white/60 dark:bg-zinc-900/60 backdrop-blur-xl gap-3 text-center"
      >
        <RefreshCw class="w-8 h-8 text-indigo-500 animate-spin" />
        <h3 class="text-sm font-semibold text-zinc-800 dark:text-zinc-200">
          正在初始化 Python FastAPI 后端服务并加载模型...
        </h3>
        <p class="text-xs text-zinc-500 dark:text-zinc-400">
          正在探测本地端口与核心算法模块，请稍候片刻
        </p>
      </div>

      <!-- 正常功能页面切换 -->
      <BurstPage v-show="activeTab === 'burst'" />
      <ModelsPage v-show="activeTab === 'models'" />
      <TrainerPage v-show="activeTab === 'trainer'" />
    </main>
  </div>
</template>