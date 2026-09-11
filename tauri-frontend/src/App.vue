<script setup lang="ts">
import { ref, onMounted } from "vue";
import {
  initApiConnection,
  isServerReady,
  isConnecting,
  serverError,
} from "./stores/api";
import {
  themePreference,
  cycleTheme,
} from "./stores/theme";
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
  Sun,
  Moon,
  Monitor,
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
  <div class="w-full h-full flex flex-col bg-slate-50 dark:bg-zinc-950 text-slate-800 dark:text-zinc-100 select-none transition-colors duration-200">
    <!-- 原生标题栏 / 顶部导航区域 (支持拖拽窗口) -->
    <header
      data-tauri-drag-region
      class="h-14 shrink-0 px-6 flex items-center justify-between border-b border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 transition-colors duration-200"
    >
      <!-- Logo 与应用标题 -->
      <div class="flex items-center gap-3" data-tauri-drag-region>
        <img
          :src="appLogo"
          alt="Imprint"
          class="w-8 h-8 rounded-lg object-contain"
        />
        <div>
          <h1 class="text-[15px] font-bold tracking-tight text-slate-950 dark:text-zinc-100 flex items-center gap-2">
            Imprint
          </h1>
        </div>
      </div>

      <!-- 居中 Tab 导航切换栏 -->
      <nav class="flex h-full items-center gap-1">
        <button
          @click="activeTab = 'burst'"
          class="relative h-full px-5 text-[13px] font-medium transition flex items-center gap-2 cursor-pointer after:absolute after:bottom-0 after:left-5 after:right-5 after:h-0.5 after:rounded-full"
          :class="
            activeTab === 'burst'
              ? 'text-blue-700 dark:text-blue-300 font-semibold after:bg-blue-600'
              : 'text-slate-500 dark:text-zinc-400 hover:text-slate-900 dark:hover:text-zinc-100 after:bg-transparent'
          "
        >
          <Images class="w-3.5 h-3.5" />
          连拍优选
        </button>

        <button
          @click="activeTab = 'models'"
          class="relative h-full px-5 text-[13px] font-medium transition flex items-center gap-2 cursor-pointer after:absolute after:bottom-0 after:left-5 after:right-5 after:h-0.5 after:rounded-full"
          :class="
            activeTab === 'models'
              ? 'text-blue-700 dark:text-blue-300 font-semibold after:bg-blue-600'
              : 'text-slate-500 dark:text-zinc-400 hover:text-slate-900 dark:hover:text-zinc-100 after:bg-transparent'
          "
        >
          <Sparkles class="w-3.5 h-3.5" />
          模型管理
        </button>

        <button
          @click="activeTab = 'trainer'"
          class="relative h-full px-5 text-[13px] font-medium transition flex items-center gap-2 cursor-pointer after:absolute after:bottom-0 after:left-5 after:right-5 after:h-0.5 after:rounded-full"
          :class="
            activeTab === 'trainer'
              ? 'text-blue-700 dark:text-blue-300 font-semibold after:bg-blue-600'
              : 'text-slate-500 dark:text-zinc-400 hover:text-slate-900 dark:hover:text-zinc-100 after:bg-transparent'
          "
        >
          <BrainCircuit class="w-3.5 h-3.5" />
          偏好训练
        </button>
      </nav>

      <!-- 右侧控制区: 后端状态指标 + 主题切换按钮 -->
      <div class="flex items-center gap-3 text-[13px]" data-tauri-drag-region>
        <!-- 主题切换按钮 -->
        <button
          @click="cycleTheme"
          class="p-1.5 rounded-lg hover:bg-slate-100 dark:hover:bg-zinc-800 text-slate-500 dark:text-zinc-400 transition flex items-center gap-1.5 cursor-pointer"
          :title="themePreference === 'system' ? '当前: 跟随系统 (点击切换)' : themePreference === 'dark' ? '当前: 深色模式 (点击切换)' : '当前: 浅色模式 (点击切换)'"
        >
          <Monitor v-if="themePreference === 'system'" class="w-3.5 h-3.5 text-blue-500" />
          <Moon v-else-if="themePreference === 'dark'" class="w-3.5 h-3.5 text-blue-400" />
          <Sun v-else class="w-3.5 h-3.5 text-amber-500" />
        </button>

        <!-- 后端就绪状态 -->
        <div
          v-if="isServerReady"
          class="flex items-center gap-1.5 text-slate-500 dark:text-zinc-400 font-medium px-1.5 py-1"
        >
          <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
          <span>本地引擎就绪</span>
        </div>

        <div
          v-else-if="isConnecting"
          class="flex items-center gap-1.5 text-amber-600 dark:text-amber-400 font-medium px-1.5 py-1"
        >
          <RefreshCw class="w-3 h-3 animate-spin" />
          <span>正在连接本地引擎</span>
        </div>

        <div
          v-else
          class="flex items-center gap-1.5 text-rose-600 dark:text-rose-400 font-medium px-1.5 py-1"
        >
          <AlertTriangle class="w-3 h-3" />
          <span>本地引擎离线</span>
        </div>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="flex-1 overflow-hidden relative">
      <!-- 异常友好提示界面（非白屏） -->
      <div
        v-if="!isServerReady && !isConnecting"
        class="absolute inset-0 z-50 flex flex-col items-center justify-center p-8 bg-zinc-100/90 dark:bg-zinc-900/90 backdrop-blur-xl gap-4 text-center"
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
        class="absolute inset-0 z-50 flex flex-col items-center justify-center p-8 bg-zinc-100/80 dark:bg-zinc-900/80 backdrop-blur-xl gap-3 text-center"
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
