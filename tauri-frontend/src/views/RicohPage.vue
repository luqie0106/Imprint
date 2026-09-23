<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { Camera, CheckCircle2, FolderOpen, ImagePlus, LoaderCircle, X, AlertCircle } from "lucide-vue-next";
import { BASE_URL, isServerReady } from "../stores/api";

interface Preset { id: string; model: string; name: string; description: string }
interface ResultFile { name: string; status: string; error?: string }
interface ApplyResult { total: number; written: number; skipped: number; failed: number; files: ResultFile[] }

const presets = ref<Preset[]>([]);
const selectedPreset = ref("");
const paths = ref<string[]>([]);
const loading = ref(false);
const applying = ref(false);
const error = ref("");
const result = ref<ApplyResult | null>(null);
const currentPreset = computed(() => presets.value.find((item) => item.id === selectedPreset.value));

async function loadPresets() {
  if (!isServerReady.value || !BASE_URL.value) return;
  loading.value = true;
  error.value = "";
  try {
    const response = await fetch(`${BASE_URL.value}/api/ricoh/presets`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "无法读取理光风格预设");
    presets.value = data.presets ?? [];
    if (!presets.value.some((item) => item.id === selectedPreset.value)) {
      selectedPreset.value = presets.value[0]?.id ?? "";
    }
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "无法读取理光风格预设";
  } finally {
    loading.value = false;
  }
}

watch([isServerReady, BASE_URL], () => { void loadPresets(); }, { immediate: true });

function addPaths(value: string | string[] | null) {
  if (!value) return;
  const added = Array.isArray(value) ? value : [value];
  paths.value = [...new Set([...paths.value, ...added])];
  result.value = null;
}

async function choosePhotos() {
  addPaths(await open({ multiple: true, directory: false, title: "选择要写入理光 XMP 的照片" }));
}

async function chooseFolder() {
  addPaths(await open({ multiple: false, directory: true, title: "选择照片文件夹" }));
}

async function applyPreset() {
  if (!selectedPreset.value || paths.value.length === 0 || applying.value) return;
  applying.value = true;
  error.value = "";
  result.value = null;
  try {
    const response = await fetch(`${BASE_URL.value}/api/ricoh/apply`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paths: paths.value, preset_id: selectedPreset.value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "写入 XMP 失败");
    result.value = data as ApplyResult;
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : "写入 XMP 失败";
  } finally {
    applying.value = false;
  }
}
</script>

<template>
  <div class="h-full overflow-y-auto p-6 lg:p-8 space-y-6">
    <div class="flex items-start gap-3">
      <div class="rounded-xl bg-blue-100 dark:bg-blue-950 p-2.5 text-blue-700 dark:text-blue-300"><Camera class="w-5 h-5" /></div>
      <div>
        <h2 class="text-xl font-semibold">理光风格</h2>
        <p class="mt-1 text-sm text-slate-500 dark:text-zinc-400">选择 GR2 或 GR3 风格，为照片写入 Camera Raw 可读取的 XMP 调整参数。</p>
      </div>
    </div>

    <div v-if="error" class="flex items-start gap-2 rounded-xl bg-rose-50 dark:bg-rose-950/40 p-3 text-sm text-rose-700 dark:text-rose-300">
      <AlertCircle class="w-4 h-4 shrink-0 mt-0.5" />{{ error }}
    </div>

    <section class="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5">
      <h3 class="font-semibold">1. 选择风格</h3>
      <div v-if="loading" class="mt-4 text-sm text-slate-500 flex items-center gap-2"><LoaderCircle class="w-4 h-4 animate-spin" />正在读取预设…</div>
      <div v-else class="mt-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
        <button v-for="preset in presets" :key="preset.id" type="button" @click="selectedPreset = preset.id; result = null"
          class="rounded-xl border p-4 text-left transition cursor-pointer"
          :class="selectedPreset === preset.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-950/30' : 'border-slate-200 dark:border-zinc-700 hover:border-blue-300'">
          <span class="text-xs font-semibold text-blue-600 dark:text-blue-400">{{ preset.model }}</span>
          <span class="block mt-1 font-medium">{{ preset.name }}</span>
          <span class="block mt-2 text-xs leading-5 text-slate-500 dark:text-zinc-400">{{ preset.description }}</span>
        </button>
      </div>
    </section>

    <section class="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5">
      <h3 class="font-semibold">2. 选择照片</h3>
      <div class="mt-4 flex flex-wrap gap-2">
        <button type="button" @click="choosePhotos" class="flex items-center gap-2 rounded-lg bg-slate-100 dark:bg-zinc-800 px-4 py-2 text-sm cursor-pointer hover:bg-slate-200 dark:hover:bg-zinc-700"><ImagePlus class="w-4 h-4" />添加照片</button>
        <button type="button" @click="chooseFolder" class="flex items-center gap-2 rounded-lg bg-slate-100 dark:bg-zinc-800 px-4 py-2 text-sm cursor-pointer hover:bg-slate-200 dark:hover:bg-zinc-700"><FolderOpen class="w-4 h-4" />添加文件夹</button>
      </div>
      <p v-if="!paths.length" class="mt-4 text-sm text-slate-500">尚未选择照片或文件夹</p>
      <ul v-else class="mt-4 max-h-40 overflow-y-auto space-y-1 text-sm">
        <li v-for="path in paths" :key="path" class="flex items-center justify-between gap-3 rounded-lg bg-slate-50 dark:bg-zinc-800/60 px-3 py-2">
          <span class="truncate" :title="path">{{ path }}</span>
          <button type="button" :aria-label="`移除 ${path}`" @click="paths = paths.filter((item) => item !== path); result = null" class="shrink-0 cursor-pointer text-slate-400 hover:text-rose-600"><X class="w-4 h-4" /></button>
        </li>
      </ul>
    </section>

    <section class="rounded-2xl border border-slate-200 dark:border-zinc-800 bg-white dark:bg-zinc-900 p-5">
      <h3 class="font-semibold">3. 写入 XMP</h3>
      <p class="mt-2 text-sm text-slate-500 dark:text-zinc-400">{{ currentPreset ? `将“${currentPreset.name}”写到所选照片旁的同名 .xmp 文件。` : '请先选择风格。' }}已有 XMP 会跳过；原照片不会修改。请在 Adobe Camera Raw 或 Lightroom 中查看效果。</p>
      <button type="button" :disabled="!selectedPreset || !paths.length || applying || !isServerReady" @click="applyPreset"
        class="mt-4 flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white cursor-pointer hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed">
        <LoaderCircle v-if="applying" class="w-4 h-4 animate-spin" /><CheckCircle2 v-else class="w-4 h-4" />{{ applying ? '正在写入…' : '写入 XMP' }}
      </button>
      <div v-if="result" class="mt-4 text-sm">
        <p class="font-medium">完成：写入 {{ result.written }}，跳过 {{ result.skipped }}，失败 {{ result.failed }}。</p>
        <ul v-if="result.files?.length" class="mt-2 max-h-40 overflow-y-auto space-y-1 text-slate-500 dark:text-zinc-400">
          <li v-for="(file, index) in result.files" :key="index">{{ file.name }} · {{ file.status }}{{ file.error ? `：${file.error}` : '' }}</li>
        </ul>
      </div>
    </section>
  </div>
</template>
