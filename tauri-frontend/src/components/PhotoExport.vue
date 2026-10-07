<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { open } from "@tauri-apps/plugin-dialog";
import { Download, FolderOpen, LoaderCircle, X } from "lucide-vue-next";
import { BASE_URL } from "../stores/api";
import { basicBackend, renderBackend, ricohBackend } from "../stores/renderOptions";
import { photoExportRunning } from "../stores/photoExport";
import {
  autoSaveError, flushPendingSaves, sharedPhotoSource, sharedSelectedPhotoId,
  sharedBasicByPhoto, sharedPresetByPhoto, sharedDehazeByPhoto,
  sharedDehazeAutoByPhoto, sharedDehazeAutoExposureByPhoto, sharedDehazeNonlocalByPhoto,
} from "../stores/photoSource";
const props = defineProps<{ visible: boolean }>();
interface JobFile { photo_id: string; name: string; status: string; error?: string }
interface Job { job_id: string; status: string; total: number; processed: number; success: number; failed: number; progress: number; current_file: string; files: JobFile[] }
const menuOpen = ref(false);
const dialogOpen = ref(false);
const scope = ref<"current" | "all">("current");
const outputDir = ref("");
const compression = ref("lossless_jpeg");
const bitDepth = ref("source");
const job = ref<Job | null>(null);
const error = ref("");
const starting = ref(false);
const cancelling = ref(false);
let pollTimer: number | undefined;
let polling = false;
const selected = computed(() => sharedPhotoSource.value?.files.find(f => f.photo_id === sharedSelectedPhotoId.value));
const exportFiles = computed(() => scope.value === "current" ? (selected.value ? [selected.value] : []) : sharedPhotoSource.value?.files ?? []);
const failedFiles = computed(() => job.value?.files.filter(f => f.status === "failed") ?? []);
watch(() => sharedPhotoSource.value?.session_id, () => {
  outputDir.value = sharedPhotoSource.value?.default_output_dir ?? "";
  if (!photoExportRunning.value) job.value = null;
}, { immediate: true });
watch(compression, value => { if (value === "jpegxl") bitDepth.value = "16"; });
watch(() => props.visible, () => { menuOpen.value = false; });
function configure(value: "current" | "all") {
  scope.value = value;
  menuOpen.value = false;
  error.value = "";
  dialogOpen.value = true;
}
async function chooseOutput() {
  const result = await open({ multiple: false, directory: true, title: "选择 DNG 输出目录" });
  if (typeof result === "string") outputDir.value = result;
}
async function request(path: string, payload?: object) {
  const response = await fetch(`${BASE_URL.value}${path}`, payload ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
  } : undefined);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "导出请求失败");
  return data;
}
async function pollJob() {
  if (!job.value || polling) return;
  polling = true;
  try {
    job.value = await request(`/api/enhance/job/${job.value.job_id}`);
    photoExportRunning.value = ["queued", "running"].includes(job.value!.status);
    if (!photoExportRunning.value) window.clearInterval(pollTimer);
    error.value = "";
  } catch (cause) { error.value = cause instanceof Error ? cause.message : "无法读取导出进度"; }
  finally { polling = false; }
}
async function startExport() {
  const source = sharedPhotoSource.value;
  if (!source || !exportFiles.value.length || photoExportRunning.value || starting.value) return;
  starting.value = true;
  photoExportRunning.value = true;
  error.value = "";
  const ids = exportFiles.value.map(f => f.photo_id);
  try {
    await flushPendingSaves();
    if (autoSaveError.value) throw new Error(autoSaveError.value);
    const perPhoto = <T,>(values: Record<string, T>, fallback: T) => Object.fromEntries(ids.map(id => [id, values[id] ?? fallback]));
    job.value = await request("/api/enhance/run", {
      session_id: source.session_id, photo_ids: ids, output_dir: outputDir.value,
      params_by_photo: perPhoto(sharedDehazeByPhoto.value, source.dehaze_defaults),
      auto_modes_by_photo: perPhoto(sharedDehazeAutoByPhoto.value, false),
      auto_exposures_by_photo: perPhoto(sharedDehazeAutoExposureByPhoto.value, false),
      nonlocal_modes_by_photo: perPhoto(sharedDehazeNonlocalByPhoto.value, "off"),
      basic_params_by_photo: Object.fromEntries(ids.map(id => [id, sharedBasicByPhoto.value[id]])),
      preset_ids_by_photo: perPhoto(sharedPresetByPhoto.value, null),
      render_backend: renderBackend.value, basic_backend: basicBackend.value, ricoh_backend: ricohBackend.value,
      compression: compression.value, bit_depth: bitDepth.value,
    });
    window.clearInterval(pollTimer);
    pollTimer = window.setInterval(() => void pollJob(), 700);
    await pollJob();
  } catch (cause) {
    photoExportRunning.value = false;
    error.value = cause instanceof Error ? cause.message : "无法启动导出";
  } finally { starting.value = false; }
}
async function cancelExport() {
  if (!job.value || cancelling.value) return;
  cancelling.value = true;
  try { await request(`/api/enhance/cancel/${job.value.job_id}`, {}); await pollJob(); }
  catch (cause) { error.value = cause instanceof Error ? cause.message : "停止失败"; }
  finally { cancelling.value = false; }
}
onBeforeUnmount(() => window.clearInterval(pollTimer));
</script>

<template>
  <div v-show="visible || photoExportRunning" class="relative" @keydown.esc="menuOpen = false">
    <button type="button" @click="photoExportRunning ? dialogOpen = true : menuOpen = !menuOpen" :disabled="!sharedPhotoSource?.files.length" title="导出照片" aria-label="导出照片" :aria-expanded="menuOpen" class="flex items-center gap-1.5 rounded-lg p-1.5 text-slate-500 transition hover:bg-slate-100 disabled:opacity-40 dark:text-zinc-400 dark:hover:bg-zinc-800">
      <LoaderCircle v-if="photoExportRunning" class="h-4 w-4 animate-spin" /><Download v-else class="h-4 w-4" />
      <span v-if="photoExportRunning" class="text-xs">{{ Math.round((job?.progress ?? 0) * 100) }}%</span>
    </button>
    <template v-if="menuOpen">
      <div class="fixed inset-0 z-40" @click="menuOpen = false"></div>
      <div role="menu" class="absolute right-0 top-9 z-50 w-48 rounded-xl border border-slate-200 bg-white p-1.5 text-xs shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
        <button role="menuitem" @click="configure('current')" :disabled="!selected" class="w-full rounded-lg px-3 py-2.5 text-left hover:bg-slate-100 disabled:opacity-40 dark:hover:bg-zinc-800">导出当前照片</button>
        <button role="menuitem" @click="configure('all')" class="w-full rounded-lg px-3 py-2.5 text-left hover:bg-slate-100 dark:hover:bg-zinc-800">导出全部照片（{{ sharedPhotoSource?.files.length }}）</button>
        <button v-if="job" role="menuitem" @click="menuOpen = false; dialogOpen = true" class="mt-1 w-full rounded-lg border-t border-slate-100 px-3 py-2.5 text-left text-slate-500 hover:bg-slate-100 dark:border-zinc-800 dark:hover:bg-zinc-800">查看导出结果</button>
      </div>
    </template>
    <Teleport to="body">
      <div v-if="dialogOpen" class="fixed inset-0 z-[100] flex items-center justify-center bg-black/35 p-6 text-slate-800 backdrop-blur-sm dark:text-zinc-100" @keydown.esc="dialogOpen = false" @click.self="dialogOpen = false">
        <section role="dialog" aria-modal="true" aria-labelledby="export-title" class="max-h-[90vh] w-full max-w-md overflow-y-auto rounded-2xl border border-slate-200 bg-white p-6 shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
          <div class="mb-5 flex items-center justify-between"><h2 id="export-title" class="text-base font-semibold">导出照片</h2><button aria-label="关闭导出设置" @click="dialogOpen = false" class="rounded-lg p-1.5 hover:bg-slate-100 dark:hover:bg-zinc-800"><X class="h-4 w-4" /></button></div>
          <template v-if="!photoExportRunning">
            <div class="mb-4 rounded-xl bg-slate-50 p-3 text-xs dark:bg-zinc-800">{{ scope === 'current' ? selected?.name : `全部 ${exportFiles.length} 张照片` }}</div>
            <div class="space-y-4 text-xs">
              <div class="flex items-center justify-between"><span class="text-slate-500 dark:text-zinc-400">格式</span><span class="font-medium">DNG</span></div>
              <label class="flex items-center justify-between gap-4">位深<select v-model="bitDepth" :disabled="compression === 'jpegxl'" class="rounded-lg border border-slate-200 bg-slate-50 p-2 dark:border-zinc-700 dark:bg-zinc-800"><option value="source">与原始 RAW 相同</option><option value="16">16 位</option></select></label>
              <label class="flex items-center justify-between gap-4">压缩<select v-model="compression" class="rounded-lg border border-slate-200 bg-slate-50 p-2 dark:border-zinc-700 dark:bg-zinc-800"><option value="jpegxl">JPEG XL · 无损</option><option value="lossless_jpeg">JPEG · 无损</option><option value="none">不压缩</option></select></label>
              <p class="text-[11px] leading-5 text-slate-500 dark:text-zinc-400">位深逐张读取；普通图片及无法识别位深的 RAW 使用 16 位。JPEG XL 使用 16 位，需要支持 DNG 1.7 的阅读器。</p>
              <div><p class="mb-2 text-slate-500 dark:text-zinc-400">输出位置</p><button @click="chooseOutput" :title="outputDir" class="flex w-full items-center gap-2 rounded-xl border border-slate-200 p-3 text-left hover:border-blue-400 dark:border-zinc-700"><FolderOpen class="h-4 w-4 shrink-0" /><span class="truncate">{{ outputDir || '选择输出文件夹' }}</span></button></div>
              <p class="rounded-xl bg-blue-50 p-3 text-[11px] leading-5 text-blue-700 dark:bg-blue-950/30 dark:text-blue-300">合并每张照片的去朦胧、理光风格和基础调整。未启用的效果不应用。原始照片保持不变，同名输出自动编号。</p>
            </div>
          </template>
          <div v-if="job" class="mt-4 space-y-2 text-xs">
            <div class="flex justify-between"><span>{{ photoExportRunning ? '正在导出' : job.status === 'cancelled' ? '已停止' : '导出结果' }}</span><span>{{ job.processed }} / {{ job.total }}</span></div>
            <div class="h-1.5 overflow-hidden rounded-full bg-slate-100 dark:bg-zinc-800"><div class="h-full bg-blue-600" :style="{ width: `${job.progress * 100}%` }"></div></div>
            <p>成功 {{ job.success }} 张 · 失败 {{ job.failed }} 张</p><p v-if="photoExportRunning" class="truncate text-slate-500">{{ job.current_file }}</p>
            <p v-for="file in failedFiles" :key="file.photo_id" class="text-rose-600">{{ file.name }}：{{ file.error }}</p>
          </div>
          <p v-if="error || autoSaveError" role="alert" class="mt-4 text-xs text-rose-600">{{ error || autoSaveError }}</p>
          <button v-if="photoExportRunning" @click="cancelExport" :disabled="starting || cancelling" class="mt-5 w-full rounded-xl bg-rose-600 px-4 py-3 text-xs font-semibold text-white disabled:opacity-50">{{ starting ? '正在创建任务…' : cancelling ? '正在停止…' : '停止后续处理' }}</button>
          <button v-else @click="startExport" :disabled="!exportFiles.length" class="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-40"><Download class="h-4 w-4" />开始导出{{ exportFiles.length > 1 ? ` ${exportFiles.length} 张` : '' }}</button>
        </section>
      </div>
    </Teleport>
  </div>
</template>
