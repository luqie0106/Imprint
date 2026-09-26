<script setup lang="ts">
import { ref, watch } from "vue";
import { BASE_URL } from "../stores/api";
import { basicBackend, renderBackend, ricohBackend, sortBackend, type RenderBackend } from "../stores/renderOptions";

const nativeAvailable = ref(false);
const nativeName = ref("原生渲染");
const pytorchAvailable = ref(false);
const pytorchName = ref("PyTorch");
const sortAvailable = ref(false);
const sortName = ref("连拍原生算子");
const loading = ref(false);
const options: { value: RenderBackend; label: string; detail: string }[] = [
  { value: "auto", label: "自动", detail: "优先原生，随后 PyTorch，最后使用 Python CPU" },
  { value: "native", label: "原生渲染", detail: "macOS Metal / Windows D3D12；失败时回退 Python" },
  { value: "pytorch", label: "PyTorch", detail: "仅在已安装并可用时用于去朦胧；基础调整使用 Python" },
  { value: "cpu", label: "Python CPU", detail: "使用现有 Python 计算" },
];

async function refreshStatus() {
  if (!BASE_URL.value) return;
  loading.value = true;
  try {
    const response = await fetch(`${BASE_URL.value}/api/enhance/render-status`);
    if (!response.ok) throw new Error("检测失败");
    const data = await response.json() as {
      native?: { available?: boolean; backend?: string | null };
      pytorch?: { available?: boolean; backends?: string[] };
      sort?: { available?: boolean; backend?: string | null };
    };
    nativeAvailable.value = Boolean(data.native?.available);
    nativeName.value = nativeAvailable.value ? `原生渲染 · ${data.native?.backend || "GPU"}` : "原生渲染 · 不可用";
    pytorchAvailable.value = Boolean(data.pytorch?.available);
    pytorchName.value = pytorchAvailable.value ? `PyTorch · ${data.pytorch?.backends?.join(" / ") || "GPU"}` : "PyTorch · 不可用";
    sortAvailable.value = Boolean(data.sort?.available);
    sortName.value = sortAvailable.value ? "连拍原生算子 · C++ CPU" : "连拍原生算子 · 不可用";
  } catch {
    nativeAvailable.value = false;
    pytorchAvailable.value = false;
    sortAvailable.value = false;
    nativeName.value = "原生渲染 · 检测失败";
    pytorchName.value = "PyTorch · 检测失败";
    sortName.value = "连拍原生算子 · 检测失败";
  } finally {
    loading.value = false;
  }
}
watch(BASE_URL, refreshStatus, { immediate: true });
</script>

<template>
  <div class="h-full overflow-auto p-8">
    <div class="mx-auto max-w-xl space-y-6">
      <div><h2 class="text-xl font-semibold">设置</h2><p class="mt-1 text-sm text-slate-500 dark:text-zinc-400">分别选择各功能的计算方式</p></div>
      <section class="rounded-2xl border border-slate-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-900">
        <div class="mb-4 flex items-center justify-between"><h3 class="text-sm font-semibold">去朦胧</h3><button type="button" class="text-xs text-blue-600 dark:text-blue-400" :disabled="loading" @click="refreshStatus">{{ loading ? "检测中…" : "重新检测" }}</button></div>
        <div class="space-y-2">
          <label v-for="option in options" :key="option.value" class="flex cursor-pointer items-start gap-3 rounded-xl border border-slate-200 p-3 dark:border-zinc-700">
            <input v-model="renderBackend" :value="option.value" type="radio" name="render-backend" class="mt-1 accent-blue-600" />
            <span><span class="block text-sm font-medium">{{ option.label }}</span><span class="block text-xs text-slate-500 dark:text-zinc-400">{{ option.detail }}</span></span>
          </label>
        </div>
      </section>
      <section class="rounded-2xl border border-slate-200 bg-white p-5 text-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 class="mb-3 font-semibold">基础调整</h3>
        <div class="flex gap-5"><label class="flex items-center gap-2"><input v-model="basicBackend" type="radio" value="python" name="basic-backend" />Python</label><label class="flex items-center gap-2"><input v-model="basicBackend" type="radio" value="native" name="basic-backend" />原生</label></div>
      </section>
      <section class="rounded-2xl border border-slate-200 bg-white p-5 text-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 class="mb-3 font-semibold">理光滤镜</h3>
        <div class="flex gap-5"><label class="flex items-center gap-2"><input v-model="ricohBackend" type="radio" value="python" name="ricoh-backend" />Python</label><label class="flex items-center gap-2"><input v-model="ricohBackend" type="radio" value="native" name="ricoh-backend" />原生</label></div>
        <p class="mt-2 text-xs text-slate-500 dark:text-zinc-400">原生模式从当前预设生成查找表，颜色可能与 Python 渲染略有差异。</p>
      </section>
      <section class="rounded-2xl border border-slate-200 bg-white p-5 text-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 class="mb-3 font-semibold">连拍评分</h3>
        <div class="flex gap-5"><label class="flex items-center gap-2"><input v-model="sortBackend" type="radio" value="python" name="sort-backend" />Python</label><label class="flex items-center gap-2"><input v-model="sortBackend" type="radio" value="native" name="sort-backend" />原生</label></div>
        <p class="mt-2 text-xs text-slate-500 dark:text-zinc-400">原生模式计算区域锐度与曝光；美学模型、分组和文件移动仍使用现有流程。</p>
      </section>
      <section class="rounded-2xl border border-slate-200 bg-white p-5 text-sm dark:border-zinc-800 dark:bg-zinc-900">
        <h3 class="mb-3 font-semibold">可用状态</h3>
        <div class="space-y-2"><p :class="nativeAvailable ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-400'">{{ nativeName }}</p><p :class="pytorchAvailable ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-400'">{{ pytorchName }}</p><p :class="sortAvailable ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-400'">{{ sortName }}</p></div>
      </section>
    </div>
  </div>
</template>
