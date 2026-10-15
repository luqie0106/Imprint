<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { ChevronDown } from "lucide-vue-next";
import {
  ensureAcrColorConfig, getAcrColorConfig, resetAcrColorConfig,
  sharedAcrColorByPhoto, type AcrColorRenderStatus,
} from "../stores/acrColor";

const props = withDefaults(defineProps<{
  photoId: string;
  presetApplied?: boolean;
  showingOriginal?: boolean;
  status: AcrColorRenderStatus;
}>(), {
  presetApplied: false,
  showingOriginal: false,
});
const emit = defineEmits<{ change: [] }>();

const hslColors = ["红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "洋红"];
const gradingRanges = ["暗部", "中间调", "高光", "全局"];
const hslChannels = [
  { label: "色相", min: -100, max: 100, step: 1 },
  { label: "饱和度", min: -100, max: 100, step: 1 },
  { label: "明亮度", min: -100, max: 100, step: 1 },
];
const gradingChannels = [
  { label: "色相", min: 0, max: 360, step: 1 },
  { label: "饱和度", min: 0, max: 100, step: 1 },
  { label: "明亮度", min: -100, max: 100, step: 1 },
];
const selectedHslColor = ref(0);
const selectedGradingRange = ref(0);

watch(() => props.photoId, photoId => ensureAcrColorConfig(photoId), { immediate: true });

const config = computed(() => {
  const saved = props.photoId ? sharedAcrColorByPhoto.value[props.photoId] : undefined;
  return saved ?? getAcrColorConfig("");
});
const controlsDisabled = computed(() => !props.photoId || props.presetApplied);
const statusText = computed(() => {
  if (props.showingOriginal) return "原图：未应用";
  if (props.presetApplied) return "理光预设启用期间暂停";
  if (!config.value.enabled) return "关闭";
  if (props.status === "applied") return "后端报告：已应用";
  if (props.status === "fallback") return "后端报告：已回退";
  return "等待预览结果";
});

function hslIndex(channel: number) {
  return channel * 8 + selectedHslColor.value;
}

function gradingIndex(channel: number) {
  return selectedGradingRange.value * 3 + channel;
}

function setEnabled(event: Event) {
  if (!props.photoId || props.presetApplied) return;
  config.value.enabled = (event.target as HTMLInputElement).checked;
}

function reset() {
  resetAcrColorConfig(props.photoId);
  emit("change");
}
</script>

<template>
  <section class="min-w-0 rounded-xl border border-violet-200 bg-white p-3 text-slate-900 shadow-sm dark:border-violet-900/70 dark:bg-zinc-900 dark:text-white" @input="emit('change')">
    <details class="group min-w-0">
      <summary class="flex min-w-0 cursor-pointer list-none items-center gap-2">
        <ChevronDown class="acr-color-chevron h-3.5 w-3.5 shrink-0 text-violet-600 transition dark:text-violet-300" />
        <span class="min-w-0 flex-1 truncate text-sm font-semibold">ACR CPU 颜色实验</span>
        <span class="shrink-0 text-[10px] text-slate-500 dark:text-zinc-300" role="status">{{ statusText }}</span>
      </summary>

      <div class="mt-3 min-w-0 border-t border-slate-100 pt-3 dark:border-zinc-800">
        <div class="mb-3 flex min-w-0 items-start justify-between gap-2">
          <div class="min-w-0">
            <label class="flex items-center gap-2 text-xs font-medium">
              <input :checked="config.enabled" :disabled="controlsDisabled" type="checkbox" class="h-4 w-4 shrink-0 accent-violet-600" @change="setEnabled" />
              <span>启用 ACR CPU 颜色实验</span>
            </label>
            <p v-if="presetApplied" class="mt-1 text-[10px] leading-4 text-amber-700 dark:text-amber-300">当前照片应用了理光预设，实验暂停；移除预设后可恢复开关状态。</p>
            <p v-else class="mt-1 text-[10px] leading-4 text-slate-500 dark:text-zinc-300">仅预览实验；导出仍使用现有处理。</p>
          </div>
          <button type="button" :disabled="!photoId" class="shrink-0 rounded px-2 py-1 text-[10px] text-slate-500 hover:bg-slate-100 hover:text-violet-700 disabled:opacity-40 dark:text-zinc-300 dark:hover:bg-zinc-800" @click="reset">重置</button>
        </div>

        <label class="mb-3 flex min-w-0 items-center gap-2 text-[11px]">
          <input v-model="config.profile_tone" :disabled="controlsDisabled" type="checkbox" class="h-3.5 w-3.5 shrink-0 accent-violet-600" />
          <span>新版 Profile 曲线</span>
        </label>

        <div class="min-w-0 space-y-3">
          <section class="min-w-0 rounded-lg bg-slate-50 p-2.5 dark:bg-zinc-800/70">
            <h3 class="mb-2 text-[11px] font-semibold">HSL</h3>
            <label class="mb-2 block min-w-0 text-[10px]">
              <span class="mb-1 block">色相范围</span>
              <select v-model.number="selectedHslColor" class="block w-full min-w-0 rounded-md border border-slate-200 bg-white px-2 py-1.5 text-[11px] dark:border-zinc-700 dark:bg-zinc-900" :disabled="controlsDisabled" aria-label="HSL 色相范围">
                <option v-for="(color, index) in hslColors" :key="color" :value="index">{{ color }}</option>
              </select>
            </label>
            <label v-for="(channel, index) in hslChannels" :key="channel.label" class="mb-1.5 block min-w-0 text-[10px] last:mb-0">
              <span class="flex min-w-0 justify-between gap-2"><span>{{ channel.label }}</span><span class="shrink-0 font-mono">{{ config.hsl[hslIndex(index)] }}</span></span>
              <input v-model.number="config.hsl[hslIndex(index)]" class="app-range mt-0.5 block w-full min-w-0" type="range" :min="channel.min" :max="channel.max" :step="channel.step" :disabled="controlsDisabled" :aria-label="`HSL ${hslColors[selectedHslColor]} ${channel.label}`" />
            </label>
            <label class="mt-2 block min-w-0 text-[10px]">
              <span class="flex min-w-0 justify-between gap-2"><span>自然饱和度</span><span class="shrink-0 font-mono">{{ config.hsl[24] }}</span></span>
              <input v-model.number="config.hsl[24]" class="app-range mt-0.5 block w-full min-w-0" type="range" min="-100" max="100" step="1" :disabled="controlsDisabled" aria-label="HSL 自然饱和度" />
            </label>
          </section>

          <section class="min-w-0 rounded-lg bg-slate-50 p-2.5 dark:bg-zinc-800/70">
            <h3 class="mb-2 text-[11px] font-semibold">颜色分级</h3>
            <label class="mb-2 block min-w-0 text-[10px]">
              <span class="mb-1 block">色轮范围</span>
              <select v-model.number="selectedGradingRange" class="block w-full min-w-0 rounded-md border border-slate-200 bg-white px-2 py-1.5 text-[11px] dark:border-zinc-700 dark:bg-zinc-900" :disabled="controlsDisabled" aria-label="颜色分级色轮范围">
                <option v-for="(range, index) in gradingRanges" :key="range" :value="index">{{ range }}</option>
              </select>
            </label>
            <label v-for="(channel, index) in gradingChannels" :key="channel.label" class="mb-1.5 block min-w-0 text-[10px] last:mb-0">
              <span class="flex min-w-0 justify-between gap-2"><span>{{ channel.label }}</span><span class="shrink-0 font-mono">{{ config.grading[gradingIndex(index)] }}</span></span>
              <input v-model.number="config.grading[gradingIndex(index)]" class="app-range mt-0.5 block w-full min-w-0" type="range" :min="channel.min" :max="channel.max" :step="channel.step" :disabled="controlsDisabled" :aria-label="`颜色分级 ${gradingRanges[selectedGradingRange]} ${channel.label}`" />
            </label>
            <label class="mt-2 block min-w-0 text-[10px]">
              <span class="flex min-w-0 justify-between gap-2"><span>混合</span><span class="shrink-0 font-mono">{{ config.grading[12] }}</span></span>
              <input v-model.number="config.grading[12]" class="app-range mt-0.5 block w-full min-w-0" type="range" min="0" max="100" step="1" :disabled="controlsDisabled" aria-label="颜色分级混合" />
            </label>
            <label class="mt-2 block min-w-0 text-[10px]">
              <span class="flex min-w-0 justify-between gap-2"><span>平衡</span><span class="shrink-0 font-mono">{{ config.grading[13] }}</span></span>
              <input v-model.number="config.grading[13]" class="app-range mt-0.5 block w-full min-w-0" type="range" min="-100" max="100" step="1" :disabled="controlsDisabled" aria-label="颜色分级平衡" />
            </label>
          </section>
        </div>
      </div>
    </details>
  </section>
</template>

<style scoped>
summary::-webkit-details-marker {
  display: none;
}

details[open] > summary .acr-color-chevron {
  transform: rotate(180deg);
}
</style>
