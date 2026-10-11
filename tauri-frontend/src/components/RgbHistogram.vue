<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import AdjustmentPanel from "./AdjustmentPanel.vue";
import { BASE_URL } from "../stores/api";

interface PhotoMetadata {
  focal_length: number | null;
  lens_focal_range: [number, number] | null;
  aperture: number | null;
  exposure_time: number | null;
}

const props = defineProps<{ src: string; sessionId: string; photoId: string }>();
const paths = ref<string[]>([]);
const status = ref("选择照片后显示");
const metadata = ref<PhotoMetadata | null>(null);
const showLensFocalRange = computed(() => {
  const range = metadata.value?.lens_focal_range;
  return Boolean(range && range[0] !== range[1]);
});
const channels = [
  { label: "R", color: "#ef4444" },
  { label: "G", color: "#22c55e" },
  { label: "B", color: "#2563eb" },
];
let generation = 0;
let pendingImage: HTMLImageElement | null = null;
let metadataGeneration = 0;
let metadataController: AbortController | null = null;

function cancelLoad() {
  if (!pendingImage) return;
  pendingImage.onload = null;
  pendingImage.onerror = null;
  pendingImage.src = "";
  pendingImage = null;
}

watch(() => props.src, (src) => {
  const token = ++generation;
  cancelLoad();
  paths.value = [];
  status.value = src ? "正在读取直方图…" : "选择照片后显示";
  if (!src) return;

  const image = new Image();
  pendingImage = image;
  // Reuse the existing preview; no additional backend processing or preview requests.
  image.crossOrigin = "anonymous";
  image.onload = () => {
    if (token !== generation) return;
    pendingImage = null;
    try {
      const canvas = document.createElement("canvas");
      const scale = Math.min(1, 512 / Math.max(image.naturalWidth, image.naturalHeight));
      canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
      canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
      const context = canvas.getContext("2d", { willReadFrequently: true });
      if (!context) throw new Error("Canvas unavailable");
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
      const bins = channels.map(() => new Float64Array(256));
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 3] === 0) continue;
        for (let c = 0; c < 3; c++) bins[c][pixels[i + c]]++;
      }
      // All three channels share a scale so their relative distributions remain visible.
      const peak = Math.max(1, ...bins.flatMap(bin => Array.from(bin)));
      paths.value = bins.map(bin => `M 0 100 ${Array.from(bin, (count, x) =>
        `L ${x} ${(100 - count / peak * 94).toFixed(2)}`).join(" ")} L 255 100 Z`);
      status.value = "";
    } catch {
      status.value = "暂时无法读取直方图";
    }
  };
  image.onerror = () => {
    if (token !== generation) return;
    pendingImage = null;
    status.value = "暂时无法读取直方图";
  };
  image.src = src;
}, { immediate: true });

watch(
  [() => BASE_URL.value, () => props.sessionId, () => props.photoId],
  async ([baseUrl, sessionId, photoId]) => {
    const token = ++metadataGeneration;
    metadataController?.abort();
    metadataController = null;
    metadata.value = null;
    if (!baseUrl || !sessionId || !photoId) return;

    const controller = new AbortController();
    metadataController = controller;
    try {
      const response = await fetch(
        `${baseUrl}/api/enhance/metadata/${encodeURIComponent(sessionId)}/${encodeURIComponent(photoId)}`,
        { signal: controller.signal },
      );
      if (!response.ok) throw new Error("无法读取照片信息");
      const data = await response.json() as Partial<PhotoMetadata>;
      if (token !== metadataGeneration) return;
      metadata.value = {
        focal_length: finiteNumber(data.focal_length),
        lens_focal_range: formatFocalRangeValue(data.lens_focal_range),
        aperture: finiteNumber(data.aperture),
        exposure_time: finiteNumber(data.exposure_time),
      };
    } catch {
      if (token !== metadataGeneration || controller.signal.aborted) return;
      metadata.value = null;
    } finally {
      if (token === metadataGeneration) metadataController = null;
    }
  },
  { immediate: true, flush: "sync" },
);

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatFocalRangeValue(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length !== 2) return null;
  const minimum = finiteNumber(value[0]);
  const maximum = finiteNumber(value[1]);
  return minimum === null || maximum === null ? null : [minimum, maximum];
}

function formatMetadataNumber(value: number | null): string {
  return value === null ? "—" : Number.isInteger(value) ? String(value) : String(Number(value.toFixed(1)));
}

function formatFocalLength(value: number | null): string {
  return value === null ? "—" : `${formatMetadataNumber(value)} mm`;
}

function formatLensFocalRange(value: [number, number] | null): string {
  if (!value) return "—";
  const [minimum, maximum] = value;
  if (minimum === maximum) return `${formatMetadataNumber(minimum)} mm`;
  return `${formatMetadataNumber(minimum)}–${formatMetadataNumber(maximum)} mm`;
}

function formatAperture(value: number | null): string {
  return value === null ? "—" : `f/${formatMetadataNumber(value)}`;
}

function formatExposureTime(value: number | null): string {
  if (value === null || value <= 0) return "—";
  if (value >= 1) return `${formatMetadataNumber(value)} s`;
  const denominator = 1 / value;
  const rounded = Math.round(denominator);
  const denominatorText = Math.abs(denominator - rounded) < 0.05
    ? String(rounded)
    : String(Number(denominator.toFixed(1)));
  return `1/${denominatorText} s`;
}

onBeforeUnmount(() => {
  generation++;
  cancelLoad();
  metadataGeneration++;
  metadataController?.abort();
  metadataController = null;
});
</script>

<template>
  <AdjustmentPanel title="直方图">
    <div class="relative h-[106px] overflow-hidden rounded bg-slate-100 dark:bg-zinc-800">
      <svg v-if="paths.length" viewBox="0 0 255 100" preserveAspectRatio="none" class="h-full w-full" role="img" aria-label="当前预览图的红、绿、蓝三通道叠加直方图">
        <path v-for="(path, index) in paths" :key="channels[index].label" :d="path" :fill="channels[index].color" fill-opacity="0.32" :stroke="channels[index].color" stroke-opacity="0.85" stroke-width="0.8" vector-effect="non-scaling-stroke" />
      </svg>
      <p v-else class="absolute inset-0 flex items-center justify-center text-[11px] text-slate-400 dark:text-zinc-200" role="status">{{ status }}</p>
    </div>
    <div class="mt-2 grid h-9 items-center gap-1 text-[10px] text-slate-600 dark:text-zinc-200" :class="showLensFocalRange ? 'grid-cols-4' : 'grid-cols-3'">
      <div class="min-w-0 whitespace-nowrap text-center" title="焦距">
        <span class="block">焦距</span>
        <span class="block truncate font-medium">{{ formatFocalLength(metadata?.focal_length ?? null) }}</span>
      </div>
      <div v-if="showLensFocalRange" class="min-w-0 whitespace-nowrap text-center" title="镜头焦段">
        <span class="block">镜头焦段</span>
        <span class="block truncate font-medium">{{ formatLensFocalRange(metadata?.lens_focal_range ?? null) }}</span>
      </div>
      <div class="min-w-0 whitespace-nowrap text-center" title="光圈">
        <span class="block">光圈</span>
        <span class="block truncate font-medium">{{ formatAperture(metadata?.aperture ?? null) }}</span>
      </div>
      <div class="min-w-0 whitespace-nowrap text-center" title="快门速度">
        <span class="block">快门</span>
        <span class="block truncate font-medium">{{ formatExposureTime(metadata?.exposure_time ?? null) }}</span>
      </div>
    </div>
  </AdjustmentPanel>
</template>
