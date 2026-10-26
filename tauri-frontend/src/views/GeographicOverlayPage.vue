<script setup lang="ts">
import { computed, ref } from "vue";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  ArrowDownToLine,
  ArrowLeftRight,
  Check,
  ChevronDown,
  CircleHelp,
  FileImage,
  FolderOpen,
  Layers2,
  LoaderCircle,
  Map,
  MapPin,
  Redo2,
  RotateCcw,
  Save,
  Undo2,
} from "lucide-vue-next";
import {
  canRedoGeographic,
  canUndoGeographic,
  commitGeoConfigHistory,
  exportGeographicImage,
  geographicConfig,
  geographicError,
  geographicExporting,
  geographicLocationSource,
  geographicNotice,
  geographicOpening,
  geographicPhoto,
  geographicPreview,
  geographicPreviewing,
  geographicProjectPath,
  geographicSaving,
  geographicWarning,
  hasNativeGeoBridge,
  isGeographicDirty,
  isSupportedGeoPhotoPath,
  loadGeographicProject,
  openGeographicPhoto,
  redoGeoConfig,
  refreshGeographicPreview,
  resetGeoLocationToExif,
  saveGeographicProject,
  undoGeoConfig,
  updateGeoConfig,
  type GeoConfig,
  type GeoFormat,
} from "../stores/geographicOverlay";

type NumericGeoKey = {
  [Key in keyof GeoConfig]: GeoConfig[Key] extends number ? Key : never;
}[keyof GeoConfig];
type ToggleGeoKey = "enabled" | "has_location" | "auto_sky_fade" | "labels" | "north_arrow" | "scale_bar" | "photo_marker" | "adaptive";
type PreviewMode = "effect" | "original" | "split";

const previewMode = ref<PreviewMode>("effect");
const showLayer = ref(false);
const actualPixels = ref(false);
const exportFormat = ref<GeoFormat>("png");
const sourceAndEffect = computed(() => previewMode.value === "split");
const mainImage = computed(() => previewMode.value === "original"
  ? geographicPhoto.value?.preview ?? ""
  : geographicPreview.value?.image ?? geographicPhoto.value?.preview ?? "");
const customColor = computed(() => geographicConfig.value.color_rgb === 0
  ? "#ffffff"
  : `#${geographicConfig.value.color_rgb.toString(16).padStart(6, "0")}`);
const locationSourceLabel = computed(() => {
  if (!geographicConfig.value.has_location) return "未设置位置";
  if (geographicLocationSource.value === "exif") return "来自照片 EXIF";
  return "手动位置";
});
const sourceDimensions = computed(() => geographicPhoto.value
  ? `${geographicPhoto.value.width.toLocaleString()} × ${geographicPhoto.value.height.toLocaleString()} px`
  : "尚未打开照片");
const fileStem = computed(() => {
  const name = geographicPhoto.value?.source_name ?? "photo";
  return name.replace(/\.[^.]+$/, "") || "photo";
});

const rangeControls: Array<{
  key: NumericGeoKey;
  label: string;
  min: number;
  max: number;
  step: number;
  suffix?: string;
}> = [
  { key: "opacity", label: "图层透明度", min: 0, max: 1, step: 0.01 },
  { key: "radius_km", label: "覆盖半径", min: 0.05, max: 20, step: 0.05, suffix: "km" },
  { key: "scale", label: "图层缩放", min: 0.1, max: 4, step: 0.01 },
  { key: "rotation", label: "旋转", min: -180, max: 180, step: 0.5, suffix: "°" },
  { key: "line_width", label: "线宽", min: 0.25, max: 40, step: 0.25 },
  { key: "feather", label: "边缘羽化", min: 0, max: 0.3, step: 0.01 },
  { key: "coverage", label: "最大可用覆盖范围", min: 0.05, max: 1, step: 0.01 },
  { key: "manual_horizon", label: "手动天空保护边界（0=未设置）", min: 0, max: 1, step: 0.01 },
];

function setNumeric(key: NumericGeoKey, event: Event, commit = false) {
  const target = event.target as HTMLInputElement;
  if (!target.value.trim()) return;
  const value = Number(target.value);
  if (!Number.isFinite(value)) return;
  const patch: Partial<GeoConfig> = { [key]: value };
  if (key === "latitude" || key === "longitude") patch.has_location = true;
  updateGeoConfig(patch);
  if (commit) commitGeoConfigHistory();
}

function setToggle(key: ToggleGeoKey, event: Event) {
  const checked = (event.target as HTMLInputElement).checked;
  updateGeoConfig({ [key]: checked } as Partial<GeoConfig>);
  commitGeoConfigHistory();
}

function setSelectNumber(key: "style" | "provider", event: Event) {
  const value = Number((event.target as HTMLSelectElement).value);
  if (!Number.isFinite(value)) return;
  updateGeoConfig({ [key]: value } as Partial<GeoConfig>);
  commitGeoConfigHistory();
}

function setColor(event: Event) {
  const value = (event.target as HTMLInputElement).value;
  updateGeoConfig({ color_rgb: Number.parseInt(value.slice(1), 16) });
}

function setAutoColor() {
  updateGeoConfig({ color_rgb: 0 });
  commitGeoConfigHistory();
}

function setEndpoint(event: Event, commit = false) {
  updateGeoConfig({ overpass_endpoint: (event.target as HTMLInputElement).value });
  if (commit) commitGeoConfigHistory();
}

function displayNumber(value: number, digits = 2) {
  return Number.isFinite(value) ? Number(value.toFixed(digits)).toString() : "—";
}

function getDialogError(cause: unknown, fallback: string): string {
  return cause instanceof Error && cause.message ? cause.message : fallback;
}

async function choosePhoto() {
  geographicError.value = "";
  if (!hasNativeGeoBridge) {
    geographicError.value = "本地文件访问需要 Tauri 桌面版。浏览器运行时不会读取或伪造照片预览。";
    return;
  }
  try {
    const path = await open({
      multiple: false,
      directory: false,
      title: "选择要添加地理叠加的成片",
      filters: [{ name: "支持的成片", extensions: ["jpg", "jpeg", "tif", "tiff", "png", "dng"] }],
    });
    if (typeof path !== "string") return;
    if (!isSupportedGeoPhotoPath(path)) {
      geographicError.value = "只支持 JPG、JPEG、TIF、TIFF、PNG 和 DNG 图像。";
      return;
    }
    await openGeographicPhoto(path);
    previewMode.value = "effect";
    showLayer.value = false;
  } catch (cause) {
    geographicError.value = getDialogError(cause, "无法打开文件选择器。请在 Tauri 桌面版中重试。");
  }
}

async function chooseProject() {
  geographicError.value = "";
  if (!hasNativeGeoBridge) {
    geographicError.value = "加载地理项目需要 Tauri 桌面版。";
    return;
  }
  try {
    const path = await open({
      multiple: false,
      directory: false,
      title: "打开地理叠加项目",
      filters: [{ name: "Imprint 地理项目", extensions: ["json"] }],
    });
    if (typeof path !== "string") return;
    const loaded = await loadGeographicProject(path);
    if (loaded) {
      previewMode.value = "effect";
      showLayer.value = false;
    }
  } catch (cause) {
    geographicError.value = getDialogError(cause, "无法打开地理项目文件。");
  }
}

async function saveProject() {
  geographicError.value = "";
  if (!hasNativeGeoBridge) {
    geographicError.value = "保存地理项目需要 Tauri 桌面版。";
    return;
  }
  try {
    let path = geographicProjectPath.value;
    if (!path) {
      const selected = await save({
        title: "保存地理叠加项目",
        defaultPath: `${fileStem.value}.imprint-geo.json`,
        filters: [{ name: "Imprint 地理项目", extensions: ["json"] }],
      });
      if (typeof selected !== "string") return;
      path = selected;
    }
    await saveGeographicProject(path);
  } catch (cause) {
    geographicError.value = getDialogError(cause, "保存地理项目失败。请检查文件权限后重试。");
  }
}

async function exportImage(overlayOnly: boolean) {
  geographicError.value = "";
  if (!hasNativeGeoBridge) {
    geographicError.value = "导出需要 Tauri 桌面版原生桥。";
    return;
  }
  const format: GeoFormat = overlayOnly ? "png" : exportFormat.value;
  const extension = overlayOnly ? "png" : format === "jpeg" ? "jpg" : "png";
  try {
    const path = await save({
      title: overlayOnly ? "导出透明地理图层" : "导出全尺寸地理叠加成片",
      defaultPath: `${fileStem.value}_geo.${extension}`,
      filters: [{ name: overlayOnly ? "透明 PNG 图层" : `${format.toUpperCase()} 图像`, extensions: [extension] }],
    });
    if (typeof path !== "string") return;
    await exportGeographicImage(path, format, overlayOnly);
  } catch (cause) {
    geographicError.value = getDialogError(cause, "导出失败。请检查输出目录是否可写。");
  }
}
</script>

<template>
  <div data-testid="geographic-overlay-page" class="flex h-full min-h-0 w-full overflow-hidden bg-slate-50 text-slate-800 dark:bg-zinc-950 dark:text-zinc-100">
    <section class="flex min-w-0 flex-1 flex-col overflow-hidden">
      <header class="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-5 py-3 dark:border-zinc-800 dark:bg-zinc-900">
        <div class="flex min-w-0 items-center gap-3">
          <div class="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-100 text-indigo-700 dark:bg-indigo-950/70 dark:text-indigo-300">
            <Map class="h-5 w-5" />
          </div>
          <div class="min-w-0">
            <h2 class="truncate text-sm font-semibold">地理叠加 · 后期图层</h2>
            <p class="truncate text-[11px] text-slate-500 dark:text-zinc-400">为完成的照片添加可编辑地图覆盖，不改动原图</p>
          </div>
        </div>
        <div class="flex flex-wrap items-center gap-1.5">
          <button type="button" @click="choosePhoto" :disabled="geographicOpening || geographicSaving || geographicExporting" class="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700">
            <FolderOpen class="h-3.5 w-3.5" />
            选择成片
          </button>
          <button type="button" @click="chooseProject" :disabled="geographicOpening || geographicSaving || geographicExporting" class="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700">
            <FileImage class="h-3.5 w-3.5" />
            打开项目
          </button>
          <button type="button" @click="undoGeoConfig" :disabled="!canUndoGeographic" title="撤销" aria-label="撤销" class="rounded-lg border border-slate-200 bg-white p-2 text-slate-600 transition hover:bg-slate-50 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
            <Undo2 class="h-3.5 w-3.5" />
          </button>
          <button type="button" @click="redoGeoConfig" :disabled="!canRedoGeographic" title="重做" aria-label="重做" class="rounded-lg border border-slate-200 bg-white p-2 text-slate-600 transition hover:bg-slate-50 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-700">
            <Redo2 class="h-3.5 w-3.5" />
          </button>
          <button type="button" @click="saveProject" :disabled="!geographicPhoto || geographicSaving || geographicExporting || geographicOpening" class="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700">
            <LoaderCircle v-if="geographicSaving" class="h-3.5 w-3.5 animate-spin" />
            <Save v-else class="h-3.5 w-3.5" />
            {{ geographicProjectPath ? "保存项目" : "保存项目…" }}
          </button>
        </div>
      </header>

      <div v-if="geographicPhoto" class="flex shrink-0 flex-wrap items-center justify-between gap-2 border-b border-slate-200 bg-slate-100/70 px-5 py-2 dark:border-zinc-800 dark:bg-zinc-900/70">
        <div class="flex min-w-0 items-center gap-2 text-xs">
          <span class="truncate font-medium text-slate-800 dark:text-zinc-200">{{ geographicPhoto.source_name }}</span>
          <span class="shrink-0 text-slate-400">{{ sourceDimensions }}</span>
          <span v-if="isGeographicDirty" class="shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-medium text-amber-800 dark:bg-amber-950/50 dark:text-amber-300">未保存</span>
          <span v-else-if="geographicProjectPath" class="shrink-0 rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] font-medium text-emerald-700 dark:bg-emerald-950/50 dark:text-emerald-300">项目已保存</span>
        </div>
        <div class="flex shrink-0 items-center gap-2">
          <span class="rounded-full border border-amber-300 bg-amber-50 px-2 py-1 text-[10px] font-semibold text-amber-800 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300">{{ geographicConfig.provider === 1 ? "OpenStreetMap / Overpass" : "Mock 示例数据" }}</span>
          <span class="hidden text-[10px] text-slate-500 sm:inline dark:text-zinc-400">仅照片后期图层；不会写入 RAW 或 ACR 参数</span>
        </div>
      </div>

      <div class="flex min-h-0 flex-1 flex-col overflow-auto p-4 sm:p-5">
        <div v-if="geographicError" role="alert" class="mb-3 flex shrink-0 items-start gap-2 rounded-xl border border-rose-200 bg-rose-50 px-3.5 py-3 text-xs text-rose-800 dark:border-rose-900/70 dark:bg-rose-950/30 dark:text-rose-200">
          <CircleHelp class="mt-0.5 h-4 w-4 shrink-0" />
          <span class="min-w-0 whitespace-pre-wrap break-words">{{ geographicError }}</span>
        </div>
        <div v-if="geographicNotice" role="status" class="mb-3 flex shrink-0 items-start gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-3.5 py-3 text-xs text-emerald-800 dark:border-emerald-900/70 dark:bg-emerald-950/30 dark:text-emerald-200">
          <Check class="mt-0.5 h-4 w-4 shrink-0" />
          <span class="min-w-0 whitespace-pre-wrap break-words">{{ geographicNotice }}</span>
        </div>
        <div v-if="!hasNativeGeoBridge" class="mb-3 flex shrink-0 items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 px-3.5 py-3 text-xs text-amber-900 dark:border-amber-900/70 dark:bg-amber-950/30 dark:text-amber-200">
          <CircleHelp class="mt-0.5 h-4 w-4 shrink-0" />
          <span>当前运行于浏览器。需要 Tauri 原生桥后才能打开本地照片、生成地图叠加和导出；本页不会绘制假地图。</span>
        </div>

        <div class="flex shrink-0 flex-wrap items-center justify-between gap-3 pb-3">
          <div class="flex items-center gap-1 rounded-lg border border-slate-200 bg-white p-1 dark:border-zinc-800 dark:bg-zinc-900">
            <button type="button" @click="previewMode = 'effect'" :disabled="!geographicPhoto" :class="previewMode === 'effect' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 dark:text-zinc-300 dark:hover:bg-zinc-800'" class="rounded-md px-2.5 py-1.5 text-[11px] font-medium transition disabled:opacity-40">效果</button>
            <button type="button" @click="previewMode = 'original'" :disabled="!geographicPhoto" :class="previewMode === 'original' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 dark:text-zinc-300 dark:hover:bg-zinc-800'" class="rounded-md px-2.5 py-1.5 text-[11px] font-medium transition disabled:opacity-40">原图</button>
            <button type="button" @click="previewMode = 'split'" :disabled="!geographicPhoto" :class="previewMode === 'split' ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100 dark:text-zinc-300 dark:hover:bg-zinc-800'" class="inline-flex items-center gap-1 rounded-md px-2.5 py-1.5 text-[11px] font-medium transition disabled:opacity-40">
              <ArrowLeftRight class="h-3 w-3" /> 对比
            </button>
          </div>
          <div class="flex items-center gap-2">
            <span v-if="geographicPreviewing" class="inline-flex items-center gap-1 text-[11px] text-indigo-600 dark:text-indigo-300"><LoaderCircle class="h-3 w-3 animate-spin" />正在更新预览</span>
            <button type="button" @click="refreshGeographicPreview" :disabled="!geographicPhoto || geographicPreviewing" class="rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-[11px] font-medium dark:border-zinc-700 dark:bg-zinc-800 disabled:opacity-40">{{ geographicPreviewing ? "正在生成预览…" : "刷新地图" }}</button>
            <button type="button" @click="actualPixels = !actualPixels" :disabled="!geographicPhoto || sourceAndEffect" :aria-pressed="actualPixels" class="rounded-lg border border-slate-200 bg-white px-2.5 py-2 text-[11px] font-medium dark:border-zinc-700 dark:bg-zinc-800 disabled:opacity-40">{{ actualPixels ? "适应窗口" : "100% 原尺寸" }}</button>
            <button type="button" @click="showLayer = !showLayer" :disabled="!geographicPreview?.overlay" class="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-[11px] font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800">
              <Layers2 class="h-3.5 w-3.5" />
              {{ showLayer ? "隐藏透明层" : "查看透明层" }}
            </button>
          </div>
        </div>

        <div v-if="geographicPhoto" class="flex min-h-[320px] flex-1 flex-col overflow-hidden rounded-2xl border border-slate-200 bg-zinc-900 shadow-sm dark:border-zinc-800">
          <div class="flex shrink-0 items-center justify-between border-b border-white/10 px-3.5 py-2 text-[10px] text-zinc-300">
            <span>{{ sourceAndEffect ? "原图 / 原生渲染效果" : previewMode === "original" ? "原始成片" : "原生渲染效果" }}</span>
            <span>{{ geographicPreview?.width && geographicPreview?.height ? `${geographicPreview.width} × ${geographicPreview.height}` : sourceDimensions }}</span>
          </div>
          <div v-if="sourceAndEffect" class="grid min-h-0 flex-1 grid-cols-1 gap-px overflow-auto bg-white/10 md:grid-cols-2">
            <figure class="flex min-h-[260px] flex-col items-center justify-center bg-zinc-950 p-3">
              <figcaption class="mb-2 self-start text-[10px] text-zinc-400">原图</figcaption>
              <img :src="geographicPhoto.preview" :alt="`${geographicPhoto.source_name} 原图`" class="max-h-[65vh] max-w-full object-contain" />
            </figure>
            <figure class="flex min-h-[260px] flex-col items-center justify-center bg-zinc-950 p-3">
              <figcaption class="mb-2 self-start text-[10px] text-zinc-400">效果</figcaption>
              <img v-if="geographicPreview?.image" :src="geographicPreview.image" :alt="`${geographicPhoto.source_name} 地理叠加效果`" class="max-h-[65vh] max-w-full object-contain" />
              <span v-else class="px-5 text-center text-xs text-zinc-400">等待原生预览返回</span>
            </figure>
          </div>
          <div v-else class="relative flex min-h-0 flex-1 overflow-auto bg-zinc-950 p-4" :class="actualPixels ? 'items-start justify-start' : 'items-center justify-center'">
            <img :src="mainImage" :alt="previewMode === 'original' ? `${geographicPhoto.source_name} 原图` : `${geographicPhoto.source_name} 地理叠加效果`" :class="actualPixels ? 'max-h-none max-w-none shrink-0' : 'max-h-[72vh] max-w-full object-contain'" />
            <span v-if="previewMode === 'effect' && !geographicPreview?.image" class="absolute rounded-lg bg-black/60 px-3 py-2 text-xs text-zinc-200">{{ geographicPreviewing ? "正在生成原生预览…" : "等待原生预览返回" }}</span>
          </div>
          <div v-if="showLayer && geographicPreview?.overlay" class="border-t border-white/10 bg-zinc-900 p-3">
            <div class="mb-2 flex items-center gap-1.5 text-[10px] font-medium text-zinc-300"><Layers2 class="h-3 w-3" />透明图层（棋盘格用于检查透明区域）</div>
            <div class="flex max-h-52 min-h-28 items-center justify-center overflow-hidden rounded-lg border border-white/10 p-2" style="background-image: conic-gradient(#333 25%, #444 0 50%, #333 0 75%, #444 0); background-size: 20px 20px;">
              <img :src="geographicPreview.overlay" alt="原生返回的透明地理图层" class="max-h-48 max-w-full object-contain" />
            </div>
          </div>
          <div v-if="geographicConfig.provider === 1 && geographicPreview?.attribution" class="flex flex-wrap justify-between gap-2 border-t border-white/10 px-3.5 py-2 text-[10px] text-zinc-300">
            <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer" class="underline">{{ geographicPreview.attribution }}</a>
            <span v-if="geographicConfig.enabled && (geographicPreview.feature_count ?? 0) > 0">{{ geographicPreview.feature_count }} 个地理要素 · {{ geographicPreview.cache_hit ? "本地缓存" : "已查询" }}</span>
            <span v-else>尚未请求地图数据</span>
          </div>
          <div v-if="geographicPreview?.render_backend" class="px-3.5 py-1 text-[10px] text-zinc-400">{{ geographicPreview.render_backend }} · 原尺寸预览</div>
          <div v-if="geographicWarning" class="border-t border-amber-500/20 bg-amber-950/30 px-3.5 py-2 text-[10px] text-amber-200">{{ geographicWarning }}</div>
        </div>

        <div v-else class="flex min-h-[320px] flex-1 flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-white/70 p-8 text-center dark:border-zinc-700 dark:bg-zinc-900/50">
          <div class="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-100 text-indigo-600 dark:bg-indigo-950/60 dark:text-indigo-300"><MapPin class="h-7 w-7" /></div>
          <h3 class="text-sm font-semibold">从完成的照片开始</h3>
          <p class="mt-2 max-w-md text-xs leading-5 text-slate-500 dark:text-zinc-400">选择 JPG、JPEG、TIF、TIFF、PNG 或 DNG。此页面用于照片后期添加地图图层，不修改 RAW 原片，也不会改动 ACR 可编辑参数。</p>
          <button type="button" @click="choosePhoto" class="mt-5 inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-xs font-semibold text-white shadow-sm transition hover:bg-indigo-700"><FolderOpen class="h-4 w-4" />选择成片</button>
        </div>

        <p class="mt-2 shrink-0 text-[10px] leading-4 text-slate-500 dark:text-zinc-500">预览图与透明覆盖层由 Tauri 原生渲染桥返回。导出会重新按原始全尺寸处理，并生成新文件；原图保持不变。</p>
      </div>
    </section>

    <aside class="flex w-[330px] shrink-0 flex-col border-l border-slate-200 bg-white dark:border-zinc-800 dark:bg-zinc-900 xl:w-[370px]">
      <div class="flex shrink-0 items-center justify-between border-b border-slate-200 px-4 py-3 dark:border-zinc-800">
        <div>
          <h3 class="text-xs font-semibold">地图叠加设置</h3>
          <p class="mt-0.5 text-[10px] text-slate-500 dark:text-zinc-400">{{ locationSourceLabel }}</p>
        </div>
        <button type="button" @click="resetGeoLocationToExif" :disabled="!geographicPhoto" title="重置到照片 EXIF 坐标" class="inline-flex items-center gap-1 rounded-lg px-2 py-1.5 text-[10px] font-medium text-slate-600 transition hover:bg-slate-100 disabled:opacity-40 dark:text-zinc-300 dark:hover:bg-zinc-800"><RotateCcw class="h-3 w-3" />重置 EXIF</button>
      </div>

      <div class="geo-scroll-area min-h-0 flex-1 p-3.5">
        <label class="mb-3 flex items-center justify-between rounded-xl border border-indigo-100 bg-indigo-50/70 px-3 py-2.5 dark:border-indigo-900/50 dark:bg-indigo-950/30">
          <span><span class="block text-xs font-semibold">启用地图叠加</span><span class="mt-0.5 block text-[10px] text-slate-500 dark:text-zinc-400">调整与透明层导出可单独预览</span></span>
          <input type="checkbox" :checked="geographicConfig.enabled" @change="setToggle('enabled', $event)" class="h-4 w-4 accent-indigo-600" />
        </label>

        <details open class="mb-2.5 rounded-xl border border-slate-200 dark:border-zinc-800">
          <summary class="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-xs font-semibold"><span class="flex items-center gap-2"><MapPin class="h-3.5 w-3.5 text-indigo-500" />位置</span><ChevronDown class="h-3.5 w-3.5 text-slate-400" /></summary>
          <div class="space-y-2.5 border-t border-slate-100 px-3 py-3 dark:border-zinc-800">
            <label class="flex items-center justify-between text-[11px] text-slate-600 dark:text-zinc-300"><span>使用位置坐标</span><input type="checkbox" :checked="geographicConfig.has_location" @change="setToggle('has_location', $event)" class="h-3.5 w-3.5 accent-indigo-600" /></label>
            <div class="grid grid-cols-2 gap-2">
              <label class="text-[10px] text-slate-500 dark:text-zinc-400">纬度
                <input type="number" min="-90" max="90" step="0.000001" :value="geographicConfig.latitude" @input="setNumeric('latitude', $event)" @change="commitGeoConfigHistory" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-800 outline-none focus:border-indigo-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100" />
              </label>
              <label class="text-[10px] text-slate-500 dark:text-zinc-400">经度
                <input type="number" min="-180" max="180" step="0.000001" :value="geographicConfig.longitude" @input="setNumeric('longitude', $event)" @change="commitGeoConfigHistory" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-800 outline-none focus:border-indigo-400 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100" />
              </label>
            </div>
            <p class="text-[10px] leading-4 text-slate-500 dark:text-zinc-500">(0, 0) 是有效坐标；是否设置由“使用位置坐标”单独控制。</p>
          </div>
        </details>

        <details open class="mb-2.5 rounded-xl border border-slate-200 dark:border-zinc-800">
          <summary class="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-xs font-semibold"><span class="flex items-center gap-2"><Map class="h-3.5 w-3.5 text-indigo-500" />地图来源与样式</span><ChevronDown class="h-3.5 w-3.5 text-slate-400" /></summary>
          <div class="space-y-2.5 border-t border-slate-100 px-3 py-3 dark:border-zinc-800">
            <label class="block text-[10px] text-slate-500 dark:text-zinc-400">数据来源
              <select :value="geographicConfig.provider" @change="setSelectNumber('provider', $event)" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-800 outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100">
                <option :value="0">Mock 示例数据</option>
                <option :value="1">OSM / Overpass（真实地图）</option>
              </select>
            </label>
            <label class="block text-[10px] text-slate-500 dark:text-zinc-400">叠加风格
              <select :value="geographicConfig.style" @change="setSelectNumber('style', $event)" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-800 outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100">
                <option :value="0">自动</option><option :value="1">城市</option><option :value="2">海岸</option><option :value="3">Terrain 地形（未实现）</option>
              </select>
            </label>
            <label v-if="geographicConfig.provider === 1" class="block text-[10px] text-slate-500 dark:text-zinc-400">Overpass 地址
              <input type="url" :value="geographicConfig.overpass_endpoint" @input="setEndpoint($event)" @change="commitGeoConfigHistory" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-[11px] text-slate-800 outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100" />
            </label>
            <p class="rounded-lg bg-amber-50 px-2.5 py-2 text-[10px] leading-4 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">{{ geographicConfig.provider === 1 ? "查询仅发送经纬度和半径，不上传照片；地图数据会本地缓存。公共实例不需要 API Key，查询半径上限 20 km。" : "Mock 是测试几何，不代表真实地图。" }}</p>
          </div>
        </details>

        <details open class="mb-2.5 rounded-xl border border-slate-200 dark:border-zinc-800">
          <summary class="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-xs font-semibold"><span class="flex items-center gap-2"><Layers2 class="h-3.5 w-3.5 text-indigo-500" />图层变换与外观</span><ChevronDown class="h-3.5 w-3.5 text-slate-400" /></summary>
          <div class="space-y-3 border-t border-slate-100 px-3 py-3 dark:border-zinc-800">
            <label v-for="control in rangeControls" :key="control.key" class="block">
              <span class="mb-1 flex items-center justify-between text-[10px] text-slate-600 dark:text-zinc-300"><span>{{ control.label }}</span><span class="tabular-nums text-slate-500 dark:text-zinc-400">{{ displayNumber(geographicConfig[control.key], control.step < 1 ? 2 : 1) }}{{ control.suffix ? ` ${control.suffix}` : "" }}</span></span>
              <input type="range" :min="control.min" :max="control.max" :step="control.step" :value="geographicConfig[control.key]" @input="setNumeric(control.key, $event)" @change="commitGeoConfigHistory" class="w-full accent-indigo-600" />
            </label>
            <div class="grid grid-cols-2 gap-2">
              <label class="text-[10px] text-slate-500 dark:text-zinc-400">偏移 X
                <input type="number" min="-1" max="1" step="0.01" :value="geographicConfig.offset_x" @input="setNumeric('offset_x', $event)" @change="commitGeoConfigHistory" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-800 outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100" />
              </label>
              <label class="text-[10px] text-slate-500 dark:text-zinc-400">偏移 Y
                <input type="number" min="-1" max="1" step="0.01" :value="geographicConfig.offset_y" @input="setNumeric('offset_y', $event)" @change="commitGeoConfigHistory" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-800 outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100" />
              </label>
            </div>
            <div class="flex items-center justify-between rounded-lg border border-slate-200 px-2.5 py-2 dark:border-zinc-700">
              <span class="text-[10px] text-slate-600 dark:text-zinc-300">图层颜色</span>
              <div class="flex items-center gap-2">
                <button type="button" @click="setAutoColor" :class="geographicConfig.color_rgb === 0 ? 'border-indigo-500 bg-indigo-50 text-indigo-700 dark:bg-indigo-950/50 dark:text-indigo-300' : 'border-slate-200 text-slate-500 dark:border-zinc-700 dark:text-zinc-400'" class="rounded-md border px-2 py-1 text-[10px]">自动</button>
                <input type="color" :value="customColor" @input="setColor" @change="commitGeoConfigHistory" title="选择图层颜色" class="h-6 w-8 cursor-pointer rounded border-0 bg-transparent p-0" />
              </div>
            </div>
            <div class="space-y-2 border-t border-slate-100 pt-2 dark:border-zinc-800">
              <label v-for="toggle in [
                { key: 'auto_sky_fade', label: '自动天空渐隐' },
                { key: 'labels', label: '显示地名标签' },
                { key: 'north_arrow', label: '显示指北针' },
                { key: 'scale_bar', label: '显示比例尺' },
                { key: 'photo_marker', label: '显示照片位置标记' },
                { key: 'adaptive', label: '自适应照片构图' },
              ]" :key="toggle.key" class="flex items-center justify-between text-[10px] text-slate-600 dark:text-zinc-300">
                <span>{{ toggle.label }}</span>
                <input type="checkbox" :checked="geographicConfig[toggle.key as ToggleGeoKey]" @change="setToggle(toggle.key as ToggleGeoKey, $event)" class="h-3.5 w-3.5 accent-indigo-600" />
              </label>
            </div>
          </div>
        </details>

        <details class="mb-2.5 rounded-xl border border-slate-200 dark:border-zinc-800">
          <summary class="flex cursor-pointer list-none items-center justify-between px-3 py-2.5 text-xs font-semibold"><span class="flex items-center gap-2"><ArrowDownToLine class="h-3.5 w-3.5 text-indigo-500" />导出</span><ChevronDown class="h-3.5 w-3.5 text-slate-400" /></summary>
          <div class="space-y-2.5 border-t border-slate-100 px-3 py-3 dark:border-zinc-800">
            <label class="block text-[10px] text-slate-500 dark:text-zinc-400">合成图格式
              <select v-model="exportFormat" class="mt-1 w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-800 outline-none dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-100"><option value="png">PNG</option><option value="jpeg">JPEG</option></select>
            </label>
            <button type="button" @click="exportImage(false)" :disabled="!geographicPhoto || geographicExporting || geographicSaving || geographicOpening" class="flex w-full items-center justify-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-indigo-700 disabled:opacity-40"><LoaderCircle v-if="geographicExporting" class="h-3.5 w-3.5 animate-spin" /><ArrowDownToLine v-else class="h-3.5 w-3.5" />导出全尺寸合成图</button>
            <button type="button" @click="exportImage(true)" :disabled="!geographicPhoto || geographicExporting || geographicSaving || geographicOpening" class="flex w-full items-center justify-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 transition hover:bg-slate-50 disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"><Layers2 class="h-3.5 w-3.5" />导出透明 PNG 图层</button>
            <p class="text-[10px] leading-4 text-slate-500 dark:text-zinc-500">导出由原生端重新读取全尺寸照片。输出为新文件，原图不会被覆盖。</p>
          </div>
        </details>
      </div>
    </aside>
  </div>
</template>

<style scoped>
.geo-scroll-area {
  overflow-y: scroll;
  scrollbar-gutter: stable;
}

@supports not (scrollbar-gutter: stable) {
  .geo-scroll-area {
    padding-right: 1.25rem;
  }
}

details > summary::-webkit-details-marker {
  display: none;
}
</style>
