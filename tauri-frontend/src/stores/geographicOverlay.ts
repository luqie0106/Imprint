import { computed, ref } from "vue";
import { invoke, isTauri } from "@tauri-apps/api/core";

export type GeoStyle = 0 | 1 | 2 | 3;
export type GeoProvider = 0 | 1;
export type GeoFormat = "png" | "jpeg";
export type GeoLocationSource = "exif" | "manual" | "none";

export interface GeoLocation {
  latitude: number;
  longitude: number;
}

export interface GeoPhoto {
  id: string;
  source_name: string;
  width: number;
  height: number;
  preview: string;
  exif_location: GeoLocation | null;
}

export interface GeoConfig {
  latitude: number;
  longitude: number;
  opacity: number;
  radius_km: number;
  offset_x: number;
  offset_y: number;
  scale: number;
  rotation: number;
  line_width: number;
  feather: number;
  coverage: number;
  manual_horizon: number;
  color_rgb: number;
  enabled: boolean;
  has_location: boolean;
  style: GeoStyle;
  auto_sky_fade: boolean;
  labels: boolean;
  north_arrow: boolean;
  scale_bar: boolean;
  photo_marker: boolean;
  adaptive: boolean;
  provider: GeoProvider;
  overpass_endpoint: string;
}

export interface GeoPreview {
  image: string;
  overlay: string;
  width: number;
  height: number;
  warning: string;
}

interface GeoHistoryEntry {
  config: GeoConfig;
  locationSource: GeoLocationSource;
}

interface GeoProjectResult {
  photo: GeoPhoto;
  config: GeoConfig;
}

const defaultOverpassEndpoint = "https://overpass-api.de/api/interpreter";
const allowedImageExtensions = new Set(["jpg", "jpeg", "tif", "tiff", "png", "dng"]);

export const geographicPhoto = ref<GeoPhoto | null>(null);
export const geographicConfig = ref<GeoConfig>(createDefaultGeoConfig());
export const geographicPreview = ref<GeoPreview | null>(null);
export const geographicLocationSource = ref<GeoLocationSource>("none");
export const geographicProjectPath = ref("");
export const geographicError = ref("");
export const geographicNotice = ref("");
const historyEntries: GeoHistoryEntry[] = [];
const historyIndex = ref(0);
const historyLength = ref(1);
const savedConfigJson = ref(JSON.stringify(geographicConfig.value));
export const geographicWarning = computed(() => geographicPreview.value?.warning ?? "");
export const geographicOpening = ref(false);
export const geographicPreviewing = ref(false);
export const geographicSaving = ref(false);
export const geographicExporting = ref(false);
export const isGeographicDirty = computed(() => JSON.stringify(geographicConfig.value) !== savedConfigJson.value);
export const canUndoGeographic = computed(() => historyIndex.value > 0);
export const canRedoGeographic = computed(() => historyIndex.value < historyLength.value - 1);
export const hasNativeGeoBridge = isTauri();

function createDefaultGeoConfig(): GeoConfig {
  return {
    latitude: 0,
    longitude: 0,
    opacity: 0.18,
    radius_km: 3,
    offset_x: 0,
    offset_y: 0,
    scale: 1,
    rotation: 0,
    line_width: 1,
    feather: 0.04,
    coverage: 0.8,
    manual_horizon: 0,
    color_rgb: 0,
    enabled: false,
    has_location: false,
    style: 0,
    auto_sky_fade: true,
    labels: true,
    north_arrow: true,
    scale_bar: true,
    photo_marker: true,
    adaptive: true,
    provider: 0,
    overpass_endpoint: defaultOverpassEndpoint,
  };
}

function cloneConfig(config: GeoConfig): GeoConfig {
  return { ...config };
}

function normalizeConfig(config: Partial<GeoConfig> | null | undefined): GeoConfig {
  return { ...createDefaultGeoConfig(), ...config };
}

function inferLocationSource(photo: GeoPhoto, config: GeoConfig): GeoLocationSource {
  if (!config.has_location) return "none";
  const exif = photo.exif_location;
  if (exif && Math.abs(exif.latitude - config.latitude) < 1e-7 && Math.abs(exif.longitude - config.longitude) < 1e-7)
    return "exif";
  return "manual";
}

function isSupportedPhotoPath(path: string): boolean {
  const extension = path.split(".").pop()?.toLowerCase() ?? "";
  return allowedImageExtensions.has(extension);
}

function getErrorMessage(cause: unknown, fallback: string): string {
  if (cause instanceof Error && cause.message.trim()) return cause.message;
  if (typeof cause === "string" && cause.trim()) return cause;
  return fallback;
}

let openGeneration = 0;
let previewGeneration = 0;
let previewTimer: number | undefined;

function snapshotHistoryEntry(): GeoHistoryEntry {
  return { config: cloneConfig(geographicConfig.value), locationSource: geographicLocationSource.value };
}

function resetHistory(config: GeoConfig, locationSource: GeoLocationSource, setSavedBaseline = true) {
  geographicConfig.value = cloneConfig(config);
  geographicLocationSource.value = locationSource;
  historyEntries.splice(0, historyEntries.length, { config: cloneConfig(config), locationSource });
  historyLength.value = historyEntries.length;
  historyIndex.value = 0;
  if (setSavedBaseline) savedConfigJson.value = JSON.stringify(config);
}

export function isSupportedGeoPhotoPath(path: string): boolean {
  return isSupportedPhotoPath(path);
}

export function updateGeoConfig(patch: Partial<GeoConfig>, commitHistory = false) {
  const next = { ...geographicConfig.value, ...patch };
  if (JSON.stringify(next) === JSON.stringify(geographicConfig.value)) return;
  geographicConfig.value = next;
  if (Object.prototype.hasOwnProperty.call(patch, "latitude") || Object.prototype.hasOwnProperty.call(patch, "longitude")) {
    geographicLocationSource.value = "manual";
  }
  if (Object.prototype.hasOwnProperty.call(patch, "has_location")) {
    if (patch.has_location) geographicLocationSource.value = "manual";
    else geographicLocationSource.value = "none";
  }
  geographicError.value = "";
  geographicNotice.value = "";
  const unavailableSource = unsupportedSourceMessage(next);
  if (unavailableSource) geographicError.value = unavailableSource;
  if (commitHistory) commitGeoConfigHistory();
  scheduleGeoPreview();
}

export function commitGeoConfigHistory() {
  const current = snapshotHistoryEntry();
  const active = historyEntries[historyIndex.value];
  if (active && JSON.stringify(active) === JSON.stringify(current)) return;
  historyEntries.splice(historyIndex.value + 1);
  historyEntries.push(current);
  if (historyEntries.length > 100) historyEntries.shift();
  historyLength.value = historyEntries.length;
  historyIndex.value = historyEntries.length - 1;
}

function restoreHistory(index: number) {
  const entry = historyEntries[index];
  if (!entry) return;
  historyIndex.value = index;
  geographicConfig.value = cloneConfig(entry.config);
  geographicLocationSource.value = entry.locationSource;
  geographicError.value = "";
  geographicNotice.value = "";
  scheduleGeoPreview();
}

export function undoGeoConfig() {
  if (historyIndex.value > 0) restoreHistory(historyIndex.value - 1);
}

export function redoGeoConfig() {
  if (historyIndex.value < historyEntries.length - 1) restoreHistory(historyIndex.value + 1);
}

export function resetGeoLocationToExif() {
  const exif = geographicPhoto.value?.exif_location;
  updateGeoConfig({
    latitude: exif?.latitude ?? 0,
    longitude: exif?.longitude ?? 0,
    has_location: exif !== null && exif !== undefined,
  });
  geographicLocationSource.value = exif ? "exif" : "none";
  commitGeoConfigHistory();
}

function validateGeoConfig(config: GeoConfig): string {
  if (config.has_location && (!Number.isFinite(config.latitude) || config.latitude < -90 || config.latitude > 90))
    return "纬度必须在 -90 到 90 之间。";
  if (config.has_location && (!Number.isFinite(config.longitude) || config.longitude < -180 || config.longitude > 180))
    return "经度必须在 -180 到 180 之间。";
  const normalizedValues: Array<[number, number, number, string]> = [
    [config.opacity, 0, 1, "透明度"],
    [config.radius_km, 0.05, 1000, "覆盖半径"],
    [config.scale, 0.1, 4, "缩放"],
    [config.line_width, 0.25, 40, "线宽"],
    [config.feather, 0, 0.3, "羽化"],
    [config.coverage, 0.05, 1, "覆盖范围"],
    [config.manual_horizon, 0, 1, "天空保护边界"],
  ];
  const outOfRange = normalizedValues.find(([value, min, max]) => !Number.isFinite(value) || value < min || value > max);
  return outOfRange ? `${outOfRange[3]}超出可用范围。` : "";
}

function unsupportedSourceMessage(config: GeoConfig): string {
  if (config.provider === 1) return "OSM / Overpass 数据源尚未实现。当前只能使用明确标记的 Mock 示例数据。";
  if (config.style === 3) return "Terrain 地形图层尚未实现。请选择自动、城市或海岸样式。";
  return "";
}

function scheduleGeoPreview() {
  const photo = geographicPhoto.value;
  if (!photo) return;
  window.clearTimeout(previewTimer);
  const requestGeneration = ++previewGeneration;
  const config = cloneConfig(geographicConfig.value);
  const unsupported = unsupportedSourceMessage(config);
  if (unsupported) {
    geographicPreview.value = null;
    geographicPreviewing.value = false;
    geographicError.value = unsupported;
    return;
  }
  const validationError = validateGeoConfig(config);
  if (validationError) {
    geographicPreview.value = null;
    geographicPreviewing.value = false;
    geographicError.value = validationError;
    return;
  }
  geographicPreviewing.value = true;
  previewTimer = window.setTimeout(async () => {
    if (!hasNativeGeoBridge) {
      if (requestGeneration !== previewGeneration) return;
      geographicPreviewing.value = false;
      geographicPreview.value = null;
      geographicError.value = "地理叠加需要 Tauri 桌面版原生桥；浏览器预览不会绘制模拟地图。";
      return;
    }
    try {
      const result = await invoke<GeoPreview>("geo_preview", { id: photo.id, config });
      if (requestGeneration !== previewGeneration || geographicPhoto.value?.id !== photo.id) return;
      geographicPreview.value = result;
      geographicError.value = "";
    } catch (cause) {
      if (requestGeneration !== previewGeneration || geographicPhoto.value?.id !== photo.id) return;
      geographicPreview.value = null;
      geographicError.value = getErrorMessage(cause, "原生地理叠加预览失败。请确认桌面桥已安装并可用。");
    } finally {
      if (requestGeneration === previewGeneration) geographicPreviewing.value = false;
    }
  }, 260);
}

export async function openGeographicPhoto(path: string): Promise<boolean> {
  geographicError.value = "";
  geographicNotice.value = "";
  if (geographicSaving.value || geographicExporting.value) {
    geographicError.value = "请等待当前保存或导出完成后再打开照片。";
    return false;
  }
  if (!isSupportedPhotoPath(path)) {
    geographicError.value = "只支持 JPG、JPEG、TIF、TIFF、PNG 和 DNG 图像。";
    return false;
  }
  if (!hasNativeGeoBridge) {
    geographicError.value = "打开照片需要 Tauri 桌面版原生桥；浏览器模式不能读取本地文件。";
    return false;
  }
  const requestGeneration = ++openGeneration;
  previewGeneration += 1;
  window.clearTimeout(previewTimer);
  geographicPreviewing.value = false;
  geographicOpening.value = true;
  geographicPreview.value = null;
  try {
    const opened = await invoke<GeoPhoto>("geo_open", { path });
    if (requestGeneration !== openGeneration) return false;
    geographicPhoto.value = opened;
    geographicProjectPath.value = "";
    const initial = createDefaultGeoConfig();
    const exif = opened.exif_location;
    if (exif) {
      initial.latitude = exif.latitude;
      initial.longitude = exif.longitude;
      initial.has_location = true;
    }
    geographicLocationSource.value = exif ? "exif" : "none";
    resetHistory(initial, geographicLocationSource.value, true);
    scheduleGeoPreview();
    return true;
  } catch (cause) {
    if (requestGeneration === openGeneration)
      geographicError.value = getErrorMessage(cause, "无法读取所选照片。请确认原生地理叠加桥已安装。");
    return false;
  } finally {
    if (requestGeneration === openGeneration) geographicOpening.value = false;
  }
}

export async function loadGeographicProject(path: string): Promise<boolean> {
  geographicError.value = "";
  geographicNotice.value = "";
  if (geographicSaving.value || geographicExporting.value) {
    geographicError.value = "请等待当前保存或导出完成后再加载项目。";
    return false;
  }
  if (!hasNativeGeoBridge) {
    geographicError.value = "加载地理项目需要 Tauri 桌面版原生桥。";
    return false;
  }
  const requestGeneration = ++openGeneration;
  previewGeneration += 1;
  window.clearTimeout(previewTimer);
  geographicOpening.value = true;
  geographicPreviewing.value = false;
  geographicPreview.value = null;
  try {
    const loaded = await invoke<GeoProjectResult>("geo_load_project", { path });
    if (requestGeneration !== openGeneration) return false;
    const nextConfig = normalizeConfig(loaded.config);
    geographicPhoto.value = loaded.photo;
    geographicProjectPath.value = path;
    const locationSource = inferLocationSource(loaded.photo, nextConfig);
    resetHistory(nextConfig, locationSource, true);
    scheduleGeoPreview();
    return true;
  } catch (cause) {
    if (requestGeneration === openGeneration)
      geographicError.value = getErrorMessage(cause, "无法加载地理项目。请确认项目文件有效。");
    return false;
  } finally {
    if (requestGeneration === openGeneration) geographicOpening.value = false;
  }
}

export async function saveGeographicProject(path: string): Promise<string | null> {
  const photo = geographicPhoto.value;
  if (!photo) {
    geographicError.value = "请先打开照片，再保存地理叠加项目。";
    return null;
  }
  if (!hasNativeGeoBridge) {
    geographicError.value = "保存地理项目需要 Tauri 桌面版原生桥。";
    return null;
  }
  const config = cloneConfig(geographicConfig.value);
  const requestGeneration = openGeneration;
  const photoId = photo.id;
  geographicSaving.value = true;
  geographicError.value = "";
  geographicNotice.value = "";
  try {
    const actualPath = await invoke<string>("geo_save_project", { id: photo.id, config, path });
    if (requestGeneration !== openGeneration || geographicPhoto.value?.id !== photoId) return null;
    geographicProjectPath.value = actualPath;
    savedConfigJson.value = JSON.stringify(config);
    geographicNotice.value = `地理项目已保存：${actualPath}`;
    return actualPath;
  } catch (cause) {
    geographicError.value = getErrorMessage(cause, "保存地理项目失败。原项目未确认更新。");
    return null;
  } finally {
    geographicSaving.value = false;
  }
}

export async function exportGeographicImage(path: string, format: GeoFormat, overlayOnly: boolean): Promise<string | null> {
  const photo = geographicPhoto.value;
  if (!photo) {
    geographicError.value = "请先打开照片，再导出地理叠加图层。";
    return null;
  }
  if (!hasNativeGeoBridge) {
    geographicError.value = "导出需要 Tauri 桌面版原生桥。";
    return null;
  }
  const config = cloneConfig(geographicConfig.value);
  const requestGeneration = openGeneration;
  const photoId = photo.id;
  geographicExporting.value = true;
  geographicError.value = "";
  geographicNotice.value = "";
  try {
    const actualPath = await invoke<string>("geo_export", {
      id: photo.id,
      config,
      path,
      format: overlayOnly ? "png" : format,
      overlayOnly,
    });
    if (requestGeneration !== openGeneration || geographicPhoto.value?.id !== photoId) return null;
    geographicNotice.value = overlayOnly
      ? `透明图层已导出：${actualPath}`
      : `全尺寸地理叠加图已导出：${actualPath}`;
    return actualPath;
  } catch (cause) {
    geographicError.value = getErrorMessage(cause, "地理叠加导出失败；请确认输出目录可写且原图未被修改。");
    return null;
  } finally {
    geographicExporting.value = false;
  }
}

export function clearGeographicMessages() {
  geographicError.value = "";
  geographicNotice.value = "";
}
