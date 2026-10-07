import {
  sharedBasicByPhoto, sharedDehazeAutoByPhoto, sharedDehazeAutoExposureByPhoto,
  sharedDehazeByPhoto, sharedDehazeNonlocalByPhoto, sharedPhotoSource,
  sharedPresetByPhoto,
  type DehazeParams,
} from "./stores/photoSource";

const EPSILON = 1e-6;

export function photoChangeSummary(photoId: string): string {
  const categories: string[] = [];
  const basicParams = sharedBasicByPhoto.value[photoId];
  if (basicParams && Object.values(basicParams).some(value => Number.isFinite(value) && Math.abs(value) > EPSILON)) {
    categories.push("基础调整");
  }
  if (sharedPresetByPhoto.value[photoId]) categories.push("理光预设");

  const params = sharedDehazeByPhoto.value[photoId];
  const defaults = sharedPhotoSource.value?.dehaze_defaults;
  const dehazeParamsChanged = params && defaults
    && (Object.keys(defaults) as Array<keyof DehazeParams>).some(key => Math.abs(params[key] - defaults[key]) > EPSILON);
  const autoMode = sharedDehazeAutoByPhoto.value[photoId] ?? false;
  const autoExposure = sharedDehazeAutoExposureByPhoto.value[photoId] ?? false;
  const nonlocalMode = sharedDehazeNonlocalByPhoto.value[photoId] ?? "off";

  if (dehazeParamsChanged) categories.push("去朦胧参数");
  if (autoMode) categories.push("自动处理");
  if (autoExposure) categories.push("自动曝光");
  if (nonlocalMode !== "off") categories.push(`实验${nonlocalMode === "strong" ? "强" : "保守"}`);
  return categories.join(" · ");
}

export function rangeChangeStyle(value: number, minimum: number, maximum: number, baseline: number) {
  const toPercent = (current: number) => {
    const clamped = Math.min(maximum, Math.max(minimum, current));
    return ((clamped - minimum) / (maximum - minimum)) * 100;
  };
  const currentPosition = toPercent(value);
  const baselinePosition = toPercent(baseline);
  return {
    "--range-start": `${Math.min(currentPosition, baselinePosition)}%`,
    "--range-end": `${Math.max(currentPosition, baselinePosition)}%`,
  };
}
