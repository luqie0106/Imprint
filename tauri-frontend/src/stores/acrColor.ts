import { ref } from "vue";

export interface AcrColorConfig {
  enabled: boolean;
  profile_tone: boolean;
  hsl: number[];
  grading: number[];
  hsl_fragment: "none" | "modern03_hsl" | "modern04_hsl" | "vintage07_hsl";
  hsl_fragment_amount: number;
}

export type AcrColorRenderStatus = "off" | "applied" | "fallback";

export const sharedAcrColorByPhoto = ref<Record<string, AcrColorConfig>>({});

export function createDefaultAcrColorConfig(): AcrColorConfig {
  return {
    enabled: false,
    profile_tone: true,
    hsl: Array(25).fill(0),
    grading: [...Array(12).fill(0), 50, 0],
    hsl_fragment: "none",
    hsl_fragment_amount: 1,
  };
}

export function cloneAcrColorConfig(config: AcrColorConfig): AcrColorConfig {
  return {
    enabled: config.enabled,
    profile_tone: config.profile_tone,
    hsl: [...config.hsl],
    grading: [...config.grading],
    hsl_fragment: config.hsl_fragment ?? "none",
    hsl_fragment_amount: config.hsl_fragment_amount ?? 1,
  };
}

export function getAcrColorConfig(photoId: string): AcrColorConfig {
  return photoId
    ? sharedAcrColorByPhoto.value[photoId] ?? createDefaultAcrColorConfig()
    : createDefaultAcrColorConfig();
}

export function ensureAcrColorConfig(photoId: string): AcrColorConfig | undefined {
  if (!photoId) return undefined;
  if (!sharedAcrColorByPhoto.value[photoId]) {
    sharedAcrColorByPhoto.value[photoId] = createDefaultAcrColorConfig();
  }
  return sharedAcrColorByPhoto.value[photoId];
}

export function resetAcrColorConfig(photoId: string): void {
  if (!photoId) return;
  sharedAcrColorByPhoto.value[photoId] = createDefaultAcrColorConfig();
}

export function parseAcrColorRenderStatus(value: string | null): AcrColorRenderStatus {
  return value === "applied" || value === "fallback" ? value : "off";
}
