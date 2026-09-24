import { ref } from "vue";
import { BASE_URL } from "./api";

export interface BasicParams {
  exposure: number; contrast: number; highlights: number; shadows: number;
  whites: number; blacks: number; vibrance: number; saturation: number;
}
export interface DehazeParams {
  strength: number; naturalness: number; fog_retention: number;
  local_contrast: number; color_recovery: number; color_protection: number;
  highlight_protection: number; shadow_protection: number; brightness_protection: number;
}
export interface SessionPhoto {
  photo_id: string; name: string; extension: string;
  dehaze_params?: Partial<DehazeParams> | null;
  ricoh_preset_id?: string | null;
  basic_params?: Partial<BasicParams> | null;
}
export interface PhotoSession {
  session_id: string;
  files: SessionPhoto[];
  default_output_dir: string;
  ricoh_default_output_dir: string;
}
export interface PhotoSource extends PhotoSession {
  paths?: string[];
  input_dir?: string;
  owner: "enhance" | "ricoh";
  revision: number;
}

export const basicDefaults: BasicParams = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0,
  whites: 0, blacks: 0, vibrance: 0, saturation: 0,
};
export const dehazeDefaults: DehazeParams = {
  strength: 0, naturalness: 0.70, fog_retention: 0.55,
  local_contrast: 0.25, color_recovery: 0.35, color_protection: 0.80,
  highlight_protection: 0.75, shadow_protection: 0.75, brightness_protection: 0.70,
};

export const sharedPhotoSource = ref<PhotoSource | null>(null);
export const sharedSelectedPhotoId = ref("");
export const sharedDehazeByPhoto = ref<Record<string, DehazeParams>>({});
export const sharedBasicByPhoto = ref<Record<string, BasicParams>>({});
export const sharedPresetByPhoto = ref<Record<string, string | null>>({});
export const previewUseGpu = ref(false);
export const autoSaveError = ref("");

const savedSnapshots = new Map<string, string>();
const saveTimers = new Map<string, number>();
const saveChains = new Map<string, Promise<void>>();
const saveErrors = new Map<string, string>();
let initializing = false;

function snapshot(photoId: string) {
  return {
    session_id: sharedPhotoSource.value?.session_id ?? "",
    photo_id: photoId,
    dehaze_params: { ...sharedDehazeByPhoto.value[photoId] },
    basic_params: { ...sharedBasicByPhoto.value[photoId] },
    ricoh_preset_id: sharedPresetByPhoto.value[photoId] ?? null,
  };
}

function scheduleSave(photoId: string) {
  window.clearTimeout(saveTimers.get(photoId));
  saveTimers.set(photoId, window.setTimeout(() => {
    saveTimers.delete(photoId);
    void persistPhoto(photoId);
  }, 350));
}

function persistPhoto(photoId: string): Promise<void> {
  const payload = snapshot(photoId);
  const saveKey = `${payload.session_id}:${photoId}`;
  const previous = saveChains.get(photoId) ?? Promise.resolve();
  const next = previous.catch(() => {}).then(async () => {
    if (!payload.session_id || !BASE_URL.value) throw new Error("本地服务未就绪，设置未保存到 XMP");
    const response = await fetch(`${BASE_URL.value}/api/photo/settings`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || "自动保存 XMP 失败");
    }
    saveErrors.delete(saveKey);
    autoSaveError.value = [...saveErrors.values()][0] ?? "";
  }).catch((error) => {
    saveErrors.set(saveKey, error instanceof Error ? error.message : "自动保存 XMP 失败");
    autoSaveError.value = [...saveErrors.values()][0] ?? "";
  });
  saveChains.set(photoId, next);
  return next;
}

export function markPhotoChanged(photoId: string) {
  if (initializing || !sharedPhotoSource.value?.files.some(file => file.photo_id === photoId)) return;
  const next = JSON.stringify(snapshot(photoId));
  if (savedSnapshots.get(photoId) === next) return;
  savedSnapshots.set(photoId, next);
  scheduleSave(photoId);
}

export async function sharePhotoSource(
  owner: PhotoSource["owner"], source: Pick<PhotoSource, "paths" | "input_dir">,
  session: PhotoSession,
) {
  await flushPendingSaves();
  initializing = true;
  savedSnapshots.clear();
  saveChains.clear();
  saveErrors.clear();
  sharedDehazeByPhoto.value = Object.fromEntries(session.files.map(file => [
    file.photo_id, { ...dehazeDefaults, ...file.dehaze_params },
  ]));
  sharedBasicByPhoto.value = Object.fromEntries(session.files.map(file => [
    file.photo_id, { ...basicDefaults, ...file.basic_params },
  ]));
  sharedPresetByPhoto.value = Object.fromEntries(session.files.map(file => [
    file.photo_id, file.ricoh_preset_id ?? null,
  ]));
  sharedPhotoSource.value = {
    ...source, ...session, owner,
    revision: (sharedPhotoSource.value?.revision ?? 0) + 1,
  };
  sharedSelectedPhotoId.value = session.files[0]?.photo_id ?? "";
  for (const file of session.files) savedSnapshots.set(file.photo_id, JSON.stringify(snapshot(file.photo_id)));
  autoSaveError.value = "";
  initializing = false;
}

export async function flushPendingSaves() {
  const pending = [...saveTimers.keys()];
  for (const photoId of pending) {
    window.clearTimeout(saveTimers.get(photoId));
    saveTimers.delete(photoId);
  }
  await Promise.all([...pending.map(persistPhoto), ...saveChains.values()]);
}

export function clearAutoSaveErrors() {
  saveErrors.clear();
  autoSaveError.value = "";
}
