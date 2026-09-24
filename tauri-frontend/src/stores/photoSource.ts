import { ref } from "vue";

export interface PhotoSource {
  paths?: string[];
  input_dir?: string;
  owner: "enhance" | "ricoh";
  revision: number;
}

export const sharedPhotoSource = ref<PhotoSource | null>(null);

export function sharePhotoSource(owner: PhotoSource["owner"], source: Pick<PhotoSource, "paths" | "input_dir">) {
  sharedPhotoSource.value = {
    ...source,
    owner,
    revision: (sharedPhotoSource.value?.revision ?? 0) + 1,
  };
}
