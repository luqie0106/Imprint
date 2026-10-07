import { ref } from "vue";
// Shared by both editors so importing a new session cannot interrupt an export.
export const photoExportRunning = ref(false);
