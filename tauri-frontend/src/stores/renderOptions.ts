import { ref, watch } from "vue";

export type RenderBackend = "auto" | "native" | "pytorch" | "cpu";
const saved = localStorage.getItem("imprint-render-backend");
export const renderBackend = ref<RenderBackend>(
  saved === "native" || saved === "pytorch" || saved === "cpu" ? saved : "auto",
);
watch(renderBackend, value => localStorage.setItem("imprint-render-backend", value));

export type OperatorBackend = "python" | "native";
function savedOperator(key: string, fallback: OperatorBackend): OperatorBackend {
  const value = localStorage.getItem(key);
  return value === "native" || value === "python" ? value : fallback;
}
export const basicBackend = ref<OperatorBackend>(savedOperator(
  "imprint-basic-backend", saved === "native" || saved === "auto" || saved === null ? "native" : "python",
));
export const ricohBackend = ref<OperatorBackend>(savedOperator("imprint-ricoh-backend", "python"));
export const sortBackend = ref<OperatorBackend>(savedOperator("imprint-sort-backend", "python"));
watch(basicBackend, value => localStorage.setItem("imprint-basic-backend", value));
watch(ricohBackend, value => localStorage.setItem("imprint-ricoh-backend", value));
watch(sortBackend, value => localStorage.setItem("imprint-sort-backend", value));
