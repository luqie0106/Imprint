import { ref } from "vue";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";

export type ThemePreference = "system" | "dark" | "light";

const STORAGE_KEY = "imprint_theme_preference";

export const themePreference = ref<ThemePreference>("system");
export const isDark = ref(false);

/**
 * 根据偏好和系统实际状态，更新 <html> 标签 classList 和响应式状态
 */
function applyResolvedTheme() {
  let activeDark = false;

  if (themePreference.value === "dark") {
    activeDark = true;
  } else if (themePreference.value === "light") {
    activeDark = false;
  } else {
    // 跟随系统：优先使用浏览器的媒体查询，兼容性最佳且即时响应
    activeDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  isDark.value = activeDark;
  if (activeDark) {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
}

/**
 * 切换或设置主题模式
 */
export function setThemePreference(pref: ThemePreference) {
  themePreference.value = pref;
  try {
    localStorage.setItem(STORAGE_KEY, pref);
  } catch {
    // 忽略存储异常
  }
  applyResolvedTheme();
}

/**
 * 循环切换主题: 跟随系统 -> 深色模式 -> 浅色模式 -> 跟随系统
 */
export function cycleTheme() {
  const current = themePreference.value;
  if (current === "system") {
    setThemePreference("dark");
  } else if (current === "dark") {
    setThemePreference("light");
  } else {
    setThemePreference("system");
  }
}

/**
 * 初始化主题监听与存储加载
 */
export async function initThemeStore() {
  // 1. 从 LocalStorage 恢复保存的用户偏好
  try {
    const saved = localStorage.getItem(STORAGE_KEY) as ThemePreference | null;
    if (saved === "system" || saved === "dark" || saved === "light") {
      themePreference.value = saved;
    }
  } catch {
    // 忽略存储异常
  }

  // 2. 立即解析应用一次主题
  applyResolvedTheme();

  // 3. 监听浏览器/系统媒体查询变化
  const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
  mediaQuery.addEventListener("change", () => {
    if (themePreference.value === "system") {
      applyResolvedTheme();
    }
  });

  // 4. 尝试监听 Tauri 窗口原生主题事件 (兜底与增强)
  try {
    const appWindow = getCurrentWebviewWindow();
    await appWindow.onThemeChanged(() => {
      if (themePreference.value === "system") {
        applyResolvedTheme();
      }
    });
  } catch {
    // 处于普通 Web 调试环境时忽略
  }
}
