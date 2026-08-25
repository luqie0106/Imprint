import { createApp } from "vue";
import "./style.css";
import App from "./App.vue";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";

async function initTheme() {
  try {
    const appWindow = getCurrentWebviewWindow();
    const theme = await appWindow.theme(); // 'light' | 'dark' | null
    applyTheme(theme);
    // 监听系统主题变化（用户切换时实时响应）
    await appWindow.onThemeChanged(({ payload: theme }) => {
      applyTheme(theme);
    });
  } catch {
    // 浏览器预览模式：降级为媒体查询
    if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      document.documentElement.classList.add("dark");
    }
  }
}

function applyTheme(theme: string | null) {
  if (theme === "dark") {
    document.documentElement.classList.add("dark");
  } else {
    document.documentElement.classList.remove("dark");
  }
}

initTheme();
createApp(App).mount("#app");
