import { createApp } from "vue";
import "./style.css";
import App from "./App.vue";
import { initThemeStore } from "./stores/theme";

// 初始化主题响应引擎
initThemeStore();

createApp(App).mount("#app");
