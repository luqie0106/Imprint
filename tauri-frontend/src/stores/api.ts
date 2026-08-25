import { ref, computed } from "vue";
import { invoke } from "@tauri-apps/api/core";

export const apiPort = ref<number | null>(null);
export const isServerReady = ref(false);
export const serverError = ref<string | null>(null);
export const isConnecting = ref(true);

export const BASE_URL = computed(() => {
  if (apiPort.value) {
    return `http://127.0.0.1:${apiPort.value}`;
  }
  return "";
});

/**
 * 轮询等待 Python FastAPI Sidecar 启动并就绪
 */
export async function initApiConnection(maxRetries = 60, intervalMs = 100): Promise<boolean> {
  isConnecting.value = true;
  serverError.value = null;

  for (let i = 0; i < maxRetries; i++) {
    try {
      const port = await invoke<number>("get_api_port");
      if (port) {
        apiPort.value = port;
        // 测试后端 health / status
        try {
          const resp = await fetch(`http://127.0.0.1:${port}/api/models/status`);
          if (resp.ok) {
            isServerReady.value = true;
            isConnecting.value = false;
            return true;
          }
        } catch {
          // 端口已分配但服务可能正在初始化，继续等待
        }
      }
    } catch {
      // 端口尚未准备好
    }
    // 前 10 次尝试 100ms 极速轮询，后续回退为 250ms
    const waitTime = i < 10 ? intervalMs : 250;
    await new Promise((resolve) => setTimeout(resolve, waitTime));
  }

  isConnecting.value = false;
  serverError.value = "无法连接至 Python FastAPI 后端服务，请检查 Python 环境及依赖配置。";
  return false;
}
