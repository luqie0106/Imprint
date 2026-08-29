import { ref, onUnmounted } from "vue";
import { BASE_URL, initApiConnection } from "../stores/api";

export interface SseOptions {
  onProgress?: (msg: string, pct?: number | null) => void;
  onDone?: (data: any) => void;
  onError?: (err: string) => void;
}

export function useSse(path: string, options?: SseOptions) {
  const messages = ref<string[]>([]);
  const lastMessage = ref<string>("");
  const progressPct = ref<number | null>(null);
  const isDone = ref(false);
  const isRunning = ref(false);
  const error = ref<string | null>(null);
  const resultData = ref<any>(null);

  let abortController: AbortController | null = null;
  let activeReader: ReadableStreamDefaultReader<Uint8Array> | null = null;

  async function cancel() {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    if (activeReader) {
      try {
        await activeReader.cancel();
      } catch {
        // 忽略已关闭的 reader 异常
      }
      activeReader = null;
    }
    isRunning.value = false;
  }

  async function start(body: object) {
    await cancel();

    messages.value = [];
    lastMessage.value = "";
    progressPct.value = null;
    isDone.value = false;
    error.value = null;
    resultData.value = null;
    isRunning.value = true;

    abortController = new AbortController();

    const fullUrl = path.startsWith("http") ? path : `${BASE_URL.value}${path}`;

    try {
      const resp = await fetch(fullUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: abortController.signal,
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      }

      if (!resp.body) {
        throw new Error("响应流为空");
      }

      const reader = resp.body.getReader();
      activeReader = reader;
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        // 保留最后一个可能不完整的行
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;

          const jsonStr = trimmed.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);

            if (event.type === "progress") {
              if (event.msg) {
                messages.value.push(event.msg);
                lastMessage.value = event.msg;
              }
              if (typeof event.pct === "number") {
                progressPct.value = event.pct;
              }
              options?.onProgress?.(event.msg, event.pct);
            } else if (event.type === "done") {
              isDone.value = true;
              resultData.value = event;
              if (event.msg) {
                messages.value.push(event.msg);
                lastMessage.value = event.msg;
              }
              options?.onDone?.(event);
            } else if (event.type === "error") {
              const errText = event.msg || "发生未知错误";
              error.value = errText;
              messages.value.push(`❌ ${errText}`);
              options?.onError?.(errText);
            }
          } catch (jsonErr) {
            console.error("解析 SSE 数据失败:", jsonErr, jsonStr);
          }
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        const errMsg = err.message || String(err);
        error.value = errMsg;
        messages.value.push(`❌ 连接异常: ${errMsg}`);
        options?.onError?.(errMsg);
        if (
          errMsg.toLowerCase().includes("fetch") ||
          errMsg.toLowerCase().includes("network") ||
          errMsg.toLowerCase().includes("failed")
        ) {
          initApiConnection();
        }
      }
    } finally {
      isRunning.value = false;
      activeReader = null;
      abortController = null;
    }
  }

  // 组件卸载时自动取消流，防止内存/连接泄漏
  onUnmounted(() => {
    cancel();
  });

  return {
    messages,
    lastMessage,
    progressPct,
    isDone,
    isRunning,
    error,
    resultData,
    start,
    cancel,
  };
}
