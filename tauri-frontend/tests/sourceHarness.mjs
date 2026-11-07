import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const frontendRoot = fileURLToPath(new URL("../", import.meta.url));

export function readSource(relativePath) {
  return readFileSync(`${frontendRoot}${relativePath}`, "utf8");
}

export function sourceBetween(source, start, end) {
  const startIndex = source.indexOf(start);
  const endIndex = source.indexOf(end, startIndex + start.length);
  if (startIndex < 0 || endIndex < 0) throw new Error(`Source section not found: ${start}`);
  return source.slice(startIndex, endIndex);
}

export function loadFunctions(source, names, scope) {
  const javascript = stripTypeScriptTypes(source, { mode: "strip" });
  return runInNewContext(`${javascript}\n;({ ${names.join(", ")} });`, scope);
}

export function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

export const ref = value => ({ value });
