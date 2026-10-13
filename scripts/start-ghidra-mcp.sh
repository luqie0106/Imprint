#!/bin/bash
# Open the already analyzed ACR project; never import or force analysis.
# Runs in the foreground. Ctrl+C uses pyghidra-mcp's clean shutdown handler.
set -euo pipefail

project=/Users/hualaiwu/Documents/GhidraProjects/CameraRaw-17.2-arm64/CameraRaw-17.2-arm64.gpr
runtime=/Users/hualaiwu/.conda/envs/pyghidra-mcp/bin/pyghidra-mcp
install=/Users/hualaiwu/Applications/Ghidra/ghidra_12.1.4_PUBLIC
lock_file=${project%.gpr}.lock

if [[ ! -f "$project" || ! -x "$runtime" || ! -d "$install" ]]; then
  echo 'Ghidra project, dedicated MCP environment, or installation is missing.' >&2
  exit 1
fi

listener=$(/usr/sbin/lsof -nP -t -iTCP:8000 -sTCP:LISTEN 2>/dev/null || true)
if [[ -n "$listener" ]]; then
  echo 'Port 8000 already has a listener. Verify its MCP response; no second instance started.'
  exit 0
fi
# A server still initializing may not listen yet. Refuse to compete with it.
processes=$(/bin/ps -axo command)
if printf '%s\n' "$processes" | /usr/bin/grep -E '[p]yghidra-mcp|[p]yghidra_mcp' >/dev/null; then
  echo 'An existing PyGhidra MCP process may be initializing. Inspect it before retrying.' >&2
  exit 1
fi
if [[ -f "$lock_file" ]] && /usr/sbin/lsof -t "$lock_file" >/dev/null 2>&1; then
  echo 'The saved project lock is in use. No second instance started.' >&2
  exit 1
fi
# Stale lock files are left untouched; Ghidra itself determines lock ownership.
export GHIDRA_INSTALL_DIR="$install"
export JAVA_HOME
JAVA_HOME=$(/usr/libexec/java_home -v 25)
exec "$runtime" --transport streamable-http --host 127.0.0.1 --port 8000 \
  --project-path "$project" --no-force-analysis --no-wait-for-analysis
