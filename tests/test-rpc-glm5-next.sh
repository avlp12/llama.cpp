#!/usr/bin/env bash
# Synthetic CPU-only GLM full-graph TP check; never loads checkpoint weights.
set -euo pipefail
build=${1:?build directory required}
ranks=${2:-4}
case "$ranks" in 2|4) ;; *) echo 'ranks must be 2 or 4' >&2; exit 2 ;; esac
scratch=$(mktemp -d)
pids=()
args=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    rm -rf "$scratch"
}
trap cleanup EXIT
port_base=$((40000 + $$ % 9000))
for ((rank=0; rank<ranks; rank++)); do
    port=$((port_base+rank))
    "$build/bin/ggml-rpc-server" --device CPU --host 127.0.0.1 --port "$port" >"$scratch/rank-$rank.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    ready=false
    for ((attempt=0; attempt<200; attempt++)); do
        kill -0 "$pid" 2>/dev/null || { cat "$scratch/rank-$rank.log" >&2; exit 1; }
        if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
            ready=true; break
        fi
        sleep 0.05
    done
    "$ready" || { cat "$scratch/rank-$rank.log" >&2; exit 1; }
    args+=(--rpc "127.0.0.1:$port")
done
LLAMA_SPLIT_EXPERTS=slice "$build/bin/test-llama-archs" --arch glm5-next "${args[@]}"
