#!/bin/bash
# DSpark track — setup pod 3090: ds4 stock @80ebbc3 + modelli. Idempotente.
set -e
export DEBIAN_FRONTEND=noninteractive

echo "=== env ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g
df -h / | tail -1
nproc

echo "=== ds4 clone+build (stock 80ebbc3, NESSUN patch) ==="
if [ ! -d /root/ds4 ]; then
  git clone https://github.com/antirez/ds4 /root/ds4
fi
cd /root/ds4
git fetch --all --quiet || true
git checkout 80ebbc3
git log --oneline -1
make cuda CUDA_ARCH=sm_86 -j"$(nproc)" 2>&1 | tail -5
ls -la ./ds4

echo "=== modelli ==="
mkdir -p /root/models
cd /root/models
# MTP prima (piccolo, 3.8GB)
if [ ! -f ds4-mtp.gguf ]; then
  wget -q --show-progress --progress=dot:giga -c -O ds4-mtp.gguf \
    "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf"
fi
# target 2-bit (86.7GB)
if [ ! -f ds4-2bit.gguf ]; then
  wget -q --show-progress --progress=dot:giga -c -O ds4-2bit.gguf \
    "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf"
fi
ls -la /root/models
echo "=== setup done ==="
