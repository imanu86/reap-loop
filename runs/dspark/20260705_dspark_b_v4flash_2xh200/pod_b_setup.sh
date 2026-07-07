#!/bin/bash
# Strada B — pod 2xH200 secure: DeepSeek-V4-Flash-DSpark ufficiale (167GB fp8/fp4).
# Pipeline: deps + download HF in parallelo -> convert MP=2 -> pronto per generate.
set -e
cd /root

echo "=== env ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | head -2 | tail -1
df -h / | tail -1

echo "=== pip deps (in parallelo col download) ==="
pip install -q -U "huggingface_hub[hf_transfer]" &
PIP1=$!

echo "=== download modello (167GB, hf_transfer) ==="
wait $PIP1
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p /root/hf
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --local-dir /root/hf/dspark --max-workers 16 2>&1 | tail -5 &
DL=$!

echo "=== deps inference (torch 2.10 + tilelang + fht) ==="
pip install -q -U "torch>=2.10.0" --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
pip install -q -U "transformers>=5.0.0" "safetensors>=0.7.0" tilelang==0.1.8 2>&1 | tail -2
pip install -q fast_hadamard_transform 2>&1 | tail -2

echo "=== attesa fine download ==="
wait $DL
du -sh /root/hf/dspark

echo "=== convert MP=2 ==="
cd /root/hf/dspark/inference
python convert.py --hf-ckpt-path /root/hf/dspark --save-path /root/ckpt-mp2 \
  --n-experts 256 --model-parallel 2 2>&1 | tail -5
ls -la /root/ckpt-mp2 | head -5

echo "=== SETUP B DONE ==="
