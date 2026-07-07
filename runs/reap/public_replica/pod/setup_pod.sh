#!/bin/bash
# Setup pod A6000 sm_86: download modello bg + clone/patch(0001-0007)/build/regression.
set -u
export DEBIAN_FRONTEND=noninteractive
export PATH=/usr/local/cuda/bin:$PATH
(apt-get update -qq && apt-get install -y -qq aria2) > /tmp/apt.log 2>&1
mkdir -p /root/models
aria2c -c -x8 -s8 -k4M --console-log-level=warn --summary-interval=0 -d /root/models -o ds4-2bit.gguf \
  "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" > /root/aria.log 2>&1 &
ARIA_PID=$!

sed -i "s/\r$//" /root/patches/000*.patch /root/scripts/*.py /root/prod/prompts/*.txt /root/prod/corpus/*.txt /root/ric/prompts/*.txt /root/ric/corpus/*.txt 2>/dev/null
cd /root
rm -rf ds4
git clone -q https://github.com/antirez/ds4
cd ds4 && git checkout -q 80ebbc3
for p in /root/patches/0001*.patch /root/patches/0002*.patch /root/patches/0003*.patch \
         /root/patches/0004*.patch /root/patches/0005*.patch /root/patches/0006*.patch \
         /root/patches/0007*.patch; do
  git apply "$p" || { echo "PATCH_FAIL $p"; exit 1; }
  echo "applied $(basename $p)"
done
make cuda CUDA_ARCH=sm_86 -j"$(nproc)" > /root/build_pod.log 2>&1
echo "BUILD_EXIT=$?"
make cuda-regression CUDA_ARCH=sm_86 > /root/regression_pod.log 2>&1
echo "REGRESSION_EXIT=$?"
tail -1 /root/regression_pod.log

echo "waiting model download..."
wait $ARIA_PID
SIZE=$(stat -c%s /root/models/ds4-2bit.gguf 2>/dev/null || echo 0)
if [ "$SIZE" != "86720111488" ] || [ -f /root/models/ds4-2bit.gguf.aria2 ]; then
  echo "DOWNLOAD_FAIL size=$SIZE"; tail -3 /root/aria.log; exit 1
fi
echo "SETUP_OK size=$SIZE"
