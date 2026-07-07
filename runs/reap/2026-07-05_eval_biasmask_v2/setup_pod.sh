#!/bin/bash
# Setup standard pod REAP: aria2 download bg + clone/patch/build/regression, poi attende il modello.
set -u
export DEBIAN_FRONTEND=noninteractive
(apt-get update -qq && apt-get install -y -qq aria2) > /tmp/apt.log 2>&1
mkdir -p /root/models
aria2c -c -x8 -s8 -k4M --console-log-level=warn --summary-interval=0 -d /root/models -o ds4-2bit.gguf \
  "https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf" > /root/aria.log 2>&1 &
ARIA_PID=$!

sed -i "s/\r$//" /root/000*.patch /root/*.sh /root/reap_bias_mask_ds4.py /root/corpus/*.txt 2>/dev/null
cd /root
git clone -q https://github.com/antirez/ds4
cd ds4 && git checkout -q 80ebbc3
for p in /root/0001*.patch /root/0002*.patch /root/0003*.patch /root/0004*.patch /root/0005*.patch /root/0006*.patch; do
  git apply "$p" || { echo "PATCH_FAIL $p"; exit 1; }
done
export PATH=/usr/local/cuda/bin:$PATH
make cuda CUDA_ARCH=sm_86 -j32 > /root/build_pod.log 2>&1
echo "BUILD_EXIT=$?"
make cuda-regression CUDA_ARCH=sm_86 > /root/regression_pod.log 2>&1
echo "REGRESSION_EXIT=$?"

wait $ARIA_PID
SIZE=$(stat -c%s /root/models/ds4-2bit.gguf 2>/dev/null || echo 0)
if [ "$SIZE" != "86720111488" ] || [ -f /root/models/ds4-2bit.gguf.aria2 ]; then
  echo "DOWNLOAD_FAIL size=$SIZE"; tail -3 /root/aria.log; exit 1
fi
echo "SETUP_OK size=$SIZE"
