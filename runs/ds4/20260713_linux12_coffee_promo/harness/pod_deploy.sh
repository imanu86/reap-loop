#!/usr/bin/env bash
# Runs ON THE POD. Sets up R2, pulls model, verifies GPU/PCIe. No secrets echoed.
set -u
export DEBIAN_FRONTEND=noninteractive
mkdir -p /root/bin /root/masks /root/harness /root/models /root/out

echo "=== GPU / CUDA / PCIe ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head
echo "--- PCIe link ---"
nvidia-smi -q 2>/dev/null | grep -A3 -i "pci" | grep -iE "link|gen|width" | head
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader 2>&1 | head
echo "--- CUDA runtime ---"; ls -d /usr/local/cuda* 2>/dev/null | head; nvcc --version 2>/dev/null | tail -1
echo "--- RAM ---"; free -g | head -2
echo "--- disk /workspace ---"; df -h /root | tail -1

echo "=== rclone install/version (prefer latest via official installer for fast multi-thread) ==="
curl -s https://rclone.org/install.sh 2>/dev/null | bash >/dev/null 2>&1 || true
if ! command -v rclone >/dev/null 2>&1; then
  apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rclone >/dev/null 2>&1
fi
rclone version 2>/dev/null | head -1

echo "=== configure R2 (no secret echo) ==="
python3 - <<'PY'
import re,os
txt=open("/root/cf.txt",encoding="utf-8",errors="ignore").read()
toks=re.findall(r'[^\s]+',txt)
access=next((t for t in toks if re.fullmatch(r'[0-9a-fA-F]{32}',t)),None)
secret=next((t for t in toks if re.fullmatch(r'[0-9a-fA-F]{64}',t)),None)
endpoint=next((t for t in toks if t.startswith("https://") and "r2.cloudflarestorage.com" in t),None)
d=os.path.expanduser("~/.config/rclone");os.makedirs(d,exist_ok=True)
open(os.path.join(d,"rclone.conf"),"w").write(
 "[r2]\ntype = s3\nprovider = Cloudflare\n"
 f"access_key_id = {access}\nsecret_access_key = {secret}\n"
 f"endpoint = {endpoint}\nacl = private\nno_check_bucket = true\n")
print("rclone R2 configured:", "OK" if (access and secret and endpoint) else "MISSING FIELDS")
PY
rm -f /root/cf.txt
echo "--- R2 listing ---"
rclone lsf r2:ds4-models/ 2>&1 | grep -iE "ds4-2bit|sha" | head

echo "=== pull model (86GB) if absent ==="
MODEL=/root/models/ds4-2bit.gguf
WANT_SIZE=86720111488
if [ -f "$MODEL" ] && [ "$(stat -c %s "$MODEL")" = "$WANT_SIZE" ]; then
  echo "model already present, size OK"
else
  echo "downloading model from R2..."
  # --ignore-checksum + --size-only avoid the slow 86GB post-copy hash read-back; multi-thread for speed
  time rclone copyto r2:ds4-models/ds4-2bit.gguf "$MODEL" \
    --s3-chunk-size 64M --multi-thread-streams 12 --multi-thread-cutoff 128M \
    --ignore-checksum --no-traverse --transfers 1 2>&1 | tail -3
fi
echo "model size: $(stat -c %s "$MODEL" 2>/dev/null) (want $WANT_SIZE)"

chmod +x /root/bin/ds4-server
echo "=== binary quick check (linkage) ==="
ldd /root/bin/ds4-server 2>&1 | grep -iE "not found" | head || echo "all libs resolved"
echo "=== deploy done ==="
