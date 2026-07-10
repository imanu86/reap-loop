#!/bin/bash
set -e
echo "[canon] start $(date -Is)"
cd /root
if [ ! -d /root/ds4-canon/.git ]; then
  git clone --quiet https://github.com/antirez/ds4 /root/ds4-canon
fi
cd /root/ds4-canon
git checkout --quiet 80ebbc3
git clean -qfdx || true
PD=/root/patches_repo/patches/ds4
CHAIN=$(ls "$PD/canonical"/*.patch | sort)
CHAIN="$CHAIN $PD/0027-rewind-exactness-harness.patch $PD/0028-spex-trace-tokens.patch"
N=0
for p in $CHAIN; do
  N=$((N+1))
  if ! git apply --check "$p" 2>>/root/canon_build_apply.err; then
    echo "[canon] FAIL apply-check #$N: $(basename $p)"; exit 3
  fi
  git apply "$p"
  echo "[canon] applied #$N $(basename $p)"
done
echo "[canon] ds4.c md5: $(md5sum ds4.c | cut -c1-8) (expect 62ed2e71)"
export PATH=/usr/local/cuda/bin:$PATH
make cuda CUDA_ARCH=sm_86 -j32 > /root/canon_make.log 2>&1 && echo "[canon] MAKE OK" || { echo "[canon] MAKE FAIL"; tail -30 /root/canon_make.log; exit 4; }
ls -la ds4 ds4-server 2>/dev/null
sha256sum ds4 ds4-server 2>/dev/null | tee /root/canon_sha256.txt
echo "[canon] done $(date -Is)"
