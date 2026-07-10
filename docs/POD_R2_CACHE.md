# Pod deploy cache on Cloudflare R2

Fast-deploy cache so a fresh pod skips the ~87 GB HuggingFace model download and
the ds4 CUDA source build. Pull prebuilt binaries + the model from R2 instead
(R2 egress to a pod is fast; ingress to R2 is free).

## Credentials

R2 S3 credentials live in a LOCAL file on the workstation:
`C:\Users\imanu\Desktop\cf.txt` (Cloudflare API token, Access Key ID, Secret
Access Key, jurisdiction, S3 endpoint `https://<account>.r2.cloudflarestorage.com`).
NEVER copy the values into any repo, log, or command line. Configure rclone by
reading the file into a config writer that does not echo the values (see the
`r2_config.py` pattern below).

## Configure rclone on the pod (no value echo)

`scp` the credentials file to the pod, then run a parser that writes
`~/.config/rclone/rclone.conf` directly (regex-parses the 32-hex Access Key ID,
64-hex Secret Access Key, and the `https://…r2.cloudflarestorage.com` endpoint;
the `cfat_…` API token is not needed for S3):

```python
# r2_config.py — run ON THE POD; prints only YES/NO, never the secrets
import re, os, sys
txt = open("/root/cf.txt", encoding="utf-8", errors="ignore").read()
toks = re.findall(r'[^\s]+', txt)
access   = next((t for t in toks if re.fullmatch(r'[0-9a-fA-F]{32}', t)), None)
secret   = next((t for t in toks if re.fullmatch(r'[0-9a-fA-F]{64}', t)), None)
endpoint = next((t for t in toks if t.startswith("https://") and "r2.cloudflarestorage.com" in t), None)
d = os.path.expanduser("~/.config/rclone"); os.makedirs(d, exist_ok=True)
open(os.path.join(d, "rclone.conf"), "w").write(
    "[r2]\ntype = s3\nprovider = Cloudflare\n"
    f"access_key_id = {access}\nsecret_access_key = {secret}\n"
    f"endpoint = {endpoint}\nacl = private\nno_check_bucket = true\n")
```

Then remove the credentials file from the pod: `rm -f /root/cf.txt`.

## Bucket — `ds4-models` (already exists, reuse it)

`rclone lsd r2:` shows an existing **`ds4-models`** bucket. As of 2026-07-10 it
already holds:

| object | what |
|---|---|
| `ds4-2bit.gguf` | the 86 720 111 488-byte model (same HF quant) — **already cached** |
| `ds4-2bit.gguf.sha256` | model checksum (added 2026-07-10) |
| `ds4-server_sm86_livetree-771a39a8` (+`.meta`) | rotate32+sensor+PACE live-tree build (added 2026-07-10) |
| `ds4_sm86_livetree-771a39a8` | ds4 CLI, same build |
| `ds4-reaploop-sm86` (+`.meta`) | earlier **canonical** build (0001-0007,0011-0013, no rotation/PACE) |
| `ds4-pace-sm86` (+`.sha`) | earlier pace build |
| `ds4-src_livetree-771a39a8.tgz` (+`.sha256`) | **SOURCE tarball** of the live tree (post-0018 `/root/ds4`, `ds4.c` md5 `771a39a8…`; sources only — no `.o`/binaries/`.git`, 50 MB, 343 files). Lets a fresh pod REBUILD for any arch and apply the pace chain: extract, `git init`, `git apply` 0020→0021→0026 (verified apply-clean on this base 2026-07-10), `make cuda CUDA_ARCH=sm_NN -j$(nproc)` (~4 min on 32 vcpu, 0 warnings). Added 2026-07-10. |
| `patches_0020_0021_0026.tgz` | the pace patch chain (canonical copies from reap-loop `patches/ds4/`) matching the source tarball above (added 2026-07-10) |

So a fresh pod needs **no model download and no build** — just pull the model +
the binary matching the mask/rotation features you need. If no prebuilt binary
matches (different arch, or you need a patch past the cached builds), pull the
source tarball + patch chain instead and rebuild on-pod (minutes, see row).

rclone v1.58 predates the `Cloudflare` S3 provider id, so pass
`--s3-provider=Other` on each command (Cloudflare R2 is plain S3-compatible).
(rclone ≥ v1.60 knows `provider = Cloudflare` natively — the install.sh on a
2026 pod image gives v1.74, where plain `rclone` commands work without the
`--s3-provider=Other` override.)

## Upload a new binary (from the pod)

```
# versioned by live-tree md5 + arch; write a companion .meta with sha256+provenance
rclone --s3-provider=Other copyto /root/ds4-live/ds4-server r2:ds4-models/ds4-server_sm86_livetree-771a39a8
rclone --s3-provider=Other copyto /root/ds4-live/ds4        r2:ds4-models/ds4_sm86_livetree-771a39a8
rclone --s3-provider=Other copyto /root/ds4-server_sm86_livetree-771a39a8.meta r2:ds4-models/ds4-server_sm86_livetree-771a39a8.meta
```

The model is already cached; only re-upload it if the quant changes (ingress is
free; `--s3-chunk-size 64M`). Each `.meta` records ds4.c md5, inline patches,
gpu_arch, image, build date, and sha256.

## Fast deploy from cache (next pod)

```
mkdir -p /root/models /root/bin
rclone --s3-provider=Other copyto r2:ds4-models/ds4-2bit.gguf /root/models/ds4-2bit.gguf
rclone --s3-provider=Other copyto r2:ds4-models/ds4-server_sm86_livetree-771a39a8 /root/bin/ds4-server
chmod +x /root/bin/ds4-server
# verify: sha256sum -c against r2:ds4-models/ds4-2bit.gguf.sha256 and the binary .meta
```

Minutes instead of an ~87 GB HF download + full `make cuda`. Pick the binary by
feature set: `*_livetree-771a39a8` has rotate32 + S1 sensor + PACE;
`ds4-reaploop-sm86` is canonical (no rotation). Binaries are arch-pinned
(`sm_86`); rebuild from live-tree source if the GPU arch differs.

## Cost

R2 storage ≈ $0.015/GB-month ⇒ the ~81 GB model ≈ **$1.2/month**; binaries are
negligible. R2 has no egress fees.

See also the pod deploy recipe:
`runs/ds4/20260710_pod_t1_full_positive_control/README.md`.
