"""Read-only donors -> new isolated source; no build, GPU, git or live changes."""
import hashlib
import json
import shutil
from pathlib import Path

DONOR = Path(r"C:\Users\imanu\Documents\Codex\2026-09-13\rip\work\qwen_ram_decode_opt_20260920\source")
OVERLAY = Path(r"D:\ds4_work\elastico_q36_m3\source-overlay")
MANIFEST = Path(r"D:\ds4_work\elastico_q36_m3\evidence\q36m3-build-manifest.json")
CUDA_BASE = Path(r"D:\ds4_work\qwen4b_kernel_lab\source")
DEST = Path(r"D:\ds4_work\qwen36_reap_lab\source")
REPORT = Path(__file__).with_name("source-provenance.json")

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def main():
    if DEST.exists() or REPORT.exists():
        raise RuntimeError("Refuse to overwrite destination or provenance")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    overlays = sorted(p for p in OVERLAY.rglob("*") if p.is_file())
    for p in overlays:
        if digest(p) != manifest["sources"][str(p)]:
            raise RuntimeError(f"Overlay provenance mismatch: {p}")
    live = Path(r"D:\ds4_work\bin_q36m3\llama.dll")
    if digest(live) != manifest["q36m3_sha256"]["llama.dll"]:
        raise RuntimeError("Live M3 fingerprint differs from manifest")
    records = []
    cuda_differences = []
    files = sorted(p for p in DONOR.rglob("*") if p.is_file())
    if any(p.is_symlink() for p in DONOR.rglob("*")):
        raise RuntimeError("Refuse symlink donor")
    for source in files:
        rel = source.relative_to(DONOR)
        selected = OVERLAY / rel
        selected = selected if selected.is_file() else source
        before = digest(selected)
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, target)
        after = digest(target)
        if before != after or before != digest(selected):
            raise RuntimeError(f"Copy verification failed: {selected}")
        records.append({"path": rel.as_posix(), "source": str(selected), "sha256": after})
        base = CUDA_BASE / rel
        if rel.parts[0] == "ggml" and (not base.is_file() or digest(base) != digest(source)):
            cuda_differences.append(rel.as_posix())
    for selected in overlays:
        rel = selected.relative_to(OVERLAY)
        if (DONOR / rel).exists():
            continue
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected, target)
        if digest(selected) != digest(target):
            raise RuntimeError(f"New overlay copy mismatch: {selected}")
        records.append({"path": rel.as_posix(), "source": str(selected), "sha256": digest(target)})
    REPORT.write_text(json.dumps({"schema_version": 1, "donor": str(DONOR), "overlay": str(OVERLAY), "destination": str(DEST), "manifest_sha256": digest(MANIFEST), "live_llama_sha256": digest(live), "cuda_base_ggml_differences": cuda_differences, "files": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"copied_verified": len(records), "cuda_base_ggml_differences": cuda_differences, "report": str(REPORT)}))

if __name__ == "__main__":
    main()
