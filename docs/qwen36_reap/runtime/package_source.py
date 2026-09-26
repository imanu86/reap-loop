"""Package only the lab delta over the hash-verified donor+M3 assembly."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
LAB = Path(r"D:\ds4_work\qwen36_reap_lab\source")
CHANGED = {"src/CMakeLists.txt", "src/llama-context.cpp", "src/llama-context.h", "src/llama-graph.cpp"}
NEW = {"src/llama-reap.cpp", "src/llama-reap.h"}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    provenance = json.loads((ROOT / "source-provenance.json").read_text(encoding="utf-8"))
    diff = []
    records = []
    observed = set()
    known = {r["path"] for r in provenance["files"]}
    for record in provenance["files"]:
        rel = record["path"]
        base = Path(record["source"]).read_bytes()
        current = (LAB / rel).read_bytes()
        if sha(base) != record["sha256"]:
            raise RuntimeError(f"Donor changed since assembly: {rel}")
        if base == current:
            continue
        if rel not in CHANGED:
            raise RuntimeError(f"Unexpected modified lab file: {rel}")
        observed.add(rel)
        diff.extend(difflib.unified_diff(base.decode("utf-8").splitlines(True), current.decode("utf-8").splitlines(True), fromfile="a/" + rel, tofile="b/" + rel))
        records.append({"path": rel, "base_sha256": sha(base), "patched_sha256": sha(current)})
    if observed != CHANGED:
        raise RuntimeError("Missing expected integration edit")
    actual_new = {p.relative_to(LAB).as_posix() for p in LAB.rglob("*") if p.is_file()} - known
    if actual_new != NEW:
        raise RuntimeError(f"Unexpected new source files: {actual_new ^ NEW}")
    for rel in sorted(NEW):
        current = (LAB / rel).read_bytes()
        diff.extend(difflib.unified_diff([], current.decode("utf-8").splitlines(True), fromfile="/dev/null", tofile="b/" + rel))
        records.append({"path": rel, "base_sha256": None, "patched_sha256": sha(current)})
    patch = "".join(diff).encode("utf-8")
    (ROOT / "qwen36-reap-runtime.patch").write_bytes(patch)
    report = {"schema_version": 1, "status": "packaged_runtime_identity_pending", "base": "donor_plus_verified_M3_overlay", "patch_sha256": sha(patch), "files": records, "constraints": ["Parent-authorized isolated CPU build; no GPU runs", "No live/donor/base CUDA edits", "No weight edits", "No commits or pushes"]}
    (ROOT / "patch-manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
