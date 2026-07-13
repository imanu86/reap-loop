#!/usr/bin/env python3
"""Extract coffee-mask promo/prune metrics for one arm dir.

Parses server.stderr.log (zero-copy DIAG, register line, budget, cache profile),
response.json + stream_events.jsonl (decode/overall t/s), gpu_mem.log (peak VRAM).
"""
import json, re, sys, glob
from pathlib import Path


def num(s):
    try:
        return float(s)
    except Exception:
        return None


def main(argd):
    d = Path(argd)
    out = {"arm": d.name}
    stderr = (d / "server.stderr.log").read_text(errors="replace") if (d / "server.stderr.log").exists() else ""

    # --- register line: "X/Y ranges mapped, Z GiB pinned in Ts" ---
    m = re.search(r"masked zero-copy register:\s*(\d+)/(\d+) ranges mapped,\s*([\d.]+) GiB pinned in ([\d.]+)s", stderr)
    if m:
        out["ranges_mapped"] = f"{m.group(1)}/{m.group(2)}"
        out["gib_pinned"] = float(m.group(3))
        out["register_s"] = float(m.group(4))
        out["register_gib_s"] = round(float(m.group(3)) / float(m.group(4)), 3) if float(m.group(4)) else None

    m = re.search(r"stream-from-RAM masked window:\s*(\d+) layer,\s*(\d+) expert kept,\s*(\d+)/(\d+) ranges registered,\s*([\d.]+) GiB zero-copy", stderr)
    if m:
        out["masked_layers"] = int(m.group(1))
        out["experts_kept"] = int(m.group(2))
        out["gib_zerocopy"] = float(m.group(5))

    # --- budget cap hit ---
    m = re.search(r"budget ([\d.]+) GiB total / ([\d.]+) GiB per layer .*reached for (\d+) expert runs", stderr)
    if m:
        out["budget_total_gib"] = float(m.group(1))
        out["budget_per_layer_gib"] = float(m.group(2))
        out["budget_reached_runs"] = int(m.group(3))
    else:
        out["budget_reached_runs"] = 0

    # --- final zero-copy diag ---
    diag = re.findall(r"masked zero-copy diag (?:final|periodic):\s*queries=(\d+)/([\d.]+) MiB covered=(\d+)/([\d.]+) MiB dma_ok=(\d+)/([\d.]+) MiB dma_failed=(\d+).*?miss_before=(\d+) miss_range=(\d+)", stderr)
    if diag:
        last = diag[-1]
        q, qmib, cov, covmib, dok, dokmib, dfail, mb, mr = last
        out["zc_queries"] = int(q)
        out["zc_query_MiB"] = float(qmib)
        out["zc_covered"] = int(cov)
        out["zc_covered_MiB"] = float(covmib)
        out["zc_dma_ok_MiB"] = float(dokmib)
        out["zc_dma_failed"] = int(dfail)
        out["zc_coverage_pct"] = round(100.0 * int(cov) / int(q), 2) if int(q) else None
        out["zc_miss_before"] = int(mb)
        out["zc_miss_range"] = int(mr)

    # --- expert-cache hit rate (CACHE_PROFILE json) ---
    hrs = re.findall(r'\{"n":(\d+),"hits":(\d+),"selections":(\d+),"hit_rate":([\d.]+),"weighted_hit_rate":([\d.]+)\}', stderr)
    if not hrs:
        hrs2 = re.findall(r'\{"n":(\d+),"hits":(\d+),"hit_rate":([\d.]+),"weighted_hit_rate":([\d.]+)\}', stderr)
        if hrs2:
            n, h, hr, whr = hrs2[-1]
            out["cache_hit_rate"] = float(hr)
            out["cache_weighted_hit_rate"] = float(whr)
    else:
        n, h, sel, hr, whr = hrs[-1]
        out["cache_n"] = int(n)
        out["cache_hits"] = int(h)
        out["cache_selections"] = int(sel)
        out["cache_hit_rate"] = float(hr)
        out["cache_weighted_hit_rate"] = float(whr)

    # --- per-layer streaming selected: sum hits/misses/copied MiB ---
    sel = re.findall(r"CUDA streaming selected layer=\d+ .*?hits=(\d+) misses=(\d+) direct=(\d+) evictions=(\d+) gate/up ([\d.]+) MiB down ([\d.]+) MiB", stderr)
    if sel:
        hits = sum(int(x[0]) for x in sel)
        miss = sum(int(x[1]) for x in sel)
        gu = sum(float(x[4]) for x in sel)
        dn = sum(float(x[5]) for x in sel)
        out["sel_hits"] = hits
        out["sel_misses"] = miss
        out["sel_hit_rate"] = round(hits / (hits + miss), 4) if (hits + miss) else None
        out["sel_copied_MiB"] = round(gu + dn, 1)

    # --- response.json / usage ---
    rj = d / "response.json"
    if rj.exists():
        r = json.loads(rj.read_text(errors="replace"))
        out["elapsed_s"] = r.get("elapsed_s")
        out["finish_reason"] = (r.get("choices") or [{}])[0].get("finish_reason")
        out["client_stop"] = (r.get("client_stop") or {}).get("reason") if r.get("client_stop") else None
        out["stream_error"] = r.get("stream_error")
        u = r.get("usage") or {}
        out["completion_tokens"] = u.get("completion_tokens")
        out["prompt_tokens"] = u.get("prompt_tokens")
        content = (r.get("choices") or [{}])[0].get("message", {}).get("content", "")
        out["content_chars"] = len(content)
        out["has_html_close"] = "</html>" in content.lower()

    # --- decode t/s from stream events ---
    ev = d / "stream_events.jsonl"
    if ev.exists():
        rows = [json.loads(l) for l in ev.read_text(errors="replace").splitlines() if l.strip()]
        toks = [r for r in rows if r.get("delta")]
        if len(toks) >= 3:
            t_first = toks[0]["t_s"]
            t_last = toks[-1]["t_s"]
            n = len(toks)
            out["ttft_s"] = round(t_first, 3)
            out["n_delta_events"] = n
            if t_last > t_first:
                out["decode_tps_stream"] = round((n - 1) / (t_last - t_first), 3)
            # steady-state: last 60% of tokens
            k = int(n * 0.4)
            if n - k >= 3 and toks[-1]["t_s"] > toks[k]["t_s"]:
                out["decode_tps_regime"] = round((n - 1 - k) / (toks[-1]["t_s"] - toks[k]["t_s"]), 3)
        if out.get("elapsed_s") and out.get("completion_tokens"):
            out["overall_tps"] = round(out["completion_tokens"] / out["elapsed_s"], 3)
        elif out.get("elapsed_s") and toks:
            out["overall_tps"] = round(len(toks) / out["elapsed_s"], 3)

    # --- MiB/tok, copy_ms/noncopy_ms derivation ---
    ct = out.get("completion_tokens") or out.get("n_delta_events")
    if ct and out.get("zc_dma_ok_MiB") is not None:
        out["MiB_per_tok"] = round(out["zc_dma_ok_MiB"] / ct, 3)
    if out.get("decode_tps_regime") or out.get("decode_tps_stream"):
        tps = out.get("decode_tps_regime") or out.get("decode_tps_stream")
        out["ms_per_tok"] = round(1000.0 / tps, 2)
        # copy_ms/tok = DMA MiB per tok / H2D bandwidth (use register bw as proxy if runtime bw absent)
        bw = out.get("register_gib_s")  # GiB/s
        if bw and out.get("MiB_per_tok") is not None:
            copy_ms = (out["MiB_per_tok"] / 1024.0) / bw * 1000.0
            out["copy_ms_per_tok_est"] = round(copy_ms, 2)
            out["noncopy_ms_per_tok_est"] = round(out["ms_per_tok"] - copy_ms, 2)

    # --- peak VRAM ---
    gm = d / "gpu_mem.log"
    if gm.exists():
        used = []
        for line in gm.read_text(errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                used.append(int(parts[1]))
        if used:
            out["vram_used_peak_MiB"] = max(used)
            out["vram_used_last_MiB"] = used[-1]

    print(json.dumps(out, indent=2))
    (d / "metrics.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        main(a)
