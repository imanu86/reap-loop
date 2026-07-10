# Smoke 0028 (token sidecar DS4_SPEX_TRACE_TOKENS) — POD2 2026-07-10

Binary: livetree base(771a39a8)+0020+0021+0026+0027+**0028** (ds4.c md5 62ed2e71), `make cuda sm_86`, 0 warnings.
Config: coffee prompt, PACE W50 K23 static, `--ssd-streaming --ssd-streaming-cache-experts 1024`, greedy temp0, -n 160.
Envs: DS4_SPEX_TRACE_ROUTING=route.csv DS4_SPEX_TRACE_ROUTING_WEIGHTS=1 DS4_SPEX_TRACE_TOKENS=tokens.csv.

## VERDICT: PASS (a, b, c)

- **(a)** tokens.csv exists, header `pos,token_id,piece`, 160 rows = 1 per generated token (-n 160).
  Python csv.reader parses all 161 lines, every row 3 columns.
- **(b)** pos alignment: tokens.csv pos 218–377, route.csv pos 218–376 — same start, contiguous; 1:1 join on the
  159 generated positions that have a router forward pass. The final token (pos 377) has no routing row because
  generation stopped after it (no subsequent forward pass) — expected, not a gap.
- **(c)** quoting valid: 5 rows carry `""`-escaped inner quotes, special pieces (```` ``` ````, `\n`, `</`) quoted
  correctly, no broken rows. UTF-8 pieces are the real coffee-shop HTML tokens (`Bean`, ` Brew`, `<html`, `Menu`, …).

Scope scene built from these CSVs and committed to `scope/data/20260710_token_exact/` (see job 6).
Files: tokens.csv, route.csv, gen.txt.
