# Scheduler R=1 — simulazione su cicli reali

u(k) = 6.30·k^0.668 (fit su trace misurata); L_routed=40; t_fix=10% di T1; δ_draft=10% di T1 per token draftato

## TUTTI (n=585)
| policy | tok/step | IO per token | speedup IO vs nospec |
|---|---|---|---|
| nospec | 1.00 | 302 | 1.00× |
| fixed-1 | 1.88 | 160 | 1.88× |
| fixed-2 | 2.64 | 180 | 1.68× |
| fixed-3 | 3.29 | 190 | 1.59× |
| fixed-4 | 3.84 | 198 | 1.52× |
| fixed-5 | 4.29 | 207 | 1.46× |
| dynamic-STS | 2.22 | 159 | 1.91× |
| oracle | 4.29 | 155 | 1.96× |

## chat (n=195)
| policy | tok/step | IO per token | speedup IO vs nospec |
|---|---|---|---|
| nospec | 1.00 | 302 | 1.00× |
| fixed-1 | 1.71 | 176 | 1.71× |
| fixed-2 | 2.15 | 221 | 1.36× |
| fixed-3 | 2.43 | 258 | 1.17× |
| fixed-4 | 2.61 | 292 | 1.04× |
| fixed-5 | 2.70 | 329 | 0.92× |
| dynamic-STS | 1.71 | 176 | 1.71× |
| oracle | 2.70 | 169 | 1.79× |

## code (n=195)
| policy | tok/step | IO per token | speedup IO vs nospec |
|---|---|---|---|
| nospec | 1.00 | 302 | 1.00× |
| fixed-1 | 1.98 | 153 | 1.98× |
| fixed-2 | 2.91 | 164 | 1.85× |
| fixed-3 | 3.77 | 166 | 1.82× |
| fixed-4 | 4.54 | 168 | 1.80× |
| fixed-5 | 5.18 | 172 | 1.76× |
| dynamic-STS | 2.34 | 152 | 1.99× |
| oracle | 5.18 | 150 | 2.01× |

## math (n=195)
| policy | tok/step | IO per token | speedup IO vs nospec |
|---|---|---|---|
| nospec | 1.00 | 302 | 1.00× |
| fixed-1 | 1.96 | 154 | 1.96× |
| fixed-2 | 2.86 | 167 | 1.81× |
| fixed-3 | 3.67 | 171 | 1.77× |
| fixed-4 | 4.38 | 174 | 1.74× |
| fixed-5 | 4.99 | 178 | 1.70× |
| dynamic-STS | 2.59 | 153 | 1.98× |
| oracle | 4.99 | 151 | 2.00× |
