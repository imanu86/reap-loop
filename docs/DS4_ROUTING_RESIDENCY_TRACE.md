# DS4 Routing Residency Trace

Patch proposal: `patches/ds4/0017-spex-routing-trace-residency.patch`.

`DS4_SPEX_TRACE_ROUTING_RESIDENCY=1` extends `DS4_SPEX_TRACE_ROUTING` with
`storage0..storage5`.  With the env var unset or `0`, the routing CSV header
and rows remain unchanged.  If `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`, residency
columns are appended after `w0..w5`; otherwise they are appended after `e0..e5`.

Residency is determined at the same source point where decode selected expert
ids have already been read back for SPEX tracing: `metal_graph_spex_trace_selected`
is called from the selected-id readback/async-finish paths with the current
`ds4_gpu_stream_expert_table`.  CUDA then answers from host-side runtime
metadata only:

- streaming expert cache slot table match: `vram`
- device-owned model range/cache: `vram`
- registered/HMM/direct mapped host access: `cpu ram`
- SSD streaming fd-backed model source with no matching resident/cache metadata:
  `ssd offload`
- invalid, mixed, or unavailable metadata: `unknown`

Limitations: `cpu ram` vs `ssd offload` is only as precise as DS4's current
runtime metadata.  If the OS page cache holds mmap pages for an fd-backed model
but DS4 has not registered or cached that range, the patch conservatively reports
`ssd offload`.  It performs no GPU payload readback and does not probe OS page
residency.

Sample headers:

```csv
pos,layer,n,e0,e1,e2,e3,e4,e5,storage0,storage1,storage2,storage3,storage4,storage5
pos,layer,n,e0,e1,e2,e3,e4,e5,w0,w1,w2,w3,w4,w5,storage0,storage1,storage2,storage3,storage4,storage5
```

Sample row:

```csv
128,17,6,42,11,7,93,2,65,0.391233,0.18211,0.14308,0.12005,0.091,0.0725,vram,cpu ram,ssd offload,unknown,vram,vram
```
