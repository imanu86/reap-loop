# DS4 Nested Residual Sidecar v2

The sidecar is binary and self-describing.  JSON is optional receipt metadata
only; readers validate the fixed little-endian header and fixed-size records.

Header fields: `magic`, `version`, `header_bytes`, `source_size`,
`source_sha256`, `payload_bytes`, `payload_sha256`, `header_crc32`,
`record_count`, `record_size`.

In v2, `header_crc32` is calculated over the header with the CRC field zeroed
and the complete record table. This binds tensor offsets and geometry without
growing the fixed 112-byte header. V1 artifacts are rejected.

Each record covers one routed tensor `(layer, kind)` and stores source type
`16` (`IQ2_XXS`) or `10` (`Q2_K`), dimensions `(ncols, nrows, nexperts)`,
source absolute offset/bytes, residual file offset/bytes, `expert_stride`,
`kind_offset_within_expert`, `residual_expert_bytes`, and native/base/residual
block byte sizes.

Residual payload bytes are ascending layer-major, then expert-major.  For each
expert, the payload stores `gate_residual`, `up_residual`, and `down_residual`
contiguously in row, block order.  A cold expert can therefore be fetched with
one `pread` of `residual_expert_bytes` at `layer_start + expert * expert_stride`.

Records remain per tensor.  Their residual ranges are strided: the first chunk
for a tensor starts at `residual_offset`, and the same kind for expert `e` is at
`residual_offset + e * expert_stride`.

Validation is fail-closed for header/record-table CRC, source identity, payload SHA, bounds,
non-overlap, tensor geometry, interleaved gate/up/down layout coherence, and
byte-exact fixture reconstruction.
