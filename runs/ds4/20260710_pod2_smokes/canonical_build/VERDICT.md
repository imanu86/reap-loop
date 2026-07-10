# Build-test canonical series v2 (+0027) — POD2 2026-07-10

Base: clean `git clone https://github.com/antirez/ds4` @ **80ebbc3** (ds4.c md5 bf9a0b6f).
Applied: patches/ds4/canonical/ (19 patches, sorted) + 0027-rewind-exactness-harness. Build only.

## APPLY: all 20 patches apply CLEAN (git apply --check + git apply, each committed).
post-chain ds4.c md5 = `3de71ef4e5e2ec3359811bbbbfd766b9` (identical to the livetree post-0027 tree).

## BUILD: FAIL (BUILD_EXIT=2) — expected "missing declaration in siblings"

Exactly one error:
`ds4.c:10614:5: error: unknown type name 'ds4_gpu_async_read'`

ds4.c is byte-identical to the (buildable) livetree tree, so the gap is in the SIBLING files: the canonical
base 80ebbc3 sibling (ds4_cuda.c / header) does not define the `ds4_gpu_async_read` type that the patched
ds4.c references, and the canonical patch series does not add it. Not fixed per instructions.

## Verdict: switchover NOT unblocked (canonical series incomplete on sibling declarations).
