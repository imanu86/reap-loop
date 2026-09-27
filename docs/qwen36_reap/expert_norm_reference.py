"""Pure, bounded APPROXIMATE_Q4_F32_REPLAY over caller-supplied buffers.

No file/capture/model reader, GPU, torch, rerouting, ranking, masks or inference server.
Input X MUST already include the post-attention RMSNorm and learned multiplier;
this module never normalizes it again. Caller provenance is not independently verified.

Q4_K decode follows gguf 0.19 and lab ggml-quants.c:1529 (256 values / 144 bytes).
Dense F32 matmul + SiLU is NOT native quant-CUDA identity: native kernels may quantize
activations to Q8_1 and fuse/reorder arithmetic. CPU ggml Q8_K also differs. This is
an approximate conditional expert-output-norm reference, not a quality approval.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
import sys

import numpy as np
from gguf.quants import Q4_K

LABEL = 'APPROXIMATE_Q4_F32_REPLAY'
BLOCK_VALUES = 256
BLOCK_BYTES = 144
MAX_MATRIX_BYTES = 64 * 1024 * 1024
MAX_WORK_BYTES = 256 * 1024 * 1024
MAX_TILE_TOKENS = 4096
GGUF_VERSION = version('gguf')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def positive_int(value, name):
    require(type(value) is int and value > 0, 'Invalid ' + name)
    return value


def reject_features(*, bias=None, scales=None, clamp=None, lora=None):
    for name, value in (('bias', bias), ('scales', scales), ('clamp', clamp), ('lora', lora)):
        require(value is None, 'Unsupported ' + name + '; refusing to ignore it')


def packed_view(data):
    require(isinstance(data, (bytes, memoryview, np.ndarray)), 'Packed data must be bytes or a read-only uint8 buffer')
    view = memoryview(data)
    require(view.readonly and view.c_contiguous and view.ndim == 1 and view.format == 'B',
            'Packed data must be immutable/read-only contiguous one-dimensional uint8 bytes')
    return view


def matrix_shape(shape):
    require(type(shape) is tuple and len(shape) == 2, 'GGML matrix shape must be (input, output)')
    width, rows = (positive_int(v, 'matrix dimension') for v in shape)
    require(width % BLOCK_VALUES == 0, 'Q4_K input width must be divisible by 256')
    require(width * rows * 4 <= MAX_MATRIX_BYTES, 'Matrix exceeds bounded F32 allocation limit')
    return width, rows


@dataclass(frozen=True)
class PackedQ4KMatrix:
    data: object
    ggml_shape: tuple
    quant_type: str = 'Q4_K'
    byte_order: str = 'little'

    def validate(self):
        require(self.quant_type == 'Q4_K' and type(self.quant_type) is str, 'Only explicit Q4_K tensor type is supported')
        require(self.byte_order == 'little' and sys.byteorder == 'little', 'Only little-endian Q4_K is supported')
        width, rows = matrix_shape(self.ggml_shape)
        view = packed_view(self.data)
        require(view.nbytes == rows * (width // BLOCK_VALUES) * BLOCK_BYTES,
                'Packed matrix byte count mismatch (truncation, trailing data or incorrect shape)')
        return view, width, rows


def slice_expert_q4_k(data, *, ggml_shape, expert_id, quant_type='Q4_K', byte_order='little'):
    """Zero-copy view from GGML [input, output, expert] bank; never decode the bank.

    Returned matrix decodes to NumPy [output, input]. Physical expert axis is outermost.
    Merged gate/up banks are intentionally not accepted by the replay API: caller must
    explicitly supply the separate gate/up matrices with their exact intended shapes.
    """
    require(type(ggml_shape) is tuple and len(ggml_shape) == 3, 'Expert bank shape must be (input, output, experts)')
    width, rows = matrix_shape(ggml_shape[:2])
    experts = positive_int(ggml_shape[2], 'expert dimension')
    require(type(expert_id) is int and 0 <= expert_id < experts, 'Expert ID outside bank')
    view = packed_view(data)
    size = rows * (width // BLOCK_VALUES) * BLOCK_BYTES
    require(view.nbytes == size * experts, 'Packed expert bank byte count mismatch')
    result = PackedQ4KMatrix(view[expert_id * size:(expert_id + 1) * size], (width, rows), quant_type, byte_order)
    result.validate()
    return result


def decode_q4_k_matrix(matrix, *, tile_rows=32):
    """One expert matrix to C-contiguous F32 [output,input], with bounded row staging."""
    require(type(matrix) is PackedQ4KMatrix, 'Expected PackedQ4KMatrix')
    positive_int(tile_rows, 'decode tile rows')
    require(tile_rows <= MAX_TILE_TOKENS, 'Decode tile too large')
    view, width, rows = matrix.validate()
    row_bytes = (width // BLOCK_VALUES) * BLOCK_BYTES
    require(rows * width * 4 + tile_rows * width * 32 <= MAX_WORK_BYTES, 'Decode staging estimate exceeds bound')
    packed = np.frombuffer(view, dtype=np.uint8).reshape(rows, row_bytes)
    output = np.empty((rows, width), dtype=np.float32)
    for start in range(0, rows, tile_rows):
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            decoded = Q4_K.dequantize_rows(packed[start:start + tile_rows])
        require(decoded.dtype == np.float32 and decoded.shape == (min(tile_rows, rows - start), width),
                'Unexpected gguf Q4_K output type/shape')
        require(np.isfinite(decoded).all(), 'Nonfinite dequantized matrix')
        output[start:start + len(decoded)] = decoded
    return output


@dataclass(frozen=True)
class PackedExpert:
    gate: PackedQ4KMatrix
    up: PackedQ4KMatrix
    down: PackedQ4KMatrix

    def validate(self, width):
        require(all(type(m) is PackedQ4KMatrix for m in (self.gate, self.up, self.down)), 'Invalid expert matrix object')
        for matrix in (self.gate, self.up, self.down):
            matrix.validate()
        require(self.gate.ggml_shape == self.up.ggml_shape, 'Gate/up shapes differ')
        require(self.gate.ggml_shape[0] == width, 'Postnorm X width differs from gate/up input')
        hidden = self.gate.ggml_shape[1]
        require(self.down.ggml_shape == (hidden, width), 'Down must have GGML shape (hidden,input_width)')
        return hidden


def f32_array(value, name, ndim=2):
    require(type(value) is np.ndarray and value.dtype == np.dtype('float32') and value.ndim == ndim,
            name + ' must be an explicit native-endian float32 ndarray')


def finite(value, name):
    require(np.isfinite(value).all(), 'Nonfinite ' + name)


def expert_outputs(postnorm_x, gate, up, down, *, bias=None, scales=None, clamp=None, lora=None):
    """One bounded F32 tile: Y = down(SiLU(gate(X)) * up(X)); no route weight here.

    Matrix orientation is NumPy gate/up[H,D], down[D,H]. No shared expert/residual.
    The size guard covers owned array estimates, not opaque BLAS thread workspaces.
    """
    reject_features(bias=bias, scales=scales, clamp=clamp, lora=lora)
    for name, array in (('postnorm_x', postnorm_x), ('gate', gate), ('up', up), ('down', down)):
        f32_array(array, name)
    batch, width = postnorm_x.shape
    hidden = gate.shape[0]
    require(batch > 0 and width > 0 and hidden > 0 and batch <= MAX_TILE_TOKENS, 'Invalid/oversized expert tile')
    require(gate.shape == up.shape == (hidden, width) and down.shape == (width, hidden), 'MLP matrix shape mismatch')
    require(12 * width * hidden + 64 * batch * (width + hidden) <= MAX_WORK_BYTES, 'Expert replay workspace estimate exceeds bound')
    for name, array in (('postnorm_x', postnorm_x), ('gate', gate), ('up', up), ('down', down)):
        # Caller supplies bounded weights/one tile here; no whole-capture boolean allocation.
        finite(array, name)
    return _expert_outputs_validated(postnorm_x, gate, up, down)


def _expert_outputs_validated(postnorm_x, gate, up, down):
    """Private core: caller validated shapes/types/finite X+weights and workspace.

    Decoder verifies every weight row once. Replay must not rescan those unchanged
    matrices for every selected-row tile. All arithmetic intermediates still checked.
    """
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        g = np.matmul(postnorm_x, gate.T)
        u = np.matmul(postnorm_x, up.T)
        finite(g, 'gate projection')
        finite(u, 'up projection')
        # Stable F32 SiLU: avoid exp(-large_negative) overflow without clamping values.
        sigmoid = np.empty_like(g)
        positive = g >= 0
        sigmoid[positive] = 1 / (1 + np.exp(-g[positive]))
        exp_negative = np.exp(g[~positive])
        sigmoid[~positive] = exp_negative / (1 + exp_negative)
        z = (g * sigmoid) * u
        finite(z, 'SiLU-times-up')
        y = np.matmul(z, down.T)
    finite(y, 'expert output')
    return y


def conditional_norm_scores(postnorm_x, selected_ids, gates, experts, *, expert_count=256, top_k=8,
                            tile_tokens=128, bias=None, scales=None, clamp=None, lora=None):
    """One layer, supplied postnorm X[N,D] + IDs/gates[N,K], no file access.

    Score[e] = sum_selected(g * L2(Y_e(X))) / N_selected; unobserved score=0.
    Count includes selected zero-gate occurrences. Supplied gates are used verbatim,
    never renormalized. All selected experts are replayed, even when their gate is zero.
    For each expert, bounded original-ID scans fill a <=tile_tokens index/gate queue
    in original token order; GEMM flushes full SELECTED-row tiles, then one tail.
    One expert's weights and at most tile_tokens gathered rows are live at a time;
    inputs are caller-owned. No global N*K index list, duplicated activations or full-model decode.
    """
    reject_features(bias=bias, scales=scales, clamp=clamp, lora=lora)
    f32_array(postnorm_x, 'postnorm_x')
    f32_array(gates, 'gates')
    positive_int(expert_count, 'expert_count')
    positive_int(top_k, 'top_k')
    positive_int(tile_tokens, 'tile_tokens')
    require(top_k <= expert_count <= 256 and top_k <= 8, 'Reference bounded to at most256 experts/top8')
    require(tile_tokens <= MAX_TILE_TOKENS, 'Token tile too large')
    require(type(selected_ids) is np.ndarray and selected_ids.dtype in (np.dtype('int32'), np.dtype('int64')) and selected_ids.ndim == 2,
            'selected_ids must be an int32/int64 ndarray')
    tokens, width = postnorm_x.shape
    require(tokens > 0 and width > 0 and selected_ids.shape == gates.shape == (tokens, top_k), 'Input token/top-k shapes differ')
    require(isinstance(experts, Mapping), 'experts must map IDs to PackedExpert objects')
    require(all(type(e) is int and 0 <= e < expert_count and type(w) is PackedExpert for e, w in experts.items()),
            'Invalid supplied expert mapping')
    selected = set()
    for start in range(0, tokens, tile_tokens):
        stop = min(start + tile_tokens, tokens)
        ids, weights = selected_ids[start:stop], gates[start:stop]
        finite(postnorm_x[start:stop], 'postnorm X')
        finite(weights, 'route gates')
        require(np.all((ids >= 0) & (ids < expert_count)), 'Selected expert ID outside range')
        require(np.all(weights >= 0), 'Negative route gate')
        require(np.all(np.diff(np.sort(ids, axis=1), axis=1) != 0), 'Duplicate selected expert in token')
        selected.update(int(e) for e in np.unique(ids))
    require(selected <= experts.keys(), 'Missing selected expert weights')
    queue_bytes = tile_tokens * (np.dtype('int64').itemsize + np.dtype('float32').itemsize)
    for expert_id, packed in experts.items():
        hidden = packed.validate(width)
        require(12 * width * hidden + max(64 * tile_tokens * (width + hidden), 1024 * max(width, hidden)) + queue_bytes <= MAX_WORK_BYTES,
                'Expert replay workspace estimate exceeds bound')
    totals = np.zeros(expert_count, dtype=np.float64)
    counts = np.zeros(expert_count, dtype=np.int64)
    row_queue = np.empty(tile_tokens, dtype=np.int64)
    gate_queue = np.empty(tile_tokens, dtype=np.float32)
    for expert_id in sorted(selected):
        packed = experts[expert_id]
        gate = decode_q4_k_matrix(packed.gate)
        up = decode_q4_k_matrix(packed.up)
        down = decode_q4_k_matrix(packed.down)
        queued = 0

        def flush(size):
            # X was checked during the initial bounded scan; decoded weights were
            # checked once by the decoder. No repeated full weight validation here.
            x = postnorm_x[row_queue[:size]]
            y = _expert_outputs_validated(x, gate, up, down)
            # F64 squares/sum/sqrt are declared offline metric semantics, not CUDA parity.
            squares = y.astype(np.float64)
            np.square(squares, out=squares)
            norms = np.sqrt(np.sum(squares, axis=1, dtype=np.float64))
            weighted = gate_queue[:size].astype(np.float64) * norms
            finite(weighted, 'weighted expert output norm')
            # Preserve original per-expert token order, including zero-gate occurrences.
            for value in weighted:
                totals[expert_id] += value
            counts[expert_id] += size

        for start in range(0, tokens, tile_tokens):
            stop = min(start + tile_tokens, tokens)
            # At most one match/token because duplicate IDs were rejected above.
            rows, slots = np.nonzero(selected_ids[start:stop] == expert_id)
            offset = 0
            while offset < rows.size:
                take = min(tile_tokens - queued, rows.size - offset)
                source = slice(offset, offset + take)
                destination = slice(queued, queued + take)
                row_queue[destination] = start + rows[source]
                gate_queue[destination] = gates[start:stop][rows[source], slots[source]]
                queued += take
                offset += take
                if queued == tile_tokens:
                    flush(queued)
                    queued = 0
        if queued:
            flush(queued)
        del gate, up, down
    finite(totals, 'conditional numerator')
    scores = np.divide(totals, counts, out=np.zeros(expert_count, dtype=np.float64), where=counts != 0)
    finite(scores, 'conditional scores')
    return {'scores': scores, 'weighted_norm_sums': totals, 'selected_counts': counts,
            'metadata': {'schema_version': 1, 'label': LABEL, 'native_identity': False,
                'quality_approval': False, 'model_provenance': 'supplied_buffers_not_independently_verified',
                'formula': 'sum_selected(g * L2(down(SiLU(gate(X)) * up(X)))) / N_selected',
                'input': 'caller_supplied_postnorm_F32_no_second_normalization',
                'output_before_route_gate_shared_expert_and_residual': True,
                'route_gates': 'supplied_F32_verbatim_no_renormalization',
                'zero_gate_occurrences_counted': True, 'unobserved_score': 0,
                'matmul_activation_dtype': 'float32', 'norm_and_accumulator_dtype': 'float64',
                'quant_type': 'Q4_K', 'endianness': 'little', 'tokens': tokens, 'input_width': width,
                'expert_count': expert_count, 'top_k': top_k, 'tile_tokens': tile_tokens,
                'batching': 'per_expert_selected_row_queue_in_original_token_order',
                'selected_row_queue_bytes': queue_bytes,
                'max_owned_workspace_estimate_bytes': MAX_WORK_BYTES,
                'workspace_excludes': 'caller_owned_buffers_and_BLAS_internal_workspaces',
                'numpy_version': np.__version__, 'gguf_version': GGUF_VERSION,
                'native_C_dequant_oracle_verified': False}}
