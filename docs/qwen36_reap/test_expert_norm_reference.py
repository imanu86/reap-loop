"""Invented buffers only. No files/model/capture/GPU, torch, ranking or mask generation.
Q4_K expected values below are hand-packed scalar checks, NOT a native-C oracle.
Tile/MLP numerical tolerance is predeclared, not bit-identity or CUDA equivalence.
"""
import math
import struct
import unittest
from unittest.mock import patch

import numpy as np
import expert_norm_reference as ref

ATOL = 1e-7
RTOL = 1e-5


def block(d=1.0, dmin=0.0, scales=None, mins=None, quants=None):
    """Independent hand-packing of the documented Q4_K 144-byte block."""
    scales = [1] * 8 if scales is None else scales
    mins = [0] * 8 if mins is None else mins
    quants = [[1] * 32 for _ in range(8)] if quants is None else quants
    packed = bytearray(struct.pack('<ee', d, dmin))
    s = [0] * 12
    for j in range(4):
        s[j] = scales[j] | ((scales[j + 4] >> 4) << 6)
        s[j + 4] = mins[j] | ((mins[j + 4] >> 4) << 6)
        s[j + 8] = (scales[j + 4] & 15) | ((mins[j + 4] & 15) << 4)
    packed.extend(s)
    for pair in range(4):
        packed.extend(quants[2 * pair][i] | (quants[2 * pair + 1][i] << 4) for i in range(32))
    assert len(packed) == 144
    return bytes(packed)


def matrix(value, width=256, rows=256):
    return ref.PackedQ4KMatrix(block(d=value) * (rows * (width // 256)), (width, rows))


def expert(multiplier=1.0):
    return ref.PackedExpert(matrix(multiplier / 256), matrix(1 / 512), matrix(1 / 1024))


class QuantBlockTests(unittest.TestCase):
    def decode(self, raw):
        return ref.decode_q4_k_matrix(ref.PackedQ4KMatrix(raw, (256, 1)))[0]

    def test_zero_and_max_nibbles(self):
        np.testing.assert_array_equal(self.decode(bytes(144)), np.zeros(256, dtype=np.float32))
        raw = block(d=1, dmin=1, scales=[63] * 8, mins=[63] * 8, quants=[[15] * 32 for _ in range(8)])
        np.testing.assert_array_equal(self.decode(raw), np.full(256, 882, dtype=np.float32))

    def test_high_six_bits_and_nibble_order(self):
        scales = [1, 2, 3, 4, 17, 33, 49, 63]
        mins = [5, 6, 7, 8, 18, 34, 50, 62]
        quants = [[(j + i) % 16 for i in range(32)] for j in range(8)]
        d, dmin = 0.25, 0.125
        expected = []
        for j in range(8):
            for q in quants[j]:
                expected.append(np.float32(np.float32(np.float32(d) * scales[j]) * q) - np.float32(dmin * mins[j]))
        np.testing.assert_array_equal(self.decode(block(d, dmin, scales, mins, quants)), np.asarray(expected, dtype=np.float32))

    def test_min_offset_negative_values_and_fp16_subnormal(self):
        negative = block(d=1, dmin=0.5, mins=[63] * 8, quants=[[0] * 32 for _ in range(8)])
        np.testing.assert_array_equal(self.decode(negative), np.full(256, -31.5, dtype=np.float32))
        np.testing.assert_array_equal(self.decode(block(d=2 ** -24)), np.full(256, 2 ** -24, dtype=np.float32))

    def test_multiple_blocks_row_layout_and_expert_axis(self):
        raw = b''.join(block(d=i) for i in range(1, 7))
        result = ref.decode_q4_k_matrix(ref.PackedQ4KMatrix(raw, (512, 3)), tile_rows=1)
        self.assertEqual(result.shape, (3, 512))
        for row in range(3):
            np.testing.assert_array_equal(result[row, :256], np.full(256, row * 2 + 1, dtype=np.float32))
            np.testing.assert_array_equal(result[row, 256:], np.full(256, row * 2 + 2, dtype=np.float32))
        bank = block(d=1) + block(d=2) + block(d=3)
        for e in range(3):
            part = ref.slice_expert_q4_k(bank, ggml_shape=(256, 1, 3), expert_id=e)
            self.assertIsInstance(part.data, memoryview)
            self.assertEqual(part.data.nbytes, 144)
            np.testing.assert_array_equal(ref.decode_q4_k_matrix(part), np.full((1, 256), e + 1, dtype=np.float32))

    def test_block_truncation_trailing_bytes_shape_and_type_fail_closed(self):
        raw = block()
        cases = [ref.PackedQ4KMatrix(raw[:-1], (256, 1)), ref.PackedQ4KMatrix(raw + b'\0', (256, 1)),
                 ref.PackedQ4KMatrix(raw, (255, 1)), ref.PackedQ4KMatrix(raw, (256, 2)),
                 ref.PackedQ4KMatrix(raw, (256,)), ref.PackedQ4KMatrix(raw, (256, 1, 1)),
                 ref.PackedQ4KMatrix(raw, [256, 1]), ref.PackedQ4KMatrix(raw, (256, True)),
                 ref.PackedQ4KMatrix(raw, (256, 1), quant_type='Q4_K_M'),
                 ref.PackedQ4KMatrix(raw, (256, 1), quant_type='Q6_K'),
                 ref.PackedQ4KMatrix(raw, (256, 1), quant_type=12),
                 ref.PackedQ4KMatrix(raw, (256, 1), byte_order='big')]
        for value in cases:
            with self.subTest(value=value.ggml_shape, qtype=value.quant_type):
                with self.assertRaises(ValueError):
                    ref.decode_q4_k_matrix(value)
        for shape, eid in [((256, 1, 3), -1), ((256, 1, 3), 3), ((256, 1), 0), ((256, 1, True), 0)]:
            with self.assertRaises(ValueError):
                ref.slice_expert_q4_k(raw * 3, ggml_shape=shape, expert_id=eid)
        with self.assertRaises(ValueError):
            ref.slice_expert_q4_k(raw * 3 + b'x', ggml_shape=(256, 1, 3), expert_id=0)

    def test_mutable_noncontiguous_and_wrong_dtype_buffers_rejected(self):
        for data in (bytearray(block()), np.zeros(144, dtype=np.float32), np.zeros((1, 144), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                ref.decode_q4_k_matrix(ref.PackedQ4KMatrix(data, (256, 1)))
        array = np.zeros(288, dtype=np.uint8)
        array.flags.writeable = False
        with self.assertRaises(ValueError):
            ref.decode_q4_k_matrix(ref.PackedQ4KMatrix(array[::2], (256, 1)))
        array = np.frombuffer(block(), dtype=np.uint8)
        np.testing.assert_array_equal(ref.decode_q4_k_matrix(ref.PackedQ4KMatrix(array, (256, 1))), np.ones((1, 256), dtype=np.float32))

    def test_nonfinite_scales_and_allocation_limits_rejected(self):
        for value in (float('nan'), float('inf')):
            with self.assertRaises((ValueError, FloatingPointError)):
                self.decode(block(d=value))
        with self.assertRaisesRegex(ValueError, 'allocation limit'):
            ref.decode_q4_k_matrix(ref.PackedQ4KMatrix(b'', (256, 1000000)))
        with self.assertRaises(ValueError):
            ref.decode_q4_k_matrix(ref.PackedQ4KMatrix(block(), (256, 1)), tile_rows=0)


class MLPReferenceTests(unittest.TestCase):
    def test_scalar_mlp_reference_generic_invented_geometry(self):
        x = np.array([[0.3, -0.7, 1.2], [-0.8, 0.9, 0.1]], dtype=np.float32)
        gate = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.2]], dtype=np.float32)
        up = np.array([[0.1, 0.7, -0.2], [0.5, -0.4, 0.3]], dtype=np.float32)
        down = np.array([[0.2, -0.5], [0.7, 0.3], [-0.2, 0.8]], dtype=np.float32)
        expected = []
        for row in x:
            hidden = []
            for j in range(2):
                g = math.fsum(float(a) * float(b) for a, b in zip(row, gate[j]))
                u = math.fsum(float(a) * float(b) for a, b in zip(row, up[j]))
                hidden.append((g / (1 + math.exp(-g))) * u)
            expected.append([math.fsum(float(w) * v for w, v in zip(weights, hidden)) for weights in down])
        originals = [a.copy() for a in (x, gate, up, down)]
        actual = ref.expert_outputs(x, gate, up, down)
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
        for original, after in zip(originals, (x, gate, up, down)):
            np.testing.assert_array_equal(original, after)
        self.assertEqual(actual.dtype, np.float32)

    def test_stable_silu_extremes_and_no_double_normalization(self):
        x = np.array([[-1000, 1000], [0, 0]], dtype=np.float32)
        gate = np.eye(2, dtype=np.float32)
        up = np.eye(2, dtype=np.float32)
        down = np.eye(2, dtype=np.float32)
        actual = ref.expert_outputs(x, gate, up, down)
        np.testing.assert_allclose(actual, [[0, 1000000], [0, 0]], atol=ATOL, rtol=RTOL)
        x = np.array([[1]], dtype=np.float32)
        one = np.ones((1, 1), dtype=np.float32)
        y1, y2 = ref.expert_outputs(x, one, one, one), ref.expert_outputs(x * 2, one, one, one)
        self.assertGreater(float(y2[0, 0]), float(y1[0, 0]) * 4)  # RMS-normalizing X would erase this scale change

    def test_unsupported_features_wrong_shapes_dtypes_and_nonfinite_fail(self):
        x, weights = np.ones((1, 2), dtype=np.float32), np.eye(2, dtype=np.float32)
        for feature in ('bias', 'scales', 'clamp', 'lora'):
            with self.assertRaisesRegex(ValueError, 'Unsupported ' + feature):
                ref.expert_outputs(x, weights, weights, weights, **{feature: 0})
        with self.assertRaises(ValueError):
            ref.expert_outputs(x.astype(np.float64), weights, weights, weights)
        with self.assertRaises(ValueError):
            ref.expert_outputs(x, weights, weights, np.ones((3, 2), dtype=np.float32))
        bad = weights.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            ref.expert_outputs(x, bad, weights, weights)
        with self.assertRaises((ValueError, FloatingPointError)):
            ref.expert_outputs(x * np.finfo(np.float32).max, weights * 2, weights, weights)


class ConditionalReferenceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.zeros((4, 256), dtype=np.float32)
        self.x[:, 0] = [1, 2, 4, 8]
        self.ids = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int32)
        self.gates = np.array([[0.0, 0.75], [0.25, 0.5], [0.5, 0.1], [0.3, 0.0]], dtype=np.float32)
        self.experts = {0: expert(), 1: expert(2)}

    def run_reference(self, **kwargs):
        return ref.conditional_norm_scores(self.x, self.ids, self.gates, self.experts,
                                           expert_count=3, top_k=2, **kwargs)

    def test_conditional_formula_zero_gates_unobserved_and_metadata(self):
        result = self.run_reference(tile_tokens=2)
        totals = np.zeros(3, dtype=np.float64)
        for row, ids, gates in zip(self.x, self.ids, self.gates):
            sx = math.fsum(float(v) for v in row)
            for e, g in zip(ids, gates):
                a = sx * ((int(e) + 1) / 256)
                u = sx / 512
                # 256 identical hidden entries, each down coefficient1/1024;
                # 256 identical output entries, so L2 =16 * absolute scalar output.
                y = (a / (1 + math.exp(-a))) * u / 4
                totals[e] += float(g) * abs(y) * 16
        np.testing.assert_allclose(result['weighted_norm_sums'], totals, atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(result['selected_counts'], [4, 4, 0])
        np.testing.assert_allclose(result['scores'], totals / [4, 4, 1], atol=ATOL, rtol=RTOL)
        m = result['metadata']
        self.assertEqual(m['label'], 'APPROXIMATE_Q4_F32_REPLAY')
        self.assertFalse(m['native_identity'])
        self.assertFalse(m['quality_approval'])
        self.assertFalse(m['native_C_dequant_oracle_verified'])
        self.assertEqual(m['norm_and_accumulator_dtype'], 'float64')
        self.assertEqual(m['model_provenance'], 'supplied_buffers_not_independently_verified')
        self.assertNotIn('model_sha256', m)  # invented fixture must not impersonate actual model
        self.assertEqual(result['scores'].dtype, np.float64)

    def test_tile_agreement_not_bit_identity_and_inputs_unchanged(self):
        originals = [a.copy() for a in (self.x, self.ids, self.gates)]
        one = self.run_reference(tile_tokens=1)
        four = self.run_reference(tile_tokens=4)
        np.testing.assert_allclose(one['scores'], four['scores'], atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(one['weighted_norm_sums'], four['weighted_norm_sums'], atol=ATOL, rtol=RTOL)
        np.testing.assert_array_equal(one['selected_counts'], four['selected_counts'])
        for original, current in zip(originals, (self.x, self.ids, self.gates)):
            np.testing.assert_array_equal(original, current)
        with patch.object(ref, 'decode_q4_k_matrix', wraps=ref.decode_q4_k_matrix) as decoder:
            self.run_reference(tile_tokens=1)
            self.assertEqual(decoder.call_count, 6)  # three matrices once per observed expert, not once/token

    def test_additive_chunk_merge_not_mean_of_chunk_means(self):
        whole = self.run_reference(tile_tokens=2)
        parts = [ref.conditional_norm_scores(self.x[a:b], self.ids[a:b], self.gates[a:b], self.experts,
                                             expert_count=3, top_k=2, tile_tokens=2) for a, b in ((0, 1), (1, 4))]
        sums = sum(p['weighted_norm_sums'] for p in parts)
        counts = sum(p['selected_counts'] for p in parts)
        merged = np.divide(sums, counts, out=np.zeros(3), where=counts != 0)
        np.testing.assert_allclose(merged, whole['scores'], atol=ATOL, rtol=RTOL)
        naive = sum(p['scores'] for p in parts) / 2
        self.assertGreater(float(np.max(np.abs(naive - whole['scores']))), ATOL)

    def test_all_zero_gate_occurrences_still_counted(self):
        self.gates.fill(0)
        result = self.run_reference()
        np.testing.assert_array_equal(result['selected_counts'], [4, 4, 0])
        np.testing.assert_array_equal(result['weighted_norm_sums'], [0, 0, 0])
        np.testing.assert_array_equal(result['scores'], [0, 0, 0])

    def test_invalid_id_shape_gate_input_and_missing_weights_rejected(self):
        for ids in (self.ids.astype(np.float32), self.ids[:, :1], np.full((4, 2), 0, dtype=np.int32),
                    np.full((4, 2), -1, dtype=np.int32), np.full((4, 2), 3, dtype=np.int32)):
            with self.assertRaises(ValueError):
                ref.conditional_norm_scores(self.x, ids, self.gates, self.experts, expert_count=3, top_k=2)
        for value in (np.nan, np.inf, -0.1):
            gates = self.gates.copy()
            gates[0, 0] = value
            with self.assertRaises(ValueError):
                ref.conditional_norm_scores(self.x, self.ids, gates, self.experts, expert_count=3, top_k=2)
        with self.assertRaisesRegex(ValueError, 'Missing selected'):
            ref.conditional_norm_scores(self.x, self.ids, self.gates, {0: self.experts[0]}, expert_count=3, top_k=2)
        bad_x = self.x.copy()
        bad_x[-1, -1] = np.inf
        with self.assertRaises(ValueError):
            ref.conditional_norm_scores(bad_x, self.ids, self.gates, self.experts, expert_count=3, top_k=2)

    def test_bad_expert_orientation_features_and_workspace_fail_before_decode(self):
        wrong = ref.PackedExpert(matrix(0.01), matrix(0.01), matrix(0.01, width=256, rows=512))
        with self.assertRaisesRegex(ValueError, 'Down must'):
            ref.conditional_norm_scores(self.x, self.ids, self.gates, {0: wrong, 1: self.experts[1]}, expert_count=3, top_k=2)
        for feature in ('bias', 'scales', 'clamp', 'lora'):
            with self.assertRaises(ValueError):
                self.run_reference(**{feature: {}})
        with patch.object(ref, 'MAX_WORK_BYTES', 1024):
            with patch.object(ref, 'decode_q4_k_matrix', side_effect=AssertionError('must reject before allocating')):
                with self.assertRaisesRegex(ValueError, 'workspace'):
                    self.run_reference()


if __name__ == '__main__':
    unittest.main()
