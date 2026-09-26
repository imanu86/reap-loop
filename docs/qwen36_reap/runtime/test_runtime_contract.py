"""CPU Python reference/static gates, NOT execution of the C++/GGML patch."""
import math
from pathlib import Path
import struct
import unittest

LAB = Path(r"D:\ds4_work\qwen36_reap_lab\source")

def f32(value):
    return struct.unpack('f', struct.pack('f', value))[0]

def reference(logits, keep=None):
    if len(logits) != 256 or any(not math.isfinite(x) for x in logits):
        raise ValueError('E/finite')
    if keep is not None and (len(keep) < 8 or len(keep) > 256 or len(set(keep)) != len(keep)
                             or any(type(x) is not int or x < 0 or x >= 256 for x in keep)):
        raise ValueError('keep/ID')
    masked = keep is not None and len(keep) != 256
    values = [x if not masked or i in keep else -math.inf for i, x in enumerate(logits)]
    exp = [f32(math.exp(x - max(values))) for x in values]
    denominator = math.fsum(exp)
    probabilities = [f32(x / denominator) for x in exp]
    ranks = values if masked else probabilities
    ids = sorted(range(256), key=lambda i: (-ranks[i], i))[:8]
    selected = [probabilities[i] for i in ids]
    total = max(math.fsum(selected), 6.103515625e-5)
    weights = [f32(w / total) for w in selected]
    return ids, weights

class Contract(unittest.TestCase):
    def test_identity_two_synthetic_fixtures(self):
        # Numeric fixtures only. Real web/DOM/MCP identity requires the runtime.
        fixtures = [[f32(math.sin(i * .7)) for i in range(256)],
                    [f32(math.cos(i * .17) + i / 300.) for i in range(256)]]
        for logits in fixtures:
            self.assertEqual(reference(logits), reference(logits, list(range(256))))

    def test_excluded_dominant_logit(self):
        logits = [10000.] + [float(i) for i in range(1, 256)]
        keep = list(range(1, 9))
        ids, weights = reference(logits, keep)
        self.assertEqual(set(ids), set(keep))
        self.assertAlmostEqual(sum(weights), 1., places=6)

    def test_underflow_tie_never_selects_excluded(self):
        logits = [-10000.] * 256
        logits[255] = 0.
        keep = list(range(248, 256))
        ids, weights = reference(logits, keep)
        self.assertEqual(set(ids), set(keep))
        self.assertEqual(sum(weights), 1.)

    def test_reject_small_keep(self):
        for n in range(8):
            with self.assertRaises(ValueError): reference([0.] * 256, list(range(n)))

    def test_reject_bad_ids_and_nonfinite(self):
        for keep in ([0] * 8, list(range(7)) + [256], list(range(7)) + [-1], list(range(7)) + [True]):
            with self.assertRaises(ValueError): reference([0.] * 256, keep)
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError): reference([value] + [0.] * 255)

    def test_source_graph_wiring(self):
        graph = (LAB / 'src/llama-graph.cpp').read_text(encoding='utf-8')
        context = (LAB / 'src/llama-context.cpp').read_text(encoding='utf-8')
        trace = (LAB / 'src/llama-reap.cpp').read_text(encoding='utf-8')
        self.assertLess(graph.index('logits = ggml_add(ctx0, logits, input->bias)'), graph.index('probs = ggml_soft_max(ctx0, logits)'))
        self.assertIn('if (reap.mask_enabled && reap.masked[il])', graph)
        self.assertIn('selection_probs = logits;', graph)
        self.assertLess(graph.index('weights = ggml_scale(ctx0, weights, w_scale)'), graph.index('"ffn_moe_weights_effective"'))
        self.assertIn('reap_trace->begin(ubatch, cparams.warmup)', context)
        self.assertIn('reap_trace->next_decode()', context)
        self.assertIn('{"record_type", "route"}', trace)
        self.assertIn('{"architecture", "qwen35moe"}', trace)
        self.assertIn('if (warmup) { s.active = true; return; }', trace)
        self.assertIn('_O_CREAT | _O_EXCL', trace)
        self.assertIn('~impl() { if (owns_guard) reap_trace_created = false; }', trace)
        self.assertNotIn('-1e9', graph)
        self.assertIn('s.previous(tensor, false, s.previous_data)', trace)

    def test_observer_does_not_split_routing_fusion(self):
        graph = (LAB / 'src/llama-graph.cpp').read_text(encoding='utf-8')
        trace = (LAB / 'src/llama-reap.cpp').read_text(encoding='utf-8')
        block = graph.split('if (!reap.trace_path.empty()) {', 1)[1].split('}', 1)[0]
        self.assertNotIn('ggml_cont', block)
        self.assertNotIn('weights =', block)
        self.assertIn('cb(weights, "ffn_moe_weights_effective", il)', block)
        self.assertIn('return upstream || (wanted && !is_ids);', trace)
        self.assertIn('s.pending_ids.fill(nullptr);', trace)
        self.assertIn('s.consume(s.pending_ids[layer], layer, true);', trace)
        self.assertIn('s.consume(tensor, layer, false);', trace)

if __name__ == '__main__':
    unittest.main(verbosity=2)
