"""Synthetic files only; no dependencies, model, inference, GPU or network.
Parent runs: python -B -m unittest discover -s docs/qwen36_reap/gate -p 'test_*.py' -v
"""
import copy
import json
import math
from pathlib import Path
import re
import struct
import tempfile
import unittest
from unittest.mock import patch

import compare_logits as gate

HERE = Path(__file__).resolve().parent


class ComparatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic-", dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def artifact(self, name, rows=None, tokens=None, prompt_ids=None, mode="greedy"):
        rows = rows or [[0.0, 3.0, -1.0], [2.0, 0.0, -1.0]]
        vocab = len(rows[0])
        tops = [max(range(vocab), key=lambda i: row[i] if math.isfinite(row[i]) else -math.inf) for row in rows]
        ids = tokens if tokens is not None else tops[:]
        prompt_ids = prompt_ids or [0, 1]
        m = {"schema_version": 1, "complete": True, "model_sha256": gate.MODEL_SHA,
             "dtype": "float32", "endianness": "little", "shape": [len(rows), vocab],
             "architecture": "qwen35moe", "layer_count": 40, "expert_count": 256, "top_k": 8,
             "mode": mode, "prompt_mode": "raw_utf8", "prompt_utf8": "Caffè\n", "prompt_bytes": 7,
             "prompt_token_ids": prompt_ids, "token_ids": ids, "greedy_token_ids": tops,
             "warmup": False, "target_only": True, "fixed_count_including_eog": True,
             "add_special": True, "parse_special": False, "settings": {"n_ctx_actual": 4096}, "environment": {},
             "steps": [{"step": i, "phase": "last_prefill" if i == 0 else "incremental_decode",
                        "prefix_length": len(prompt_ids) + i, "logits_offset_bytes": i * vocab * 4,
                        "token_id": ids[i], "greedy_token_id": tops[i]} for i in range(len(rows))]}
        prefix = self.root / name
        with Path(str(prefix) + ".json").open("x", encoding="utf-8") as handle:
            json.dump(m, handle)
        with Path(str(prefix) + ".logits.f32").open("xb") as handle:
            for row in rows:
                handle.write(struct.pack("<" + "f" * vocab, *row))
        return prefix

    def change(self, prefix, edit):
        path = Path(str(prefix) + ".json")
        data = json.loads(path.read_text(encoding="utf-8"))
        edit(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_identical_full_float32(self):
        result = gate.compare(self.artifact("a"), self.artifact("b"))
        self.assertTrue(result["pass"])
        self.assertTrue(result["bit_identical"])
        self.assertEqual(result["maxabs"], 0.0)
        self.assertEqual(result["compared_steps"], 2)

    def test_tolerance_defined_before_test(self):
        self.assertEqual((gate.ATOL, gate.RTOL), (1e-5, 1e-5))
        a = self.artifact("a", [[0.0, 1.0, -1.0]])
        b = self.artifact("b", [[1e-6, 1.0 + 1e-6, -1.0]])
        result = gate.compare(a, b)
        self.assertTrue(result["pass"])
        self.assertFalse(result["bit_identical"])
        self.assertGreater(result["maxabs"], 0)

    def test_signed_zero_bits_separate_from_numeric(self):
        a = self.artifact("a", [[0.0, 2.0, -1.0]])
        b = self.artifact("b", [[-0.0, 2.0, -1.0]])
        result = gate.compare(a, b)
        self.assertTrue(result["pass"])
        self.assertFalse(result["bit_identical"])
        self.assertEqual(result["maxabs"], 0)

    def test_relative_tolerance_combined_not_two_independent_checks(self):
        a = self.artifact("a", [[0, 1000, -1]])
        b = self.artifact("b", [[0, 1000.005, -1]])
        self.assertTrue(gate.compare(a, b)["pass"])
        c = self.artifact("c", [[0, 1000.02, -1]])
        self.assertFalse(gate.compare(a, c)["pass"])

    def test_non_top_logit_also_checked(self):
        a = self.artifact("a", [[0, 3, -1]])
        b = self.artifact("b", [[0.001, 3, -1]])
        r = gate.compare(a, b)
        self.assertFalse(r["pass"])
        self.assertEqual(r["steps"][0]["tolerance_violations"], 1)
        self.assertTrue(r["steps"][0]["top1_equal"])

    def test_top1_difference_fails_even_inside_tolerance(self):
        a = self.artifact("a", [[1, 1, 0]])
        b = self.artifact("b", [[1, 1.000001, 0]])
        r = gate.compare(a, b)
        self.assertFalse(r["pass"])
        self.assertEqual(r["steps"][0]["tolerance_violations"], 0)
        self.assertFalse(r["steps"][0]["top1_equal"])

    def test_greedy_divergence_stops_later_logit_comparison(self):
        a = self.artifact("a", [[2, 0, -1], [2, 0, -1], [2, 0, -1]])
        b = self.artifact("b", [[0, 2, -1], [2, 0, -1], [2, 0, -1]])
        r = gate.compare(a, b)
        self.assertEqual(r["first_token_divergence"], 0)
        self.assertEqual(r["compared_steps"], 1)
        self.assertTrue(r["steps"][0]["comparable_prefix"])
        for row in r["steps"][1:]:
            self.assertFalse(row["comparable_prefix"])
            self.assertIsNone(row["maxabs"])
            self.assertIsNone(row["bit_identical"])
        self.assertFalse(r["pass"])

    def test_teacher_forcing_preserves_prefix_despite_greedy_difference(self):
        a = self.artifact("a", [[2, 0, -1], [2, 0, -1]])
        b = self.artifact("b", [[0, 2, -1], [2, 0, -1]], tokens=[0, 0], mode="teacher_forcing")
        r = gate.compare(a, b)
        self.assertEqual(r["compared_steps"], 2)
        self.assertTrue(r["tokens_equal"])
        self.assertFalse(r["pass"])
        self.assertEqual(r["steps"][1]["maxabs"], 0)

    def test_nan_inf_rejected_even_if_both_files_equal(self):
        for index, value in enumerate([float("nan"), float("inf"), -float("inf")]):
            a = self.artifact("a" + str(index), [[value, 2, -1]])
            b = self.artifact("b" + str(index), [[value, 2, -1]])
            r = gate.compare(a, b)
            self.assertFalse(r["pass"])
            self.assertEqual(r["steps"][0]["nonfinite_reference"], 1)
            self.assertIsNone(r["steps"][0]["maxabs"])

    def test_shape_and_file_byte_count_fail_closed(self):
        a = self.artifact("a")
        b = self.artifact("b", [[0, 3, -1]])
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            gate.compare(a, b)
        for suffix, delta in (("short", -1), ("long", 1)):
            c = self.artifact(suffix)
            path = Path(str(c) + ".logits.f32")
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] if delta < 0 else raw + b"x")
            with self.assertRaisesRegex(ValueError, "byte size"):
                gate.compare(a, c)

    def test_prompt_and_settings_mismatch(self):
        a = self.artifact("a")
        b = self.artifact("b", prompt_ids=[0, 2])
        with self.assertRaisesRegex(ValueError, "prompt_token_ids"):
            gate.compare(a, b)
        c = self.artifact("c")
        self.change(c, lambda m: m["settings"].update(n_ctx_actual=2048))
        with self.assertRaisesRegex(ValueError, "settings"):
            gate.compare(a, c)

    def test_invalid_ids_dtype_incomplete_metadata(self):
        for idx, (key, value) in enumerate((("token_ids", [False, 0]), ("dtype", "float16"), ("complete", False))):
            a = self.artifact("a" + str(idx))
            self.change(a, lambda m: m.update({key: value}))
            with self.assertRaises(ValueError):
                gate.load_metadata(a)

    def test_metadata_top1_must_match_full_file(self):
        a = self.artifact("a")
        b = self.artifact("b")
        def corrupt(m):
            m["token_ids"][0] = m["greedy_token_ids"][0] = 0
            m["steps"][0]["token_id"] = m["steps"][0]["greedy_token_id"] = 0
        self.change(b, corrupt)
        self.assertFalse(gate.compare(a, b)["steps"][0]["top1_metadata_valid"])

    def test_chunk_boundaries_and_little_endian(self):
        # Deliberately choose a block size not dividing the vocabulary.
        rows = [[float(i) / 10 for i in range(23)] for _ in range(2)]
        a, b = self.artifact("a", rows), self.artifact("b", rows)
        with patch.object(gate, "BLOCK_FLOATS", 7):
            r = gate.compare(a, b)
        self.assertTrue(r["pass"])
        self.assertEqual(r["steps"][0]["top1_reference"], 22)
        self.assertEqual(Path(str(a) + ".logits.f32").read_bytes()[4:8], struct.pack("<f", 0.1))

    def test_single_token_prefill_phase_is_not_decode(self):
        a, b = self.artifact("a", prompt_ids=[1]), self.artifact("b", prompt_ids=[1])
        for p in (a, b):
            self.change(p, lambda m: m.update(prompt_mode="single_token", prompt_utf8="", prompt_bytes=0, add_special=False))
        self.assertTrue(gate.compare(a, b)["pass"])
        m, _ = gate.load_metadata(a)
        self.assertEqual(m["steps"][0]["phase"], "last_prefill")

    def test_exclusive_report_never_overwrites(self):
        a, b = self.artifact("a"), self.artifact("b")
        out = self.root / "report.json"
        out.write_text("sentinel", encoding="utf-8")
        self.assertEqual(gate.main([str(a), str(b), "--output", str(out)]), 2)
        self.assertEqual(out.read_text(encoding="utf-8"), "sentinel")

    def test_duplicate_json_keys_rejected(self):
        a = self.artifact("a")
        path = Path(str(a) + ".json")
        raw = path.read_text(encoding="utf-8")
        path.write_text('{"complete": false,' + raw[1:], encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            gate.load_metadata(a)


class StaticHelperContractTests(unittest.TestCase):
    def test_parser_scope_accepts_explicit_no_warmup_regression(self):
        # Regression: COMMON alone rejects --no-warmup before model loading.
        source = (HERE / "native_gate.cpp").read_text(encoding="utf-8")
        self.assertIn("common_params_parse(common_argc, args.data(), p, LLAMA_EXAMPLE_COMPLETION)", source)
        self.assertNotIn("common_params_parse(common_argc, args.data(), p, LLAMA_EXAMPLE_COMMON)", source)
        self.assertIn('explicit_no_warmup |= a == "--no-warmup";', source)
        self.assertIn("require(explicit_no_warmup && explicit_model", source)
        self.assertIn("require(!p.warmup && !p.escape && p.offline", source)

    def test_no_mmap_host_weight_configuration_is_explicitly_allowlisted(self):
        source = (HERE / "native_gate.cpp").read_text(encoding="utf-8")
        unary = re.search(r"const std::set<std::string> unary = \{(.*?)\};", source, re.S)
        self.assertIsNotNone(unary)
        self.assertIn('"--no-mmap"', unary.group(1))
        # This is opt-in on both arms, not a hidden default/configuration mutation.
        self.assertNotIn("p.load_mode =", source)

    def test_all_allowlisted_flags_exposed_by_audited_completion_parser(self):
        # Read-only static audit of the exact isolated source; no parser/backend/model invocation.
        arg_path = Path(r"D:\ds4_work\qwen36_reap_lab\source\common\arg.cpp")
        if not arg_path.is_file():
            self.skipTest("isolated source unavailable; run this contract audit on the parent host")
        helper = (HERE / "native_gate.cpp").read_text(encoding="utf-8")
        upstream = arg_path.read_text(encoding="utf-8")
        self.assertIn("arg.in_example(ex) || (inherit_common && arg.in_example(LLAMA_EXAMPLE_COMMON))", upstream)
        blocks = upstream.split("add_opt(common_arg(")[1:]
        flags = []
        for group in ("unary", "binary"):
            match = re.search(r"const std::set<std::string> " + group + r" = \{(.*?)\};", helper, re.S)
            self.assertIsNotNone(match)
            flags.extend(re.findall(r'"([^"\n]+)"', match.group(1)))
        self.assertIn("--no-warmup", flags)
        self.assertIn("--no-mmap", flags)
        for flag in flags:
            with self.subTest(flag=flag):
                matches = [block for block in blocks if '"' + flag + '"' in block]
                self.assertEqual(len(matches), 1, "flag must identify exactly one upstream option")
                block = matches[0]
                tagged = re.search(r"\.set_examples\(\{([^}]*)\}\)", block)
                examples = tagged.group(1) if tagged else "LLAMA_EXAMPLE_COMMON"
                excluded = re.search(r"\.set_excludes\(\{([^}]*)\}\)", block)
                self.assertTrue("LLAMA_EXAMPLE_COMPLETION" in examples or "LLAMA_EXAMPLE_COMMON" in examples)
                self.assertNotIn("LLAMA_EXAMPLE_COMPLETION", excluded.group(1) if excluded else "")
                if flag == "--no-warmup":
                    self.assertIn("LLAMA_EXAMPLE_COMPLETION", examples)
                    self.assertNotIn("LLAMA_EXAMPLE_COMMON", examples)
                if flag == "--no-mmap":
                    self.assertIn("params.load_mode = value ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;", block)

    def test_exact_native_api_and_safety_contracts_present(self):
        source = (HERE / "native_gate.cpp").read_text(encoding="utf-8")
        for required in ("common_params_parse(", "common_init_from_params(p)", "llama_get_logits_ith(ctx, -1)",
                         "llama_vocab_n_tokens(vocab)", "llama_model_n_layer_nextn(model) == 0", "_O_EXCL", "O_EXCL",
                         "std::isfinite(logits[id])", "--no-warmup", "--gate-token-id", "--gate-teacher",
                         'token_prompt ? "" : read_bytes(p.prompt_file)', "common_tokenize(vocab, p.prompt, true, false)"):
            self.assertIn(required, source)
        self.assertNotIn("llama_sampler_sample(", source)
        self.assertNotIn("llama_server", source)

    def test_two_raw_secret_free_fixtures_utf8(self):
        files = sorted((HERE / "fixtures").glob("*.txt"))
        self.assertEqual(len(files), 2)
        texts = [p.read_bytes().decode("utf-8", errors="strict") for p in files]
        self.assertTrue(all(text.endswith("Assistant:\n") for text in texts))
        self.assertTrue(any("\\n" in text for text in texts))
        self.assertTrue(all("FIXTURE" in text for text in texts))
        self.assertTrue(any("東京" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
