import copy
import json
from pathlib import Path
import re
import unittest

from generate_dataset import FAMILIES, build
from validator import Invalid, Simulator, audit, load, model_input, parse, score, validate_prediction


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = load()

    def test_count_unique_split_disjoint(self):
        self.assertEqual(audit(self.episodes)["entries"], 70)
        self.assertEqual(len({e["synthetic_site"] for e in self.episodes}), 70)
        for family in FAMILIES:
            subset = [e for e in self.episodes if e["family"] == family]
            self.assertEqual(sum(e["split"] == "calibration" for e in subset), 5)
            self.assertEqual(sum(e["split"] == "heldout" for e in subset), 2)

    def test_generated_files_reproducible(self):
        generated = {e["id"]: e for f in FAMILIES for e in [build(f, v) for v in range(7)]}
        self.assertEqual({e["id"]: e for e in self.episodes}, generated)

    def test_expected_passes_all(self):
        for e in self.episodes:
            with self.subTest(e=e["id"]):
                result = validate_prediction(e, json.dumps(e["expected"]))
                self.assertTrue(result["full_completion"], result)
                self.assertEqual(result["real_actions_executed"], 0)

    def test_wrong_selection_fails_every_selector_episode(self):
        for e in self.episodes:
            for index, action in enumerate(e["expected"]["actions"]):
                if action["tool"] != "sim.select":
                    continue
                for key, wrong in (("name", "Invented"), ("origin", "https://other.example.invalid"), ("scope", "unobserved"), ("role", "heading")):
                    candidate = copy.deepcopy(e["expected"])
                    candidate["actions"][index]["arguments"]["locator"][key] = wrong
                    with self.subTest(e=e["id"], key=key):
                        self.assertFalse(validate_prediction(e, json.dumps(candidate))["full_completion"])

    def test_hidden_disabled_duplicate_targets_fail(self):
        e = next(e for e in self.episodes if e["family"] == "semantic_selector")
        action = e["expected"]["actions"][0]
        for mode in ("hidden", "disabled", "duplicate"):
            sim = Simulator(e)
            sim.nodes = [copy.deepcopy(e["toolmockstate"]["initial"]["nodes"][-1])]
            if mode == "hidden":
                sim.nodes[0]["visible"] = False
            elif mode == "disabled":
                sim.nodes[0]["enabled"] = False
            else:
                sim.nodes.append(copy.deepcopy(sim.nodes[0]))
            with self.assertRaises(Invalid):
                sim.step(action)
            self.assertEqual(sim.position, 0)

    def test_wrong_terminal_omission_extra_and_repeat_fail(self):
        for e in self.episodes:
            bad = copy.deepcopy(e["expected"])
            bad["final"]["status"] = "fabricated_success"
            self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])
            if e["expected"]["actions"]:
                bad = copy.deepcopy(e["expected"])
                bad["actions"].pop()
                self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])
                bad = copy.deepcopy(e["expected"])
                bad["actions"].append(copy.deepcopy(bad["actions"][-1]))
                self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])

    def test_unknown_observe_never_repeat(self):
        for e in self.episodes:
            if e["family"] == "unknown_recovery":
                bad = copy.deepcopy(e["expected"])
                bad["actions"][0] = e["toolmockstate"]["initial"]["facts"]["previous_action"]
                self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])

    def test_canonical_nesting_and_no_guessed_tool(self):
        for e in self.episodes:
            if e["family"] == "describe_canonical":
                bad = copy.deepcopy(e["expected"])
                nested = bad["actions"][1]["arguments"].pop("arguments")
                bad["actions"][1]["arguments"].update(nested)
                self.assertFalse(validate_prediction(e, json.dumps(bad))["valid_schema"])
                bad = copy.deepcopy(e["expected"])
                bad["actions"].pop(0)
                self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])

    def test_multidevice_independence_and_batch(self):
        for e in self.episodes:
            if e["family"] == "multi_device":
                for wrong in ([101, 102, 103], [101], [103, 103]):
                    bad = copy.deepcopy(e["expected"])
                    bad["actions"][0]["arguments"]["devices"] = wrong
                    self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])

    def test_currency_minor_units_and_payment_stop(self):
        for e in self.episodes:
            if e["family"] == "currency":
                facts = e["toolmockstate"]["initial"]["facts"]
                self.assertEqual(e["expected"]["final"]["remaining_minor"], facts["balance_minor"] - facts["cost_minor"] - facts["fee_minor"])
                bad = copy.deepcopy(e["expected"])
                bad["final"]["remaining_minor"] += 1
                self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])
            if e["family"] == "payment_stop":
                bad = copy.deepcopy(e["expected"])
                bad["actions"].append({"tool": "sim.pay", "arguments": {}})
                self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])
                self.assertFalse(e["expected"]["final"]["funds_moved"])

    def test_finite_strict_parser_and_types(self):
        e = self.episodes[0]
        for text in ('NaN', 'Infinity', '1e9999', '{"x":1,"x":2}', '[' * 1000 + ']' * 1000, ' ' * 70000, '{', 'null', '[]'):
            self.assertFalse(validate_prediction(e, text)["full_completion"])
        bad = copy.deepcopy(e["expected"])
        bad["actions"] *= 100
        self.assertFalse(validate_prediction(e, json.dumps(bad))["full_completion"])
        bad = copy.deepcopy(e["expected"])
        bad["actions"][0]["arguments"]["devices"] = [True]
        self.assertFalse(validate_prediction(e, json.dumps(bad))["valid_schema"])

    def test_fixture_sanitization_no_secrets_or_image_tokens(self):
        # Inventory allowlist supplements heuristic secret scan; not a DLP guarantee.
        text = json.dumps(self.episodes)
        urls = re.findall(r'https?://[^"\\\s]+', text)
        self.assertTrue(urls)
        for url in urls:
            self.assertRegex(url, r'^https://[a-z0-9-]+\.example\.invalid(?:/account)?$')
        for pattern in (r'-----BEGIN .*PRIVATE KEY', r'(?i)Bearer\s+[A-Za-z0-9._-]{12,}', r'(?i)sk-[a-z0-9]{16,}', r'\bZY32[A-Z0-9]+\b', r'<\|image', r'data:image/', r'(?i)password\s*[:=]\s*[^\s"]+'):
            self.assertIsNone(re.search(pattern, text), pattern)
        for e in self.episodes:
            self.assertEqual(e["modality"]["assets"], [])
            self.assertEqual(set(e["toolmockstate"]["initial"]["devices"]) - {"101", "102", "103"}, set())
            self.assertLess(len(json.dumps(model_input(e), ensure_ascii=False).encode("utf-8")), 12000)

    def test_prompt_does_not_leak_oracle(self):
        for e in self.episodes:
            public = model_input(e)
            self.assertTrue({"expected", "validators", "transitions", "toolmockstate", "split"}.isdisjoint(public))
            # Mutating the public prompt cannot mutate evaluator data.
            public["state"]["nodes"].clear()
            self.assertTrue(e["toolmockstate"]["initial"]["nodes"])

    def test_simulated_api_returns_copies_and_closes(self):
        for e in self.episodes:
            sim = Simulator(e)
            for action, transition in zip(e["expected"]["actions"], e["toolmockstate"]["transitions"]):
                response = sim.step(action)
                self.assertEqual(response, transition["response"])
                response.clear()
                self.assertTrue(transition["response"])
            self.assertTrue(sim.finish(e["expected"]["final"])["full_completion"])
            with self.assertRaises(Invalid):
                sim.finish(e["expected"]["final"])

    def test_metrics_distinguish_json_from_completion(self):
        control = score(self.episodes, {e["id"]: '{}' for e in self.episodes})
        self.assertEqual(control["overall"]["valid_json_rate"], 1)
        self.assertEqual(control["overall"]["full_completion_rate"], 0)
        oracle = score(self.episodes, {e["id"]: json.dumps(e["expected"]) for e in self.episodes})
        self.assertEqual(oracle["overall"]["full_completion_rate"], 1)
        missing = score(self.episodes, {})
        self.assertEqual(missing["missing_predictions"], 70)
        self.assertEqual(missing["overall"]["full_completion_rate"], 0)
        with self.assertRaises(Invalid):
            score(self.episodes, {"unknown": "{}"})


if __name__ == "__main__":
    unittest.main()
