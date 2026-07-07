"""Per-item JSONL logger + run manifest.

One JSON object per item (spec §2-E). Denominator discipline: EVERY item that was
attempted gets a line — including failures and abstentions — so downstream averages
can use N_total. The analysis layer must never filter to successes before averaging.
"""
from __future__ import annotations
import os
import json
import datetime


class RunLogger:
    def __init__(self, runs_dir: str, arm: str, manifest: dict):
        os.makedirs(runs_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{ts}_{arm}"
        # guarantee a unique run_id: multiple runs of the same arm (e.g. a theta
        # sweep) can land in the same second -> never let them overwrite each other.
        run_id, i = base, 1
        while os.path.exists(os.path.join(runs_dir, f"{run_id}.jsonl")):
            run_id, i = f"{base}_{i}", i + 1
        self.run_id = run_id
        self.path = os.path.join(runs_dir, f"{self.run_id}.jsonl")
        self.manifest_path = os.path.join(runs_dir, f"{self.run_id}.manifest.json")
        manifest = dict(manifest)
        manifest.update({"run_id": self.run_id, "arm": arm, "created": ts})
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        self._fh = open(self.path, "w", encoding="utf-8")
        self.n = 0

    def log_item(self, *, item_id, arm, seed, distance, difficulty, category,
                 cost, confidence, correct, abstained, rung_stop=None,
                 sensor_used=None, pred="", gold="", question="",
                 n_distractors=None, extra=None):
        row = {
            "item_id": item_id,
            "arm": arm,
            "seed": seed,
            "distance": distance,
            "difficulty": difficulty,
            "category": category,
            # cost vector, flattened
            "prefill_tokens": cost.prefill_tokens,
            "decode_tokens": cost.decode_tokens,
            "n_embed_calls": cost.n_embed_calls,
            "n_rag_calls": cost.n_rag_calls,
            "n_llm_calls": cost.n_llm_calls,
            "wall_clock_s": round(cost.wall_clock_s, 4),
            "total_tokens": cost.total_tokens,
            # outcome
            "confidence": confidence,
            "sensor_used": sensor_used,
            "correct": bool(correct),
            "abstained": bool(abstained),
            "rung_stop": rung_stop,          # None in Step 0/1 (no cascade yet)
            "pred": pred,
            "gold": gold,
            "question": question,
            "n_distractors": n_distractors,
        }
        if extra:
            row.update(extra)
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.n += 1

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
