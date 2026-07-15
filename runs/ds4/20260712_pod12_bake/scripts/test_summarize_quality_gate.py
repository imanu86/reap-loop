import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("summarize_quality_gate.py")


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_quality_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_root(tmp_path, name="suite_a"):
    root = tmp_path / name
    arm = root / "k60"
    arm.mkdir(parents=True)

    request = {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Direct."},
            {"role": "user", "content": "build a dashboard"},
        ],
        "max_tokens": 3200,
        "temperature": 0.2,
        "stream": True,
        "think": False,
    }
    response = {
        "stream": True,
        "stream_events": 25,
        "elapsed_s": 5.0,
        "usage": {"prompt_tokens": 7, "completion_tokens": 11, "total_tokens": 18},
        "client_stop": None,
        "stream_error": None,
        "choices": [
            {
                "message": {"role": "assistant", "content": "<html>ok</html>"},
                "finish_reason": "stop",
            }
        ],
    }
    grade = {"task": "frontpage", "level": 3, "detail": {"ok": True}}
    summary = {
        "k60": [
            {
                "run": 1,
                "grade": grade,
                "finish_reason": "stop",
                "client_stop": None,
                "elapsed_s": 5.0,
                "usage": response["usage"],
            }
        ]
    }

    write_json(root / "summary.json", summary)
    write_json(arm / "request_r01.json", request)
    write_json(arm / "response_r01.json", response)
    write_json(arm / "grade_r01.json", grade)
    (arm / "events_r01.jsonl").write_text('{"event": 1}\n', encoding="utf-8")
    (arm / "content_r01.txt").write_text("<html>ok</html>", encoding="utf-8")
    (arm / "hardware.txt").write_text(
        "2026-07-15T00:00:00Z\n"
        "Linux host 6.0 x86_64\n"
        "NVIDIA H100 80GB HBM3, GPU-abc, 555.55, 81920 MiB\n"
        "Mem: 1 2 3\n",
        encoding="utf-8",
    )
    (arm / "binary.sha256").write_text("a" * 64 + "  /bin/ds4-server\n", encoding="utf-8")
    (arm / "mask.sha256").write_text("b" * 64 + "  /masks/k60.txt\n", encoding="utf-8")
    (arm / "server_argv.txt").write_text(
        "/bin/ds4-server -m /model.gguf --cuda --ssd-streaming "
        "--ssd-streaming-cache-experts 1024 --prefill-chunk 512 "
        "-c 4096 -n 3328 --port 18083\n",
        encoding="utf-8",
    )
    (root / "runner.sha256").write_text("c" * 64 + "  /runner.sh\n", encoding="utf-8")
    return root


def run_cli(*roots):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(root) for root in roots)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_emits_deterministic_csv(tmp_path):
    root = make_root(tmp_path)

    proc = run_cli(root)

    assert proc.returncode == 0, proc.stderr
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    assert len(rows) == 1
    row = rows[0]
    assert row["suite/root"] == str(root.resolve())
    assert row["arm"] == "k60"
    assert row["run"] == "1"
    assert row["GPU"] == "NVIDIA H100 80GB HBM3"
    assert row["model"] == "deepseek-v4-flash"
    assert row["ctx"] == "4096"
    assert row["cache_experts"] == "1024"
    assert row["prefill_chunk"] == "512"
    assert row["server_max_tokens"] == "3328"
    assert row["request_max_tokens"] == "3200"
    assert row["temperature"] == "0.2"
    assert row["think"] == "false"
    assert row["stream"] == "true"
    assert row["grade"] == "L3"
    assert row["finish_reason"] == "stop"
    assert row["elapsed_s"] == "5"
    assert row["stream_events"] == "25"
    assert row["tok/s"] == "5"
    assert row["completion_tokens"] == "11"
    assert row["content_chars"] == "15"
    assert row["prompt_chars"] == str(len("build a dashboard"))
    assert row["prompt_sha256"] == hashlib.sha256(
        "build a dashboard".encode("utf-8")
    ).hexdigest()
    assert row["binary_sha256"] == "a" * 64
    assert row["mask_sha256"] == "b" * 64
    assert row["runner_sha256"] == "c" * 64


def test_missing_artifact_fails_without_guessing(tmp_path):
    root = make_root(tmp_path)
    (root / "k60" / "mask.sha256").unlink()

    proc = run_cli(root)

    assert proc.returncode == 1
    assert "missing required artifact" in proc.stderr
    assert "mask.sha256" in proc.stderr
    assert proc.stdout == ""


def test_k0_requires_no_mask_hash(tmp_path):
    root = make_root(tmp_path)
    arm = root / "k60"
    arm.rename(root / "k0")
    (root / "k0" / "mask.sha256").unlink()
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["k0"] = summary.pop("k60")
    write_json(summary_path, summary)

    proc = run_cli(root)

    assert proc.returncode == 0, proc.stderr
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    assert rows[0]["arm"] == "k0"
    assert rows[0]["mask_sha256"] == ""


def test_incoherent_summary_fails(tmp_path):
    root = make_root(tmp_path)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["k60"][0]["finish_reason"] = "length"
    write_json(summary_path, summary)

    proc = run_cli(root)

    assert proc.returncode == 1
    assert "finish_reason disagrees" in proc.stderr


def test_module_main_returns_error_for_missing_summary(tmp_path, capsys):
    module = load_module()
    missing_root = tmp_path / "empty"
    missing_root.mkdir()

    assert module.main([str(missing_root)]) == 1
    captured = capsys.readouterr()
    assert "summary.json" in captured.err
