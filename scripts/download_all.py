"""download_all.py — scarica in SEQUENZA i 3 modelli (riprendibile), pensato per girare DETACHED.

Lanciato con Start-Process (PowerShell) sopravvive ai riavvii della sessione Claude. Sequenziale così
ogni download ha tutta la banda: 30B (gate metodo) finisce per primo, poi 235B, poi DS-V4.
snapshot_download/hf_hub_download sono IDEMPOTENTI: riprendono dal punto in cui erano.

Lancio: $env:HF_HUB_ENABLE_HF_TRANSFER='0'; Start-Process python -ArgumentList '-u',
  'scripts/download_all.py' -WindowStyle Hidden -RedirectStandardOutput runs/download_all.log
  -RedirectStandardError runs/download_all.err
Stato: tail runs/download_all.log
"""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # il downloader rust ha dato errori; usa python

from huggingface_hub import snapshot_download, hf_hub_download  # noqa: E402


def log(m):
    print(m, flush=True)


log("=== [1/3] Qwen3-30B-A3B bf16 (gate validazione metodo) ===")
snapshot_download("Qwen/Qwen3-30B-A3B",
                  allow_patterns=["*.safetensors", "*.json", "tokenizer*", "merges*", "vocab*"])
log("=== 30B DONE ===")

log("=== [2/3] Qwen3-235B-A22B Q3_K_M (scale) ===")
snapshot_download("unsloth/Qwen3-235B-A22B-GGUF", allow_patterns=["Q3_K_M/*"],
                  local_dir="models/qwen3-235b")
log("=== 235B DONE ===")

log("=== [3/3] DeepSeek-V4 Flash IQ2 (gigante via ds4) ===")
hf_hub_download("antirez/deepseek-v4-gguf",
                "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf",
                local_dir="models/ds4")
log("=== DS-V4 DONE — ALL COMPLETE ===")
