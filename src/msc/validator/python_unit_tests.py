"""python_unit_tests.py — task PRIMARIO: genera funzione Python, valida con unit test nascosti.

Segnale binario robusto: la funzione generata passa TUTTA la suite -> correct=True, altrimenti False.

Determinismo & sicurezza (rischio R5):
  - decoding greedy (T=0), seed fissi, prompt fisso
  - esecuzione del codice generato in SANDBOX isolata: subprocess separato, timeout duro,
    niente rete, niente filesystem scrivibile, niente import pericolosi
  - suite di test NASCOSTA al modello e con copertura di edge case (no test banali)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

from msc.validator.base import ContextLengthResult, ValItem, Validator


def _extract_code(model_output: str) -> str:
    """Estrae il codice Python dall'output del modello.

    Robusta a tre forme comuni:
      - blocco recintato ```python ... ``` (o ``` ... ```);
      - codice "nudo" senza recinto (l'intero output è codice);
      - testo con un blocco recintato in mezzo.
    Sceglie il PRIMO blocco recintato se presente, altrimenti l'intero output.
    """
    if model_output is None:
        return ""
    text = model_output
    fence = "```"
    if fence in text:
        # primo blocco recintato
        start = text.index(fence) + len(fence)
        # salta un eventuale specificatore di linguaggio sulla stessa riga (es. ```python)
        newline = text.find("\n", start)
        if newline != -1:
            first_line = text[start:newline].strip().lower()
            if first_line in ("python", "py", "python3") or first_line == "":
                start = newline + 1
        end = text.find(fence, start)
        if end != -1:
            return text[start:end]
        # recinto di apertura senza chiusura: prendi il resto
        return text[start:]
    return text


# Programma eseguito nel subprocess sandbox. Definisce il codice candidato e la suite di test in uno
# scope condiviso, poi esegue i test. Stampa un marcatore di successo SOLO se nessuna asserzione/
# eccezione viene sollevata. Qualsiasi errore -> exit code != 0 / nessun marcatore.
_SANDBOX_HARNESS = r"""
import sys

_OK_MARKER = "__MSC_SANDBOX_PASS__"

# Scope condiviso tra codice candidato e test (i test fanno riferimento ai nomi definiti dal codice).
_ns = {{"__name__": "__sandbox__"}}

_CANDIDATE_CODE = {candidate!r}
_TEST_CODE = {tests!r}

try:
    exec(compile(_CANDIDATE_CODE, "<candidate>", "exec"), _ns)
    exec(compile(_TEST_CODE, "<tests>", "exec"), _ns)
except BaseException as exc:  # include AssertionError, eccezioni a runtime, SystemExit
    sys.stderr.write("SANDBOX_FAIL: {{}}: {{}}\n".format(type(exc).__name__, exc))
    sys.exit(1)

# Tutto passato.
sys.stdout.write(_OK_MARKER)
sys.exit(0)
"""


class PythonUnitTestValidator(Validator):
    name = "python-unit-tests"

    def __init__(self, dataset_path: str, exec_timeout_s: float = 5.0) -> None:
        self._dataset_path = dataset_path  # es. problemi stile HumanEval/MBPP + test nascosti
        self._exec_timeout_s = exec_timeout_s

    def items(self) -> list[ValItem]:
        """Carica (prompt, suite di test nascosta) da un jsonl in ordine deterministico.

        Formato di riga atteso: {"item_id": str, "prompt": str, "tests": str}.
        Le righe vuote vengono ignorate. L'ordine è quello del file (deterministico).
        """
        items: list[ValItem] = []
        with open(self._dataset_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                items.append(
                    ValItem(
                        item_id=str(row["item_id"]),
                        prompt=str(row["prompt"]),
                        payload={"tests": str(row["tests"])},
                    )
                )
        return items

    def verify(self, item: ValItem, model_output: str) -> bool:
        """Estrae la funzione dall'output e la esegue in sandbox contro item.payload['tests']."""
        code = _extract_code(model_output)
        if not code.strip():
            return False
        return self._run_in_sandbox(code, item.payload["tests"])

    def evaluate_at_lengths(self, *, generate_fn, ctx_lengths, filler) -> list[ContextLengthResult]:
        """Per ciascuna lunghezza: riempi il contesto, genera, verifica in sandbox, aggrega in UN punto."""
        items = self.items()
        results: list[ContextLengthResult] = []
        for ctx in ctx_lengths:
            n_correct = 0
            for item in items:
                filled_prompt, _eff_len = filler.fill(item.prompt, ctx)
                output = generate_fn(filled_prompt)
                if self.verify(item, output):
                    n_correct += 1
            results.append(ContextLengthResult(ctx_len=ctx, n_items=len(items), n_correct=n_correct))
        return results

    def _run_in_sandbox(self, code: str, tests: str) -> bool:
        """Subprocess isolato + timeout duro; True solo se TUTTI i test passano.

        Robustezza (rischio R5):
          - subprocess separato: un crash/SystemExit del codice generato non tocca il validatore;
          - timeout duro: codice che cicla all'infinito viene ucciso -> verdetto False (no hang);
          - nessuna eredità di stdin; cwd = directory temporanea usa-e-getta (niente side-effect sul
            repo); l'ambiente è ridotto al minimo.
        Il verdetto è True solo se il processo esce con codice 0 E stampa il marcatore atteso.
        """
        program = _SANDBOX_HARNESS.format(candidate=code, tests=tests)

        # Directory temporanea isolata: il subprocess gira qui, così eventuali file scritti dal
        # codice generato restano confinati e vengono rimossi all'uscita (niente side-effect).
        with tempfile.TemporaryDirectory(prefix="msc_sandbox_") as workdir:
            script_path = os.path.join(workdir, "_msc_run.py")
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(program)

            # Ambiente minimale e deterministico.
            env = {
                "PYTHONHASHSEED": "0",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            # PATH minimo (alcune piattaforme ne hanno bisogno per avviare l'interprete).
            if "PATH" in os.environ:
                env["PATH"] = os.environ["PATH"]
            if "SYSTEMROOT" in os.environ:  # Windows: richiesto per inizializzare il runtime
                env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

            try:
                proc = subprocess.run(
                    [sys.executable, "-I", "-S", script_path],
                    cwd=workdir,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self._exec_timeout_s,
                )
            except subprocess.TimeoutExpired:
                # Loop infinito / blocco: subprocess.run ha già ucciso il processo. Verdetto False.
                return False
            except OSError:
                # Impossibile avviare il subprocess: fallback prudente -> False (non promuoviamo).
                return False

            if proc.returncode != 0:
                return False
            out = proc.stdout.decode("utf-8", errors="replace")
            return "__MSC_SANDBOX_PASS__" in out
