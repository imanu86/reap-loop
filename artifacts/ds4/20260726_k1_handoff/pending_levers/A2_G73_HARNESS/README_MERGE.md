# A2 G73_OPEN harness

Status: `HARNESS_READY=yes`

This bundle was prepared without starting DS4, compiling a binary, or changing
the shared runner/source tree.

## Bundle contents

- `run_two_turn_lifecycle_a2.ps1`: standalone A2 runner.
- `manifest_47.env`: immutable source manifest; the runner writes an effective
  per-run copy.
- `runtime_monitor.ps1`: RAM, process-I/O and GPU sampler.
- `test_a2_harness_static.ps1`: parser plus ON/OFF `ValidateOnly` tests.
- `compare_a2_abba.ps1`: four-run `ON,OFF,OFF,ON` identity and variance gate.
- `a2_runner_reference.patch`: reference diff against shared-runner snapshot
  SHA-256
  `358EF45ABDBDC398182441C66658F33FA4BAFF1E735B69192536B64378C15ACD`.
- `static_test_receipt.txt`: proof that both validation arms completed without
  a server start.

The reference patch must not be applied blindly if the target runner has a
different SHA-256. A1 was changing that file while this bundle was prepared.

## Merge procedure after A1 finishes

1. Freeze the A1 executable, source and runner hashes.
2. Copy the A1 runner to a temporary file outside the shared runtime directory.
3. Port the A2 sections from the standalone runner or inspect the reference
   patch. Preserve A1 changes that are unrelated to A2.
4. Update `$exe`, `$expectedExeHash`, `$expectedSourceHashes` and, if required,
   the deterministic response hashes to the final A1 receipts.
5. Keep `TraceMode`, packed copy, batched publish, G73 host-selected reuse,
   route-I/O QD4, graph-tensor-device and allocation tracing OFF for both arms.
6. Copy `compare_a2_abba.ps1` and the updated static test beside the merged
   runner.
7. Parse every PowerShell file:

```powershell
$tokens = $null
$errors = $null
[Management.Automation.Language.Parser]::ParseFile(
    '.\run_two_turn_lifecycle.ps1',
    [ref]$tokens,
    [ref]$errors) | Out-Null
if ($errors.Count) { throw ($errors | Out-String) }
```

8. Once the final source hashes match the binary receipt, execute only the two
   non-serving validations:

```powershell
.\run_two_turn_lifecycle.ps1 -G73Open On  -ValidateOnly
.\run_two_turn_lifecycle.ps1 -G73Open Off -ValidateOnly
```

9. Confirm both effective manifests contain 47 entries, have the same
   `normalized_pair_manifest_sha256`, and differ only at line 41:
   `DS4_G73_OPEN=1` versus `DS4_G73_OPEN=0`.
10. Only after those checks may a run slot use four fresh processes in the
    fixed order:

```powershell
.\run_two_turn_lifecycle.ps1 -G73Open On
.\run_two_turn_lifecycle.ps1 -G73Open Off
.\run_two_turn_lifecycle.ps1 -G73Open Off
.\run_two_turn_lifecycle.ps1 -G73Open On
```

11. Compare the four run directories:

```powershell
.\compare_a2_abba.ps1 `
  -On1  '<ON1 run directory>' `
  -Off1 '<OFF1 run directory>' `
  -Off2 '<OFF2 run directory>' `
  -On2  '<ON2 run directory>' `
  -OutputPath '.\a2_abba_result.txt'
```

`VARIANCE_LT_10_PERCENT` is the required interpretation when the absolute
median graph-throughput delta is below 10%. A delta at or above 10% is only a
candidate effect if both order-controlled pairs have the same sign and the
within-arm spread is below 10%.

Until the A1 receipts are merged and strict `ValidateOnly` passes:

`READY_FOR_RUN_SLOT=no`
