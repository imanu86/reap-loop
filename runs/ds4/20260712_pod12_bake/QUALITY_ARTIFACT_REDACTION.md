# Quality artifact environment redaction

The original quality runner captured the complete process environment in each
`server_env.txt`. Before publication, those copies were reduced to the runtime
allowlist already used by the corrected runner:

- `ARMS`, `BIN`, `LEARN`, `MAX_TOKENS`, `MODEL`, `OUT`, `PORT`, `REPO`;
- every `DS4_*` variable.

This removes shell/session metadata such as `SSH_CLIENT`, `SSH_CONNECTION`,
`HOME`, `PATH`, and `PWD`. No DS4 setting, model/mask path, binary hash, command
line, request, response, event stream, generated content, grading result, or
hardware record was removed. Each output directory also contains the exact
`runner_used.sh` copied from its producer pod.
