# Tower dependency rebuild

`requirements.lock` is the reviewed, fully resolved snapshot of the dedicated
production virtual environment. It pins every direct and transitive package.
The audited runtime is CPython 3.13.5 on Alpha arm64.

Direct runtime dependencies are `anyio`, `mcp`, `numpy`, `onnxruntime`,
`PyJWT`, `sqlite-vec`, `tokenizers`, and `uvicorn`. The remaining lock entries
are their resolved transitive dependencies. The ONNX model, tokenizer, SQLite
index, icon, and service environment file are runtime assets/configuration and
are intentionally not Python packages.

## Isolated rebuild and verification

Never rebuild the live `.venv` in place. From `~/tower`:

```bash
scratch=$(mktemp -d /tmp/tower-venv.XXXXXX)
python3.13 -m venv "$scratch/venv"
"$scratch/venv/bin/python" -m pip install --upgrade pip
"$scratch/venv/bin/python" -m pip install --requirement requirements.lock
"$scratch/venv/bin/python" -m pip check
"$scratch/venv/bin/python" -m unittest discover -p 'test_*.py'
```

The full test command must pass from the repository root. Inspect `pip check`
and the package diff before promoting a replacement environment. Promotion is
an explicit maintenance operation: retain the old venv as rollback, atomically
swap the tested environment into place, and restart `tower.service` as
the final mutation. Do not couple dependency refreshes to application changes.

## Refresh procedure

Resolve upgrades only inside a separate scratch venv, run the full suite there,
review `python -m pip freeze` against this file, and commit the lock change as
its own concern. Exact version pins are cross-arm64 reproducibility metadata;
wheel hashes are not embedded because the upstream artifact set can differ by
platform and Python ABI.
