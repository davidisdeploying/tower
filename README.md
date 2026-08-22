# Tower

Tower is the control plane for a small self-hosted fleet: an MCP server that
exposes the fleet, a notes vault, and a pool of AI workers as one set of
typed tools, so any MCP client can search, read, write, and dispatch work
without knowing which machine anything lives on.

The interesting part is not the model. It is the boundary around it: what a job
is allowed to touch, what it has to return, what happens when two jobs run at
once, and how a wrong answer gets caught.

## What it does

**21 tools over MCP.** Vault access (`search_vault`, `get_note`,
`read_vault_range`, `write_note`, `append_journal_entry`, `delete_note`,
`index_metadata`), history recall (`search_history`, `read_history`), dispatch
and inspection (`dispatch`, `status`, `inspect_run`, `relay_audit`,
`worker_routing_status`, `recommend_model`, `model_routing_status`), and a few
small condensing helpers. That is the whole interface; there is no second REST
API to keep in sync.

**Quota-aware model routing.** `dispatch(provider="auto", task_size=...)` scores
the available providers from live usage telemetry and active reservations,
grades them GREEN / YELLOW / AMBER / RED, and reserves the winner in the same
transaction that selects it. RED is never launched, missing telemetry is its own
explicit state rather than an assumption, and a healthy running job is never
migrated because later telemetry changed the winner. Work that can be fully
specified in advance goes to a local open-weight model instead and costs
nothing. See `quota_router.py`.

**Serialisation by what a job touches.** Mutating jobs declare a scope -
`repo:<path>`, `service:<name>`, `path:<path>`, or host-wide `host:<alias>` -
and jobs sharing a scope serialise. Independent scopes on the same machine run
concurrently. A job spanning two machines declares both, and claim sets are
compared symmetrically, so an undeclared second machine is genuinely
unprotected rather than merely undocumented.

**An evidence contract.** Every run carries a single-use token, writes an
immutable request and a durable response, and must return output that actually
backs its claims. Exit 0 means the process ended, not that the work happened -
so the operating rule is to independently verify one material fact against
source before believing a result.

**Identity at the origin.** Tower sits behind Cloudflare Access and verifies the
access token on every call rather than trusting the tunnel, so a request that
reaches the origin without a valid assertion is rejected by the server itself.

## Running this

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.lock
.venv/bin/python server.py     # serves on 127.0.0.1:8765
```

Tests are stdlib `unittest`, no test dependencies:

```sh
.venv/bin/python -m unittest discover -p "test_*.py"
```

`SOURCE_LAYOUT.md` maps the modules; `DEPENDENCIES.md` explains why each
dependency is present.

## A note on the naming

The code carries names that no longer match the machines: worker seats called
`worker1`, `worker3`, `worker2` and `localworker`, a `FLEET-` token prefix, run
directories named `from-worker2/`, and a config path under `~/.config/fleet/`.

These are deliberate, not leftovers. The fleet was renamed, and the identifiers
were not, because they are load-bearing: they name directories that already
exist on disk holding historical run records, tokens that must still parse years
after they were issued, and marker files whose absence changes behaviour.
Renaming them in code without migrating the state they point at is a real
failure mode - it silently broke a collector here for three hours once, which is
why the guard that caught it exists.

So the rule is forward-only. New work uses the current names; historical
evidence keeps the names that were true when it was written. If a name looks
inconsistent with the machine list, that is usually the compatibility layer
doing its job.

## Scope

This runs one person's fleet. It is not a product, it has one operator, and it
assumes a network it can trust at the edges. The design ideas - routing on cost,
locking by resource rather than by host, requiring evidence rather than trusting
a summary - are the parts worth reading.
