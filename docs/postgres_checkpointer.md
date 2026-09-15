# Postgres checkpointer: local verification & production deployment

`checkpointing/store.py`'s `get_checkpointer()` falls back to Postgres
whenever `DATABASE_URL` is set (see `_build_postgres_checkpointer`). This
doc covers two different things: how to stand up a real Postgres locally
to verify that path actually works, and what changes before it's safe to
run in production — those are not the same bar.

## Part 1 — Local verification (macOS, Homebrew)

Goal: prove `get_checkpointer()` + a real compiled graph round-trip state
through an actual Postgres instance, not just that the function returns an
object.

### 1. Install and start Postgres

```bash
brew install postgresql@16

# Start it for this session only (not as a persistent background service,
# so teardown is a clean stop rather than un-registering a launch agent):
brew_prefix=$(brew --prefix postgresql@16)
"$brew_prefix/bin/pg_ctl" -D "$brew_prefix/../../var/postgresql@16" -l /tmp/postgres.log start
```

(`brew services start postgresql@16` also works, but registers it to
survive reboots — skip that for a throwaway verification instance.)

### 2. Create a test database

Homebrew's Postgres trusts local connections as your OS user by default —
no password needed:

```bash
createdb market_research_verify
```

### 3. Install the `prod` extra

```bash
pip install -e ".[dev,prod]"   # adds langgraph-checkpoint-postgres + psycopg
```

### 4. Point at it and run the integration test

```bash
export DATABASE_URL="postgresql://$(whoami)@localhost:5432/market_research_verify"
pytest tests/test_checkpointing_postgres.py -v
```

This test mirrors `tests/test_checkpointing.py` (the SQLite equivalent):
it builds a real compiled graph with `get_checkpointer()`, runs it through
`run_graph()`, and asserts the checkpoint history and final state actually
persisted in Postgres — not a mock. It's skipped automatically (via a
real connection attempt in a fixture) when `DATABASE_URL` isn't set or
Postgres isn't reachable, so it doesn't affect `pytest` for anyone without
this local setup, including CI.

### 5. Tear down

```bash
dropdb market_research_verify
"$brew_prefix/bin/pg_ctl" -D "$brew_prefix/../../var/postgresql@16" stop
brew uninstall postgresql@16
brew cleanup
unset DATABASE_URL
```

## Part 2 — What changes for real production use

Local verification proves the code path works. It does **not** prove the
setup is production-ready. Before pointing this at a real deployment:

- **Use a managed Postgres**, not a self-hosted instance — RDS/Aurora,
  Cloud SQL, Azure Database for PostgreSQL, or a PaaS option (Neon,
  Supabase, Render). You want automated backups, patching, and failover
  that a hand-run `pg_ctl` instance doesn't give you.
- **Never put the connection string in `.env` in a deployed environment.**
  Inject `DATABASE_URL` at deploy time from a secrets manager (AWS Secrets
  Manager, GCP Secret Manager, Vault) — `.env` is a local-dev convenience
  only, and `config.py`'s `env_file=".env"` already only loads it if the
  file exists.
- **Require TLS.** Add `sslmode=require` (or `verify-full` with a CA
  bundle) to the connection string — most managed providers enforce this
  by default, but a self-hosted or bring-your-own-VM Postgres won't.
- **Pool connections.** `PostgresSaver.from_conn_string()` opens one
  direct `psycopg` connection per process. Multiple app instances/workers
  will exhaust `max_connections` fast — front it with PgBouncer, or use
  the provider's built-in pooler (RDS Proxy, Neon's pooler, etc.).
- **Treat `checkpointer.setup()` as a migration, not a boot step.** It
  runs `CREATE TABLE IF NOT EXISTS` DDL. Calling it from every app
  instance on every startup is fine for a single local process, but in a
  multi-instance deployment it should run once, from a controlled deploy
  step, using a role with DDL privileges — the running app should use a
  separate, lower-privileged role that can only read/write rows.
- **Plan for table growth.** Every graph step writes a new checkpoint row;
  nothing in this project prunes old ones. Add a retention job (e.g.
  delete checkpoints older than N days per thread) before this runs
  unattended for any length of time.
- **Close the connection on shutdown.** `_build_postgres_checkpointer`
  currently calls `PostgresSaver.from_conn_string(url).__enter__()` and
  keeps the object for the life of the process, but nothing ever calls the
  matching `__exit__()`. That's harmless for a short CLI invocation (the
  process exit reclaims the socket) but should be wired into a graceful
  shutdown handler for a long-running server process so the connection
  closes cleanly instead of being dropped by the OS.
