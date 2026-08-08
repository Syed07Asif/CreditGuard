# Session memory

Working notes carried over between Claude Code sessions on this repo. Not part of
the project's build output — read this first when resuming work here, alongside
`CLAUDE.md` (which is the actual project contract: hard rules, stack, phase
checklist).

---

## Project status (as of 2026-08-08)

- **Phase 1** (Foundation, config, database) — done, tests passing, **committed**
  (`0bbdfcc`, "Phase 1: Foundation, configuration and database").
- **Phase 2** (Synthetic data generation and ingestion) — done, all 23 tests
  passing, ruff/black clean, verified end-to-end at full 50k-customer scale
  (generation 2.8s, ingest ~40min, realised default rate 0.1105). **Not yet
  committed** as of this note — check `git status`/`git log` before assuming
  otherwise, this may be stale by the time you read it.
- **Next up:** Phase 3 (data validation and cleaning pipeline), per `CLAUDE.md`'s
  checklist. Phases are strictly sequential — don't start Phase 3 work, and don't
  add stub files for it, until the user actually asks.

This is an educational/portfolio simulation (synthetic data only, not a real
lending system), built one phase at a time with the user reviewing each phase's
completion before moving on.

---

## Environment specifics (not visible from git alone)

- **Local Postgres port conflict:** a native Windows `postgres.exe` service
  already listens on port 5432, unrelated to this project. The
  `creditguard_postgres` container is remapped to host port **5433** via
  `DB_PORT=5433` in this repo's `.env` (git-ignored — `.env.example` still shows
  the standard 5432 default for machines without this conflict). If you see
  connection/auth failures against "postgres," check this first — `docker
  compose ps` and `netstat -ano | grep 5432` will show the conflict.
- **`docker-compose.yml`** sets `max_locks_per_transaction=256` on the postgres
  service (Postgres's default is 64) — required for Phase 2's bulk ingest at
  ~100k-row scale. See "Ingest design" below for why this alone wasn't enough.
- **Python 3.11** was missing from this machine when Phase 1 started (only 3.10
  was present); installed via `winget install --id Python.Python.3.11 --source
  winget` at `C:\Users\asiff\AppData\Local\Programs\Python\Python311`, and this
  repo's `.venv` was built from that interpreter. If `.venv` is ever recreated,
  use that same 3.11 interpreter — `CLAUDE.md` fixes the stack at Python 3.11.
- **pgAdmin 4** is installed on this machine
  (`C:\Users\asiff\AppData\Local\Programs\pgAdmin 4`). Connection details (same
  server, two databases): host `localhost`, port `5433`, user `creditguard`,
  password `changeme`, databases `creditguard` (app) and `creditguard_test`
  (pytest only).
- Docker Desktop shows two containers for this project: `creditguard_postgres`
  and `creditguard_mlflow` (mlflow on host port 5000, sqlite backend store).

---

## Ingest design: why per-batch commits, not one transaction

`src/creditguard/data/ingest.py` loads generated parquet data in batches of
`COMMIT_BATCH_SIZE = 2000` rows, each its own committed transaction (bad rows
within a batch are isolated via SAVEPOINT, not by aborting the batch) — **not**
one single transaction for the whole ~400k-row load, even though the original
Phase 2 spec asked for "a single transaction."

**Why:** tried literally as one transaction first. Postgres FK checks take a
row-level lock on the referenced parent row, held until commit. At ~400k rows
across 5 FK-linked tables this first hit `psycopg.errors.OutOfMemory: out of
shared memory / HINT: increase max_locks_per_transaction` (fixed by raising that
setting to 256), but even after that fix, one real run sat for 90+ minutes
without finishing — Postgres's lock manager degrades badly as a single
transaction's held-lock list grows into the tens of thousands, regardless of the
configured ceiling. This was only discovered by actually running ingest at full
50k-customer scale; smaller test-scale runs (2k customers) never hit it.

Also fixed in the same pass: `--truncate` wasn't clearing `data_quality_issues`,
so repeated truncated reloads accumulated duplicate quarantine-log rows; and a
naive "chunk-then-fallback-to-per-row" quarantine strategy degraded to one DB
round trip per row when bad rows are spread uniformly (near-certain at any
reasonable chunk size given 1–5% injection rates) — replaced with recursive
bisection (try bulk, split in half on failure, recurse), roughly O(bad_rows ×
log n) instead of O(n).

**Don't "fix" this back to a single transaction** to match the literal spec
wording — the deviation is deliberate and already documented in the module's own
docstring and in `docs/data_generation.md`. If a later phase needs to bulk-load
large FK-linked tables again, reuse this batched-commit + bisection pattern.

---

## How this project likes to be verified

- When a phase's acceptance criteria name a specific scale (row count, customer
  count, etc.), actually run it at that scale before declaring the phase done —
  don't extrapolate from a smaller test run. Small-scale runs here (2k customers)
  passed cleanly but hid a real PostgreSQL scaling failure that only appeared at
  the actual 50k-customer acceptance-criteria scale.
- For long-running background steps, proactively set up a way for the user to
  independently verify progress themselves (e.g. direct DB row-count queries, a
  progress log) rather than only reporting status yourself — this matters more
  when the operation could plausibly hang. Prefer designs where partial progress
  is visibly committed/observable as it happens over designs that are only
  correct-looking once complete and opaque until then.
- Only commit to git when explicitly asked — true for both Phase 1 (committed
  only after being asked) and Phase 2 (left uncommitted, verified only, as of
  this note).
