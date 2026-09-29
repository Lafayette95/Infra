"""The scheduled daily cycle (CLAUDE.md section 12): px -> raw -> derived -> bmk -> backup,
each step followed by its own regression checks. Plain Python - Prefect only wraps it
(infra/cycle/flows.py), so every piece is testable without a scheduler."""
