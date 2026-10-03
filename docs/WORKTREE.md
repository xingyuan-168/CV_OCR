# Worktrees

Ordinary sequential tasks use `codex/` branches. Risky experiments or overlapping
parallel writes use AIOS-registered worktrees under `.worktrees/`. Keep user input
read-only and register build/dependency provenance. Only the coordinator writes
project memory. Review and commit before finish; remove registered worktrees only
after their changes are safely merged or retained. No orphan worktrees.

The AIOS compatibility patch uses a separate official-upstream checkout and task
branch; the user's dirty AI-OS checkout is preserved. Its upstream base and patch
are locked by `configs/governance-toolchain.json`.
