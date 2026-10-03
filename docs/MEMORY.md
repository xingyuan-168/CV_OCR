# Engineering memory

`docs/memory/memory.jsonl` is the Git-tracked fact source. Record durable decisions,
bug roots and reusable lessons with source references and status. Preserve existing
entries. Only the coordinator writes facts; subagents submit candidates. SQLite
is a derived index, rebuilt with `codex-os memory reindex`. Personal Codex memory
is outside this repository and is not changed by governance.
