# Governance rules

The canonical Git remote is `origin`, https://github.com/xingyuan-168/CV_OCR.
The previous YOLO repository is retained as `legacy-yolo`. Task branches use
`codex/`; main is updated through review. No history rewrite is part of governance.

Prepare the pinned AIOS runtime with `scripts/prepare_governance.ps1`. Use its
returned `codex-os.exe` path (or the configured plugin MCP) for the actual gates:

```powershell
& $AIOSExe check . --change-class new_integration --requirement-id REQ-GOV-001 --json
& $AIOSExe finish . --change-class new_integration --requirement-id REQ-GOV-001 --test-command "python scripts/check_governance.py --all --working-tree" --memory-written --json
```

`AIOSExe` is a task variable supplied by the preparation script. The Finish
command executes tests; keep its JSON result with the verification report.
Do not replace a blocked result with a success assertion.

`scripts/check_repository_hygiene.ps1` supports Windows PowerShell5.1 and7 and
delegates portable checks to Python. User input and approved local build/cache
directories are classified separately from tracked pollution.

AIOS config is tracked; runtime database, logs, generated context and caches are
ignored. Input is read-only; output has exactly three delivery files. Old owned
artifacts are cleaned only after inventory and retention decisions, using checked
absolute paths. See [worktrees](WORKTREE.md) and [memory](MEMORY.md).

The directory-preservation patch changes initialization/document presence checks
for existing input/output directories. It retains all repository and test gates,
with regression coverage for populated and empty project directories.

The pinned CLI does not accept `--remote` or `--base-ref`; its repository check
resolves the configured canonical remote. See [storage policy](REPOSITORY_STORAGE_CN.md)
for the 1.5GiB retained-workspace budget, verified external archives and cache lifecycle.
