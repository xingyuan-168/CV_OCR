# Governance rules

The canonical Git remote is `origin`, https://github.com/xingyuan-168/CV_OCR.
The previous YOLO repository is retained as `legacy-yolo`. Task branches use
`codex/`; main is updated through review. No history rewrite is part of governance.

Prepare the pinned AIOS runtime with `scripts/prepare_governance.ps1`. Use its
returned `codex-os.exe` path (or the configured plugin MCP) for the actual gates:

```powershell
& $AIOSExe check . --change-class major_feature --requirement-id REQ-YOLO-002 --json
& $AIOSExe finish . --change-class major_feature --requirement-id REQ-YOLO-002 --test-command "python scripts/check_governance.py --all --working-tree" --memory-written --json
```

`AIOSExe` is a task variable supplied by the preparation script. The Finish
command executes tests; keep its JSON result with the verification report.
Do not replace a blocked result with a success assertion.

For an unpromoted native task candidate, the Finish test command can pass
`--candidate-metadata outpush/<task>/build-metadata.json` to check_governance.py.
This requires a codex task branch, complete normalized source fingerprints,
functional/ABI/protocol acceptance and matching actual candidate binaries. It
still verifies all published archives/evidence and historical deliveries. Default
checks continue to require source equality with the selected current delivery.
Archive-only `package_delivery.py --reuse-current` verifies the immutable
published archives independently of a development checkout. Neither operation
promotes output or changes the current pointer; a candidate failing its declared limit stays
unpromoted with its failed performance conclusion intact.

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
