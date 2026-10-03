# ADR-0001 — Delivered baseline and governance

Date:2026-10-03. Status:accepted. Requirement:REQ-GOV-001.

The user selected CV_OCR as the sole canonical repository and authorized the
governance plan. Base the task on delivered source5e05f7f, rather than old main.
Register v23.6 as usable with target performance pending; preserve v23.5 history.

Use official AIOS gates with a minimal, independently tested directory-preservation
patch. A stock initializer adds `.gitkeep` to protected data/output directories,
conflicting with the three-file deployment contract. Fix the observable tool
behavior; do not disable gates or add a fourth deployed file.

Portable manifests drive current checks. Preserve the actual pre-commit build
snapshot and record the later source commit separately. Source hashes normalized
to Git text bytes support CRLF/LF checkouts; raw historical build hashes remain
evidence. Dependency preparation is version/hash locked.

Keep native inference sources and current binaries unchanged in this task. Their
existing validation applies by exact fingerprints; governance tooling receives
new checks and clean-clone verification. GPU performance requires target evidence.
