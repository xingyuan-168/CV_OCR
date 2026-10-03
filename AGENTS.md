# CQ_AI repository instructions

The canonical repository is https://github.com/xingyuan-168/CV_OCR. Start new work
from the current source commit, using a `codex/` task branch. Preserve Git history.

- Read `docs/REQUIREMENTS.md`, `docs/SCOPE.md` and the current delivery manifest.
- Before implementation run the AI Engineering OS Start gate. Record research for
  a new module, integration, stack or major feature; do not bypass a blocked gate.
- `input/` belongs to the user and is read-only. Never change, rename or remove it.
- `output/` contains exactly `CQ_X86.dll`, `CQ_AI_worker.exe` and
  `易语言_DLL_API_说明.html`. Stage candidates under `outpush/`, verify the paired
  binaries, and use transactional promotion with rollback.
- `release/v23.5/` and registered delivery archives are immutable. Select the
  current delivery through `release/current.json`; preserve historical evidence.
- Keep the 60 public signatures, stdcall aliases, protocol and positive
  `session_count` semantics compatible. OCR devices are 0..2; YOLO devices 0..3.
- Never terminate a user's Worker process. Tests own and clean up their processes.
- Update only affected documents. API facts come from `include/ai_engine.h`;
  generated HTML and project context are produced by their generators.
- Ordinary sequential work stays on a task branch. Risky or overlapping parallel
  work uses registered disposable worktrees; only the coordinator writes memory.
- Finish with affected tests, document and delivery checks, the real Finish gate,
  a Conventional Commit and push. Preserve unrelated user changes.

Governance commands and toolchain preparation are in `docs/GOVERNANCE_RULES.md`.
