# Test plan

Storage changes run `python scripts/check_governance.py --all --working-tree`.
Cleanup acceptance covers preview/idempotence, protected and escaping paths,
junctions, occupied files, copy/hash failures and dependency recipe/tamper checks.
Rebuild the release presets and optional module, run x86/x64 CTest sequentially,
and verify the unchanged deployed cohort through the independent x86 loader.
After cleanup compare protected snapshots and measure the entire directory.
Clean-checkout CI checks portable scripts and both native architectures; GPU
performance remains an explicit target-machine acceptance, not CI coverage.
Native CI uses the Windows2022 image with VS2022, restores the hash-verified
ORT/DirectML SDK, and tests real CPU YOLO in the x64 DLL and the x86 proxy with a
fresh paired x64 Worker. OpenCV performance and GPU checks remain local/target
acceptance. Script checks use literal Windows PowerShell5.1 and PowerShell7
steps; temporary path assertions compare resolved paths, including 8.3 aliases.

| Trigger | Required checks |
| --- | --- |
| Every PR | Manifest/hash/path checks, versions/API, docs links, governance-tool tests, Python wrapper and calibration tests |
| Build configuration change | Clean-clone x86/x64 configure/build and relevant native smoke/regression |
| New native binary | Absolute-path delivery loader, protocol/exports, paired Worker, Chinese path, CV/OCR/YOLO and installed Wheel |
| New deployment cohort | Five lanes, warmup100/sample1000, continuous and simultaneous three rounds; 30-minute changing-frame soak; safe promotion/rollback |
| NVIDIA acceptance | RTX2070 FP32/FP16 comparison, Graph, engine cache/corruption/dependency absence and GPU soak |

Governance-only changes reuse existing native evidence only when every native
source fingerprint and protected DLL/Worker/HTML hash matches. New builds get
their own hashes; matching ABI/functionality is checked rather than assuming
identical compiler timestamps or paths.

Negative tests cover path escape, file/hash tampering, protocol/version mismatch,
extra output files, missing Worker, and occupied cohort promotion. They use owned
temporary fixtures, never the user's live output. Missing GPU tests are reported
pending, not passed. CPU/DirectML availability is recorded explicitly.

The registered v23.6 evidence includes28 affected native tests,60,000 formal calls
and39,708 calls in a30-minute changing-frame soak. Errors were zero; local latency
fluctuated and target20/30ms acceptance is still pending. A diagnostic model caused
a retained allocation; the final10 minutes were stable. Retain this qualification.

HTML structure/search/navigation checks passed; visual browser preview remains
pending until explicitly inspected. Full timing: [delivery report](V23_6_DELIVERY_CN.md).
