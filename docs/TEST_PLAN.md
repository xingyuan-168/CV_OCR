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

Historical v23.6 HTML checks and timing remain in [its report](V23_6_DELIVERY_CN.md).
New v23.7 evidence is independent: [delivery report](V23_7_DELIVERY_CN.md).

## YOLO layout compatibility (REQ-YOLO-002)

Run `ai_engine_yolo_model_test` on x86 and x64 for protobuf bounds, ambiguity,
metadata, scalar score/class validation, threshold equality and NMS semantics.
The offline preparation tests compare unchanged graph bytes and three raw tensor
outputs, reject protected destinations and never mutate input.

Pin ORT1.24.4 and exporter dependencies. `verify_yolo_compatibility.py` compares
real v5, v11, v26 dense/end2end and dynamic v11 with an independent reference
using identical nearest-neighbor preprocessing and class-agnostic NMS. Test
path/memory loads, five concurrent callers, origin offsets, Chinese labels/paths
and mixed model handles. Old valid model JSON must match0.14.6 exactly on the
same CPU/DirectML backend. Freeze model and image hashes with every report.

New paired x86 binaries run100 warmups/1000 samples per lane,3 rounds each in
continuous and simultaneous modes. Compare old/new x64 under equal settings;
repeatable P50/P95 regression above5% blocks promotion. Run30 minutes of changing
frames with the user-provided V26 model; inspect the memory series as well as
errors. No GPU compatibility or target-performance claim without NVIDIA evidence.
