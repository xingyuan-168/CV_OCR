# Requirements

## REQ-GOV-001 — Govern the delivered v23.6 source

Use the delivered 0.14.6 / v23.6 / Worker protocol26 files and source commit
`5e05f7f86d7a361f52cdfa459f61d1eb3f8db8bf` as the baseline. The user's selected
canonical repository is CV_OCR. Preserve input data, the three deployed files,
their hashes, and the original v23.5 delivery.

Acceptance: a portable current manifest; clean-clone build and checks; one
packaging entry point; accurate examples; CI; observable AIOS Start/Finish results;
and a reviewable governance commit on CV_OCR.

Storage acceptance: retain the verified NVIDIA optional ZIP once outside the
repository, actually delete registered duplicated artifacts/caches, preserve
protected hashes, and keep the finished workspace including Git below1.5GiB.
Report deleted bytes separately from externally retained bytes.

## REQ-YOLO-001 — Target-machine performance

Five callers share one model handle on E5-2696 v4 / 128GB / RTX2070, with five
business windows. Each lane's complete call must reach P50 <=20ms and P95 <=30ms.
The frozen model is FP32 input [1,3,320,320], output [1,10,2100], six classes;
confidence0.5 and NMS0.45. Time the caller with QPC, including communication.

FP16 acceptance requires equal counts/classes, matched coordinate error <=1px
and score difference <=0.02, including critical targets and boundary cases.
Graph requires correctness and repeatable P95 improvement >=10% in each round.
Local functional evidence is not target-machine performance acceptance.

Current status is usable, functional validation passed in its recorded scope;
NVIDIA hardware execution and target performance remain pending.
