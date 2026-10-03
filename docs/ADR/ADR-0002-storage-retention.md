# ADR-0002: Retain one optional NVIDIA archive outside the repository

Status: accepted by the user, 2026-10-03.

The approximately19.79GiB working directory was dominated by repeated optional
runtime packages, expanded SDKs and disposable build trees. Git was0.21GiB and
native src0.73MiB. Directory naming and ignore rules did not reduce disk usage.

Retain the latest hash-verified NVIDIA ZIP once in a separate delivery directory;
actually delete registered duplicate/cache files. Preserve input, output, Git
history, release archives, rollback and evidence. Pin dependency sources and
verify installed cache recipes/file hashes before deciding to rebuild.

The completed workspace budget is1.5GiB including Git and retained dependencies.
Report external retained bytes separately from deleted bytes. Recovery may
temporarily exceed the budget; final cleanup must measure the directory again.
Native ABI and deployed files remain the registered0.14.6/v23.6/protocol26 cohort.
