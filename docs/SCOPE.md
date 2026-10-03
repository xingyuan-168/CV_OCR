# Scope

This task changes repository governance, build/packaging tools, CI and affected
documentation. It registers the existing v23.6 delivery and its actual evidence.
Native inference sources and the delivered DLL/Worker/HTML remain the baseline.

The AIOS directory-preservation compatibility patch is isolated from the user's
AI-OS working directory and is recorded with its upstream revision and patch hash.

User input, unrelated projects and business Worker processes are protected.
Historical packages stay immutable. Model and test fixture retention follows
actual build/test use. Temporary build and dependency state stays outside Git.

Target RTX2070 performance tuning remains a separate acceptance task using the
delivered one-click tools and real business BMPs. Shared memory is conditional
on isolated target-machine communication P95 exceeding2ms.
