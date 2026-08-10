# Security and privacy policy

CodaraScan processes untrusted image and PDF inputs through native libraries.
Keep CodaraScan and its bounded dependencies patched, isolate high-risk
workloads where appropriate, and apply deployment-level byte, pixel, page,
memory, CPU, timeout, and worker quotas. The library does not impose hidden
resource caps.

Core operation is local and offline: there is no telemetry, update check,
runtime download, server, or network protocol. Default scans create no
persistent artifacts. Explicit debug output may contain images, geometry,
diagnostics, decoded text, and raw payload bytes and must be protected as
sensitive data.

Report a suspected vulnerability privately through the repository's GitHub
security-advisory channel. Do not attach confidential customer documents,
credentials, or production payloads; provide a synthetic reproducer when
possible. Public issues are appropriate for ordinary correctness bugs without
sensitive material.

Security support currently covers the latest 0.x release on the interpreter
and platform matrix in `docs/COMPATIBILITY.md`.
