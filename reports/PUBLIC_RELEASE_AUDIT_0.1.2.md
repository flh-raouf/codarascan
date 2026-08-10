# CodaraScan 0.1.2 public-release audit

Date: 2026-08-10

## Outcome

CodaraScan 0.1.2 is built, tested, provenance-attested, and publicly available
from PyPI. The README shown by PyPI is current, and a clean Codara consumer
install passes its application suite.

The complete objective is not yet closed: public GitHub visibility is gated on
GitHub Support purging immutable pre-rewrite pull-request refs, and the Codara
migration commit is pushed but not merged into Codara's default branch.

## Requirement status

| Requirement | Status | Authoritative evidence |
|---|---|---|
| Python API, inputs, geometry, engines, statuses, diagnostics, serialization | Complete | 187-test source suite and shared public-contract tests |
| PDFium documents, ordering, streaming, concurrency, cancellation, error policies | Complete | document tests and multi-platform CI |
| Native Tessera and warning-based Python fallback | Complete | parity/fallback tests plus native wheel and sdist smoke |
| CLI and framed worker protocol | Complete | schema-backed CLI/worker tests and installed-artifact smoke |
| Forty selectable ZXing-C++ formats | Complete for the documented alpha contract | 40/40 clean-fixture evidence and published limitations |
| Packaging, licenses, attribution, typing, schemas, native source | Complete | artifact inspector and 21-file release manifest |
| Multi-platform release automation | Complete | release workflow `31398141966` |
| PyPI 0.1.2 and current README | Complete | PyPI release page, JSON metadata, and clean no-cache install |
| Confidential-data cleanup in normal Git history and distributable artifacts | Complete | rewritten sanitized branch, artifact privacy checks, and source security review |
| Immutable GitHub pull-request object purge | Pending external platform action | three remote `refs/pull/*/head` refs remain; repository stays private |
| Codara package migration implementation | Complete on migration branch | commit `031ee46`, registry lock, 75 tests, CI run `31401474599` |
| Codara default-branch integration | Pending repository merge | migration branch is pushed; its earlier pull request is closed and unmerged |

Detailed commands and release identifiers are recorded in
[`docs/RELEASE_EVIDENCE_0.1.2.md`](../docs/RELEASE_EVIDENCE_0.1.2.md).
