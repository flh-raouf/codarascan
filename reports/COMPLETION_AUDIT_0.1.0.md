# CodaraScan 0.1.0 completion audit

Date: 2026-08-09

Scope: GitHub issue #3 and its authoritative comments

Status: complete and published

## Requirement audit

| Area | Implemented evidence | Verification |
|---|---|---|
| Public API and contracts | Immutable `Scanner` configuration; convenience functions; typed results/errors; mode, symbol, decode, diagnostics, format, geometry, status, raw-byte, and serialization contracts | Public-contract tests cover every 2×3×2 scanner configuration, convenience-cache reuse, caller mutation, invalid engine output, and model/container invariants |
| Inputs and geometry | Paths, bytes-like objects, Pillow modes, NumPy arrays, EXIF normalization, normalized ROI, clockwise convex pixel and normalized quads | Input, EXIF, ROI, cross-page-size ROI, bounds, convexity, correspondence, and mutation-isolation tests |
| Engines | Tessera fast localization/extraction; Mosaic robust recovery/extraction; packaged native Tessera with deterministic once-only Python fallback warning | Integration, routing, native/fallback, parity, localization, decode, and diagnostics tests; installed artifacts import `codarascan._sttg_native` |
| Format catalog | 27 linear and 13 matrix selections exposed as concrete readable formats | Clean robust benchmark decoded 40/40 exact payloads with zero wrong payloads; Tessera matrix suite 13/13; representative fast Code 128 passes |
| Documents | PDFium rendering, ordered streaming, page selection, worker concurrency, cancellation/closure behavior, and `raise`/`collect`/`skip` policies | PDF, ordering, thread-limit, partial-error, cancellation, invalid-page, ROI, and lifecycle tests |
| CLI and worker | Versioned CLI JSON/JSONL; strict length-prefixed JSON worker; capability, scan, stream, cancel, shutdown, and protocol errors | CLI and worker suites plus clean-installed artifact smoke; worker schemas reject unknown fields, booleans as IDs, nonfinite values, oversized and malformed frames |
| Schemas | Result, stream, and worker JSON Schemas plus decoded/localized/stream golden examples | Package schema suite and Codara consumer validation of both a golden example and a real result |
| Packaging and provenance | Python 3.11–3.14 metadata, Apache-2.0 project license, notices, dependency provenance, native C++ source, SPDX headers, typed marker, package data, wheel/sdist automation | Artifact inspector validates required members, exact metadata, safe paths, size limits, SPDX, credentials/local paths, forbidden corpora/caches, and native wheel binary |
| Codara migration | Public-API-only adapter for four Tessera/Mosaic product IDs; package pin; copied engine, recovery, product, and native implementation removal | 75 backend tests, installed `site-packages` assertion, migration import scan, backend lint, frontend tests/typecheck/build, compose validation |
| Documentation and release automation | API, migration, compatibility, worker protocol, release, limitations, security, changelog, contribution, format, release-evidence, CI, and protected release workflows | Workflow YAML parse, source quality gates, artifact build/inspection/smoke, release-set manifest, and `git diff --check` |
| Hosted release | Twenty CPython 3.11-3.14 native wheels plus one sdist; PyPI trusted publisher bound to `flh-raouf/codarascan` | Release run `31319314031` passed every platform, artifact-evidence, and publish job; PyPI JSON reports 21 files |
| Registry consumer | Codara production lock pins the PyPI release and ignores the sibling development source | Fresh no-cache and locked no-sources installs had no `direct_url.json`; a representative adapter decoded successfully; hosted backend, frontend, and container jobs passed |

## Definitive local results

```text
Ruff shipped/CI scope:                         passed
mypy strict configuration:                    50 source files, no issues
pytest with coverage and native parity:       185 passed in 4.63s, 46% total
Codara backend pytest:                        75 passed in 0.77s
Codara frontend:                              56 tests; typecheck and build passed
Robust installed-wheel format benchmark:      40/40 decoded; 0 wrong payloads
Wheel artifact inspection:                    67 members, passed
Sdist artifact inspection:                    96 members, passed
Fresh wheel install and artifact smoke:        passed outside source tree
Fresh native-required sdist install/smoke:     passed outside source tree
Workflow YAML, compose, and both diff checks:  passed
```

Definitive local artifact hashes:

```text
13a608e7b5e54e69468af11f22303bf8997454433d99aeaba4ca468b5c839bf7  codarascan-0.1.0-cp313-cp313-macosx_10_13_universal2.whl
a02ea26bfc4d83ac58514044b4b8cdfded5801cad72a0e643087450fd00860dc  codarascan-0.1.0.tar.gz
```

Machine-readable records:

- `dist/dependency-manifest.json`
- `dist/release-manifest.json`
- `benchmarks/results/format_smoke_macos_arm64_py313_0.1.0.json`

## Hosted publication results

The repository is published at <https://github.com/flh-raouf/codarascan> and
PyPI release 0.1.0 is available at
<https://pypi.org/project/codarascan/0.1.0/>. Release workflow run
<https://github.com/flh-raouf/codarascan/actions/runs/31319314031> passed the
five-platform CPython 3.11-3.14 wheel matrix, native-required sdist build and
smoke, 21-artifact manifest, and trusted publication.

Codara commit `3490b1d` consumes the registry lock. Hosted workflow run
<https://github.com/flh-raouf/codara/actions/runs/31319896640> passed backend,
frontend, and production container jobs. The deploy job was correctly skipped
for the manually dispatched branch verification.
