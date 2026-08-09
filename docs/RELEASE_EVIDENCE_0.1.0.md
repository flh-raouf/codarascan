# CodaraScan 0.1.0 release evidence

Date: 2026-08-09

Scope: local pre-publication verification on macOS with CPython 3.13.7. This
record does not claim that the protected multi-platform release workflow or
PyPI publication has run.

## Package source gates

```bash
.venv/bin/ruff check src tests tools benchmarks/format_smoke.py setup.py
# All checks passed!

.venv/bin/mypy src
# Success: no issues found in 50 source files

.venv/bin/pytest --cov=codarascan --cov-report=term -q
# 185 passed in 4.63s; total coverage 46%

git diff --check
# exit 0
```

## Local native artifacts

The native extension was required during both construction and sdist
installation:

```bash
CODARASCAN_REQUIRE_NATIVE=1 /tmp/codarascan-build-313/bin/python -m build
# Successfully built codarascan-0.1.0.tar.gz and
# codarascan-0.1.0-cp313-cp313-macosx_10_13_universal2.whl

.venv/bin/python tools/inspect_artifacts.py \
  dist/codarascan-0.1.0-cp313-cp313-macosx_10_13_universal2.whl \
  dist/codarascan-0.1.0.tar.gz
# artifact inspection passed for both artifacts

python tools/artifact_smoke.py
# passed from clean wheel and sdist virtual environments outside the source tree

PYTHONPATH="$PWD" python benchmarks/format_smoke.py --mode robust --repeats 3 \
  --output benchmarks/results/format_smoke_macos_arm64_py313_0.1.0.json
# 40/40 advertised formats decoded their exact payload on the installed wheel;
# zero missing formats and zero wrong payloads
```

The installed-artifact smoke covered native import, forced Python fallback and
its single deterministic warning, fast and robust QR, fast Code 128, PDFium
document scanning, CLI JSON/version output, and worker capabilities/shutdown.
Artifact inspection covered licensing and notices, schemas and golden examples,
`py.typed`, native source and binary presence, safe archive paths and member
sizes, SPDX headers, forbidden corpora/caches/build outputs, credential
patterns, local machine paths, and exact package metadata. SHA-256,
declared-dependency, and complete-release-set evidence is generated in
`dist/dependency-manifest.json` and `dist/release-manifest.json` after the
definitive build.

## Codara consumer gates

Codara was refreshed from a non-editable sibling build and imported
CodaraScan from its backend virtual environment's `site-packages` directory.

```bash
cd ../codara-app/apps/backend
uv sync --refresh-package codarascan
.venv/bin/python -m pytest -q
# 75 passed in 0.77s

cd ../..
../barcode-detection/.venv/bin/ruff check --select E4,E7,E9,F,I apps/backend
# All checks passed!

cd apps/frontend
bun run typecheck
bun run test
bun run build
# typecheck passed; 56 tests passed; Vite built 102 modules

cd ../..
docker compose config --quiet
git diff --check
# both exit 0
```

Static source inspection found no Codara imports of copied barcode components,
recovery code, product implementations, native implementation modules, or
CodaraScan internals. Codara exposes Tessera/Mosaic extraction/localization
through an adapter that imports only the public `codarascan` API. Its tests also
validate both a real adapter result and a packaged golden example against the
package-owned JSON Schema.

## Hosted release and publication

```bash
gh workflow run release.yml --ref main -f publish=true
# https://github.com/flh-raouf/codarascan/actions/runs/31319314031
# success: 20 wheels, one sdist, release evidence, and trusted PyPI publish
```

The protected workflow completed on commit `f901cdc` after the repository was
renamed to `flh-raouf/codarascan`. CPython 3.11-3.14 native wheels passed on
Linux x86_64/ARM64, macOS Intel/Apple Silicon, and Windows x86_64. The native-
required sdist install, 21-artifact completeness/checksum gate, and PyPI trusted
publishing also passed. PyPI exposes version 0.1.0 with 20 wheels and one sdist
at <https://pypi.org/project/codarascan/0.1.0/>. The published sdist SHA-256 is
`976e9a1fc4986ba41084650c8c0f14b7902bc4bd2dd3916e84e40bd6632b871f`.

Codara's production lock was then refreshed from PyPI and verified in two fresh
environments. A no-cache, no-sources install ran a representative Tessera QR
adapter decode, and a locked no-sources install contained no `direct_url.json`,
proving registry-wheel installation. Hosted Codara run
<https://github.com/flh-raouf/codara/actions/runs/31319896640> passed all 75
backend tests, frontend checks, and the production container build from commit
`3490b1d`.
