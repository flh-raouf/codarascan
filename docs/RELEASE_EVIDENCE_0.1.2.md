# CodaraScan 0.1.2 release evidence

Date: 2026-08-10

Scope: documentation refresh, security hardening, public PyPI publication, and
Codara consumer verification for CodaraScan 0.1.2. This record distinguishes
completed package publication from the remaining GitHub source-visibility and
downstream merge gates.

## Source and quality gates

Source commit: `826998bbb21c41ed7bfe4191380de72ef80fe518`

The release source passed the following local gates:

```text
Ruff:                         passed
mypy:                         passed
pytest:                       187 passed
wheel and sdist build:        passed
artifact inspection:          passed
git diff --check:             passed
```

GitHub CI run
[`31397915733`](https://github.com/flh-raouf/codarascan/actions/runs/31397915733)
passed the quality and distribution jobs, CPython 3.11-3.14 tests, and platform
smoke tests on Linux x86-64/ARM64, macOS Intel/Apple Silicon, and Windows
x86-64.

The 0.1.2 hardening changes make explicit CLI debug directories owner-only
(`0700`) and their warning/result files owner-readable and owner-writable only
(`0600`), including under a permissive process umask. All third-party GitHub
Actions in the package CI and release workflows are pinned to full commit
SHAs. A regression test rejects mutable action references, and repository
Actions policy also requires SHA-pinned actions.

## Published artifacts and provenance

Release workflow run
[`31398141966`](https://github.com/flh-raouf/codarascan/actions/runs/31398141966)
completed successfully using PyPI Trusted Publishing. It built and inspected
the native-required source distribution and 20 wheels, then required a complete
21-artifact manifest before publication.

PyPI exposes the release at
<https://pypi.org/project/codarascan/0.1.2/>. The published description is the
0.1.2 README, including the installation, API, PDF/batch, format, platform,
privacy, FAQ, SEO/GEO, contact, and limitation sections. PyPI displays Sigstore
provenance for the artifacts and binds publication to the source commit and
`release.yml` workflow above.

The published source distribution is:

```text
f58e35058f98a73a407ddf2f5b705c54f02f9d246709b4ba7962daeddb94bbc8  codarascan-0.1.2.tar.gz
```

## Independent registry verification

A new virtual environment outside the repository installed the exact version
from the public simple index with no wheel cache:

```bash
python -m pip install --no-cache-dir --index-url https://pypi.org/simple \
  codarascan==0.1.2
codarascan --version
# codarascan 0.1.2 (Apache-2.0)
```

The imported module resolved from the environment's `site-packages`, not the
source checkout. The installed package also reproduced the owner-only debug
directory and file modes.

## Codara consumer verification

Codara migration commit `031ee468f5f5addc7c549bb94413cd5b73276941`
pins `codarascan==0.1.2` and commits a registry-backed production lock. A clean
`uv --no-sources --locked` environment imported 0.1.2 from `site-packages` and
passed all 75 backend tests.

Codara CI run
[`31399715346`](https://github.com/flh-raouf/codara/actions/runs/31399715346)
passed backend checks, frontend typechecking/tests/build, and the production
container build. Its production deployment job was deliberately skipped for
the manual migration-branch run.

## Public-source safety gate

The distributable package and PyPI release are public. The GitHub repository is
intentionally private while GitHub Support removes three immutable pull-request
refs that still retain objects from before the sanitized history rewrite.
Normal branches and tags expose only the sanitized history. The source
repository must not be made public until GitHub confirms the server-side purge
and cached-view cleanup.

The preserved research tree also passed a targeted publication-residue scan:
it contains no personal filesystem paths, internal document names, `VN LOT`
labels, or assistant-environment cache names. Research notebook cache examples
now use the neutral `barcode-research` label, and private benchmark groups use
generic corpus names. All retained synthetic benchmark PDFs have an empty
Author field and only anonymous or generic dataset-generator Creator metadata.
The maintainer name and contact email remain intentionally public in package
metadata, the README, and the license notices.

The repository's issue #3 was deleted. Main rejects force-pushes and deletions,
enforces administrators, linear history, resolved conversations, pull-request
flow, and the complete up-to-date CI matrix. Dependabot and automated security
updates are enabled. Code scanning, secret scanning, and push protection remain
platform/plan-dependent while the repository is private and should be enabled
immediately after the visibility gate is cleared.

The standard security review reported two medium findings; both were fixed in
0.1.2 and their original reproductions no longer succeed. The environment did
not provide the managed read-only profile required by the deeper independent
scan, so this record does not claim that unavailable coverage.
