# Mosaic

Mosaic is the robust base/fallback product line. Its guarded v3 extractor
provides physical review gates and recovery behavior for difficult pages; the
detection adapter exposes the same candidate path without payloads.

Implementation:

- Detection: [`detection.py`](detection.py)
- Extraction: [`extraction.py`](extraction.py)
- Shared components: [`../common/`](../common/__init__.py)

The implementation and shared runtime are local to this repository. The
compatibility imports under `integrations/codara/engines/` point back here.
