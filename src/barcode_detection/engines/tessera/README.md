# Tessera

Tessera is the recommended fast product line. It combines tensor-based 1-D
proposals with the validated classical 2-D route.

Implementation:

- Detection: [`detection.py`](detection.py)
- Extraction: [`extraction.py`](extraction.py)
- Shared components: [`../common/`](../common/__init__.py)

The implementation and shared runtime are local to this repository. The
compatibility imports under `integrations/codara/engines/` point back here.
