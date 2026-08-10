# Engine boundaries

This package is the active source tree for the two final product lines. The
catalog, contracts, recovery pipeline, native adapter, and product engines are
owned by this repository. No Codara checkout or `CODARA_BACKEND_ROOT` setting is
required.

- [`tessera/`](tessera/README.md) contains the recommended tensor-first
  detector and extractor.
- [`mosaic/`](mosaic/README.md) contains the guarded robust detector and
  extractor.
- [`common/`](common/README.md) contains the shared recovery pipeline,
  classical and tensor components, native acceleration, review gates, and the
  stable contract bridge used by both lines.
- [`registry.py`](registry.py) is the production selection surface.

The old [`integrations/codara`](../integrations/codara/README.md) path is now a
thin compatibility boundary. Its selected filenames re-export these local
implementations so existing Codara imports do not silently fork the source.
