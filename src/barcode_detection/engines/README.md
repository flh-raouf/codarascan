# Engine boundaries

The engine catalog is dependency-free and safe for tooling. Implementations are separated by product line:

- [`tessera/`](tessera/README.md) is the recommended tensor-first fast path.
- [`mosaic/`](mosaic/README.md) is the guarded robust fallback.

The current runnable adapters are still in the Codara integration snapshot until their application dependencies are migrated into these boundaries.
