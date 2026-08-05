# Applications

The application layer is intentionally separate from reusable engine code. The current API, CLI, and TUI were developed in Codara and are not present as complete source trees in this checkout.

- [`api/`](api/README.md) — service boundary and request/response adapters.
- [`cli/`](cli/README.md) — batch and operator command-line boundary.
- [`tui/`](tui/README.md) — interactive terminal interface boundary.

Application code should depend on `barcode_detection.core` and the engine catalog, not on `archive/`.
