# Known limitations in 0.2.0 alpha

- Degraded-corpus validation is deepest for established QR Code, Data Matrix,
  and common linear workflows. Broader catalog blur, glare, perspective,
  damage, inversion, quiet-zone, false-positive, and latency baselines remain
  in progress.
- Mosaic decodes every clean 0.1.0 catalog fixture. Tessera routes every clean
  matrix fixture and passes representative Code 128 smoke/parity tests, but its
  structure-tensor linear localizer is tuned for document layouts and does not
  guarantee clean-crop localization for every linear family.
- Confidence values are not calibrated probabilities and are not comparable
  between Tessera, Mosaic, and Panorama.
- Native and Python-reference Tessera geometry is tolerance-equivalent, not
  pixel-identical.
- PDF support does not expose a password parameter in 0.1. Password-protected
  or malformed PDFs raise typed document/render errors.
- Native-raster reuse is deliberately conservative. Rotated, vector, mixed,
  transformed, or multi-object pages render at 300 DPI.
- The worker protocol is private/versioned for backend adapters. It is not a
  network service, browser API, or stability promise before 1.0.
- The package does not impose resource maximums. Large inputs and high worker
  counts may exhaust memory or CPU; deployment policy belongs to the caller.
- Full official wheel execution across every interpreter/platform target is a
  release-environment gate and cannot be inferred from one local machine.
