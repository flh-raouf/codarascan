# Archive

The archive preserves the project's evolution without making research or old
implementations look like active production modules.

- [`experiments/`](experiments/README.md) contains exploratory work organized
  into isolated `atoms/` and end-to-end `pipelines/`.
- [`versions/`](versions/README.md) contains the official numbered releases,
  from the deterministic locator through the structure-tensor localizer.

Archive code is retained for comparison and historical reproduction. New
production code belongs in `src/`; evaluation and reproducibility work belongs
in `benchmarks/`.

## Archive hygiene

Keep this tree limited to source, output-free notebooks, documentation,
benchmark reports, and deliberately curated historical assets. Generated
outputs, private corpora, build products, Python caches, native binaries, and
learned or downloaded model weights stay ignored or outside the archive.
Historical model directories document reproducibility and provenance without
redistributing checkpoint binaries.
