# Models

Model files remain beside the historical version or external Codara component that owns their provenance. This directory is reserved for promoted production models only.

Every promoted model should include its source commit, training/evaluation dataset, expected input shape, checksum, and license in a companion manifest. Generated or local model exports belong under `models/generated/`, which is ignored.
