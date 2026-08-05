import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / "benchmarks/lab/datasets.json"
FETCHER_PATH = ROOT / "benchmarks/lab/fetch_data.py"


def load_fetcher():
    specification = importlib.util.spec_from_file_location("barcode_benchmark_fetch_data", FETCHER_PATH)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {FETCHER_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class DatasetRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetcher = load_fetcher()
        cls.registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    def test_registry_matches_supported_schema(self):
        self.fetcher.validate_registry(self.registry)

    def test_dataset_ids_and_artifact_paths_are_unique(self):
        datasets = self.registry["datasets"]
        self.assertEqual(len({dataset["id"] for dataset in datasets}), len(datasets))
        paths = [artifact["path"] for dataset in datasets for artifact in dataset["artifacts"]]
        self.assertEqual(len(set(paths)), len(paths))
        for path in paths:
            self.assertFalse(Path(path).is_absolute())
            self.assertNotIn("..", Path(path).parts)

    def test_sources_are_stable_https_pages(self):
        for dataset in self.registry["datasets"]:
            source = dataset["source"]
            self.assertTrue(source["page"].startswith("https://"), dataset["id"])
            for artifact in dataset["artifacts"]:
                if source["kind"] == "url":
                    self.assertTrue(artifact["url"].startswith("https://"), artifact["path"])

    def test_all_artifacts_have_sha256_checksums(self):
        for dataset in self.registry["datasets"]:
            for artifact in dataset["artifacts"]:
                self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$", artifact["path"])


if __name__ == "__main__":
    unittest.main()
