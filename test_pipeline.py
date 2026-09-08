import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate_data import build_dataset  # noqa: E402


class DataGenerationTests(unittest.TestCase):
    def test_generated_dataset_shape_and_labels(self):
        data = build_dataset(rows=500, seed=7)
        self.assertEqual(len(data), 500)
        self.assertTrue(data["flow_id"].is_unique)
        self.assertEqual(set(data["label"]), {"normal", "malicious"})
        self.assertEqual(len(data.columns), 26)

    def test_generated_dataset_is_reproducible(self):
        left = build_dataset(rows=100, seed=8)
        right = build_dataset(rows=100, seed=8)
        self.assertTrue(left.equals(right))


if __name__ == "__main__":
    unittest.main()
