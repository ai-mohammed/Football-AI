import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.train_football import prepare_dataset


class DatasetTests(unittest.TestCase):
    def make_dataset(self, root, duplicate=False, malformed=False):
        for split in ("train", "valid", "test"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir()
            (root / split / "images/frame.jpg").write_bytes(b"same" if duplicate else split.encode())
            (root / split / "labels/frame.txt").write_text("0 .5 .5 .2 .2" if not malformed else "3 nan")
        source = root / "data.yaml"
        source.write_text(yaml.safe_dump({"names": ["ball"], "nc": 1,
                                          "train": "../train/images", "val": "../valid/images",
                                          "test": "../test/images"}))
        return source

    def test_roboflow_paths_normalized_without_modifying_original(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_dataset(root)
            original = source.read_bytes()
            normalized, audit = prepare_dataset(source, root / "output", "ball")
            self.assertEqual(audit["counts"], {"train": 1, "val": 1, "test": 1})
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(Path(yaml.safe_load(normalized.read_text())["val"]), root / "valid/images")

    def test_cross_split_duplicate_blocks_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_dataset(root, duplicate=True)
            with self.assertRaisesRegex(ValueError, "audit failed"):
                prepare_dataset(source, root / "output", "ball")

    def test_invalid_labels_block_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_dataset(root, malformed=True)
            with self.assertRaisesRegex(ValueError, "audit failed"):
                prepare_dataset(source, root / "output", "ball")

    def test_human_pose_dataset_rejected_for_pitch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self.make_dataset(root)
            with self.assertRaisesRegex(ValueError, "32 landmarks"):
                prepare_dataset(source, root / "output", "pitch")


if __name__ == "__main__":
    unittest.main()
