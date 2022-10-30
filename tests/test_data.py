import tempfile
import unittest
from pathlib import Path

import monai
from monai.data import DataLoader
from test_utils import TEST_CONFIG

from trainlib.data import import_dataset, segmentation_dataloaders


class TestDatasetInit(unittest.TestCase):
    config = TEST_CONFIG

    def test_init_iterative_dataset(self):
        """Test that dataset is of correct instance"""
        self.config.data.dataset_type = "iterative"
        dataset = import_dataset(self.config)(data=[], transform=[])
        self.assertIsInstance(dataset, monai.data.Dataset)
        self.assertFalse(Path(self.config.data.cache_dir).exists())

    def test_init_persistent_dataset(self):
        """Test that dataset is of correct instance and cache dir is created"""
        self.config.data.dataset_type = "persistent"
        with tempfile.TemporaryDirectory() as tempdir:
            self.config.data.cache_dir = f"{tempdir}/.cache"
            dataset = import_dataset(self.config)(data=[], transform=[])
            self.assertTrue(Path(self.config.data.cache_dir).exists())
        self.assertIsInstance(dataset, monai.data.PersistentDataset)


class TestSegmentationDataLoaders(unittest.TestCase):
    config = TEST_CONFIG

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.data.train = True
        self.config.data.valid = True
        self.config.data.test = False
        self.config.data.dataset_type = "iterative"

    def test_init(self):
        dataloaders = segmentation_dataloaders(self.config)
        self.assertTrue(len(dataloaders) == 2)
        self.assertIsInstance(dataloaders[0], DataLoader)
        self.assertIsInstance(dataloaders[1], DataLoader)

    def test_show_batch(self):
        dataloaders = segmentation_dataloaders(self.config)
        self.assertTrue(hasattr(dataloaders[0], "show_batch"))
        self.assertTrue(hasattr(dataloaders[1], "show_batch"))


if __name__ == "__main__":
    unittest.main()
