import os
import tempfile
import unittest

import monai
import test_utils

from trainlib.data import import_dataset


class TestDatasetInit(test_utils.TestCase):
    def test_init_iterative_dataset(self):
        "Test that dataset is of correct instance"
        self.config.data.dataset_type = "iterative"
        dataset = import_dataset(self.config)(data=[], transform=[])
        self.assertIsInstance(dataset, monai.data.Dataset)
        self.assertFalse(os.path.exists(self.config.data.cache_dir))

    def test_init_persistent_dataset(self):
        "Test that dataset is of correct instance and cache dir is created"
        self.config.data.dataset_type = "persistent"
        with tempfile.TemporaryDirectory() as tempdir:
            self.config.data.cache_dir = f"{tempdir}/.cache"
            dataset = import_dataset(self.config)(data=[], transform=[])
            self.assertTrue(os.path.exists(self.config.data.cache_dir))
        self.assertIsInstance(dataset, monai.data.PersistentDataset)


if __name__ == "__main__":
    unittest.main()
