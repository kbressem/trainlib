import unittest
from test_utils import TestCase
from trainlib.data import segmentation_dataloaders
from monai.data import DataLoader


class TestSegmentationDataLoaders(TestCase):
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
