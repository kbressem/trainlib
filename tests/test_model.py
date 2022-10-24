import unittest

import torch
from test_utils import TEST_CONFIG, TEST_IMAGE

from trainlib.model import get_model


class TestModel(unittest.TestCase):
    config = TEST_CONFIG
    image = TEST_IMAGE

    def test_init(self):
        model = get_model(self.config)
        self.assertIsInstance(model, torch.nn.Module)
        output = model(self.image)
        self.assertEqual(output.shape[1], self.config.model.out_channels)


if __name__ == "__main__":
    unittest.main()
