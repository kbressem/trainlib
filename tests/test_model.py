import unittest

import torch
from test_utils import TEST_CONFIG_SEGM, TEST_IMAGE

from trainlib.model import get_model


class TestModel(unittest.TestCase):
    config = TEST_CONFIG_SEGM
    image = TEST_IMAGE

    def test_init(self):
        model = get_model(self.config)
        model_name = list(self.config.model.keys())[0]
        n_classes = self.config.model[model_name].out_channels
        self.assertIsInstance(model, torch.nn.Module)
        output = model(self.image)
        self.assertEqual(output.shape[1], n_classes)


if __name__ == "__main__":
    unittest.main()
