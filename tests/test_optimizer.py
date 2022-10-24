import unittest

import torch
from test_utils import TEST_CONFIG

from trainlib.model import get_model
from trainlib.optimizer import get_optimizer


class TestOptimizer(unittest.TestCase):
    config = TEST_CONFIG

    def test_init(self):
        model = get_model(self.config)
        optim = get_optimizer(model, self.config)
        self.assertIsInstance(optim, torch.optim.Optimizer)


if __name__ == "__main__":
    unittest.main()
