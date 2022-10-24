import unittest

import torch
from test_utils import TestCase

from trainlib.model import get_model
from trainlib.optimizer import get_optimizer


class TestOptimizer(TestCase):
    def test_init(self):
        model = get_model(self.config)
        optim = get_optimizer(model, self.config)
        self.assertIsInstance(optim, torch.optim.Optimizer)


if __name__ == "__main__":
    unittest.main()
