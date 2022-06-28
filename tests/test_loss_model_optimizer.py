import unittest

import torch
from test_utils import TestCase

from trainlib.loss import get_loss
from trainlib.model import get_model
from trainlib.optimizer import get_optimizer


class TestModel(TestCase):
    def test_init(self):
        model = get_model(self.config)
        self.assertIsInstance(model, torch.nn.Module)
        output = model(self.image)
        self.assertEqual(output.shape[1], self.config.model.out_channels)


class TestLoss(TestCase):
    def test_init(self):
        loss = get_loss(self.config)
        self.assertTrue(callable(loss))
        self.assertIsInstance(loss, torch.nn.modules.loss._Loss)


class TestOptimizer(TestCase):
    def test_init(self):
        model = get_model(self.config)
        optim = get_optimizer(model, self.config)
        self.assertIsInstance(optim, torch.optim.Optimizer)


if __name__ == "__main__":
    unittest.main()
