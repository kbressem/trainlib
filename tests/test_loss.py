import inspect
import unittest

import monai
import torch
from parameterized import parameterized
from test_utils import TestCase, func_req_additional_args

from trainlib.loss import get_loss

monai_losses = [item[0] for item in inspect.getmembers(monai.losses) if item[0].endswith("Loss")]
torch_losses = [item[0] for item in inspect.getmembers(torch.nn) if item[0].endswith("Loss")]
monai_losses = [x for x in monai_losses if not func_req_additional_args(getattr(monai.losses, x))]
torch_losses = [x for x in torch_losses if not func_req_additional_args(getattr(torch.nn, x))]

LOSSES = monai_losses + torch_losses


class TestLoss(TestCase):
    @parameterized.expand(LOSSES)
    def test_init(self, loss_name):
        "Simple test if losses can be initialized"
        config = self.config
        config.loss = {loss_name: {}}
        loss = get_loss(config)
        self.assertTrue(callable(loss))
        self.assertEqual(loss.__class__.__name__, loss_name)


if __name__ == "__main__":
    unittest.main()
