import unittest

import ignite
import torch

from trainlib.utils import load_config


def mock_one_epoch(engine, batch):
    "Mock of one training iteration. Returns 1 as loss"
    return 1


class TestCase(unittest.TestCase):
    "unittest.TestCase with additional parameters"
    config = load_config("test_config.yaml")
    engine = ignite.engine.Engine(mock_one_epoch)
    image = torch.randn(1, 1, 3, 3, 3)
    label = torch.stack(
        [torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3), torch.full((1, 3, 3, 3), 2)]
    )
