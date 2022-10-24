import inspect
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
        [torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3), torch.full((1, 3, 3, 3), 2)],
        1,
    )


def func_req_additional_args(func):
    "Checks if non-default arguments exist in function"
    signature = inspect.signature(func)
    return any([v.default is inspect.Parameter.empty and "args" not in k for k, v in signature.parameters.items()])
