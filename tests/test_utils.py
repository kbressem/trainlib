import inspect
from pathlib import Path

import ignite
import torch

from trainlib.utils import load_config


def mock_one_epoch(engine, batch):
    """Mock of one training iteration. Returns 1 as loss"""
    return 1


TEST_CONFIG_SEGM = load_config(Path(__file__).parent / "test_config_segm.yaml")
TEST_CONFIG_CLF = load_config(Path(__file__).parent / "test_config_clf.yaml")
TEST_ENGINE = ignite.engine.Engine(mock_one_epoch)
TEST_IMAGE = torch.randn(1, 1, 3, 3, 3)
TEST_LABEL = torch.stack(
    [torch.zeros(1, 3, 3, 3), torch.ones(1, 3, 3, 3), torch.full((1, 3, 3, 3), 2)],
    1,
)


def does_func_have_non_default_args(func):
    """Checks if non-default arguments exist in function"""
    signature = inspect.signature(func)
    return any([v.default is inspect.Parameter.empty and "args" not in k for k, v in signature.parameters.items()])
