import unittest
from trainlib.utils import load_config


class TestCase(unittest.TestCase):
    "A testcase with preloaded parameters"

    def __init__(self, *args, **kwargs):
        self.config = load_config("test_config.yaml")
        super().__init__(*args, **kwargs)
