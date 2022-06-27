import unittest

import munch

from trainlib.utils import load_config


class TestLoadConfig(unittest.TestCase):
    def test_load(self):
        "Test that config is a dict and munch.Munch object"
        config = load_config("test_config.yaml")
        self.assertIsInstance(config, dict)
        self.assertIsInstance(config, munch.Munch)

    def test_listification(self):
        "Test that image_cols and label_cols are converted to lists"
        config = load_config("test_config.yaml")
        self.assertIsInstance(config.data.image_cols, list)
        self.assertIsInstance(config.data.label_cols, list)

    def test_mode(self):
        "Test that length of mode paramter is same as numer of image + label cols"
        config = load_config("test_config.yaml")
        self.assertEqual(
            len(config.transforms.mode),
            len(config.data.image_cols + config.data.label_cols),
        )

    def test_paths(self):
        "Test that paths for output are specified correctly"
        config = load_config("test_config.yaml")
        self.assertIn(config.run_id, config.out_dir)
        self.assertIn(config.run_id, config.log_dir)


if __name__ == "__main__":
    unittest.main()
