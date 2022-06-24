import unittest
import munch
from trainlib.utils import load_config


class TestLoadConfig(unittest.TestCase):
    def test_load(self):
        config = load_config("test_config.yaml")
        self.assertIsInstance(config, dict)
        self.assertIsInstance(config, munch.Munch)

    def test_listification(self):
        config = load_config("test_config.yaml")
        self.assertIsInstance(config.data.image_cols, list)
        self.assertIsInstance(config.data.label_cols, list)

    def test_mode(self):
        config = load_config("test_config.yaml")
        self.assertEqual(
            len(config.transforms.mode),
            len(config.data.image_cols + config.data.label_cols),
        )

    def test_paths(self):
        config = load_config("test_config.yaml")
        self.assertIn(config.run_id, config.out_dir)
        self.assertIn(config.run_id, config.log_dir)


if __name__ == "__main__":
    unittest.main()
