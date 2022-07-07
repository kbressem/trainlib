import multiprocessing
import tempfile
import unittest

import munch

from trainlib.utils import import_patched, load_config, num_workers


class TestNumWorkers(unittest.TestCase):
    def test_num_workers(self):
        "Test that num_workers returns an integer that is smaller than the max. amount of workers"
        workers = num_workers()
        self.assertIsInstance(workers, int)
        self.assertTrue(workers < multiprocessing.cpu_count())


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
            len(config.transforms.mode), len(config.data.image_cols + config.data.label_cols),
        )

    def test_paths(self):
        "Test that paths for output are specified correctly"
        config = load_config("test_config.yaml")
        self.assertIn(config.run_id, config.out_dir)
        self.assertIn(config.run_id, config.log_dir)


class TestImportPatched(unittest.TestCase):
    def test_import_patched(self):
        "Test if function can be sucessfully overwritten by import_patched"

        def return_five():
            return 5

        self.assertEqual(return_five(), 5)
        with tempfile.TemporaryDirectory() as tempdir:
            fn = f"{tempdir}/tmp.py"
            with open(fn, "w+") as f:
                f.write("def return_six(): return 6\n\n")
            return_five = import_patched(fn, "return_six")

        self.assertEqual(return_five(), 6)


if __name__ == "__main__":
    unittest.main()
