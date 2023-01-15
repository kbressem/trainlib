import shutil
import unittest
from copy import deepcopy
from pathlib import Path

from test_utils import TEST_CONFIG_CLF, TEST_CONFIG_SEGM

from trainlib.trainer import ClassificationTrainer, SegmentationTrainer


class TestSegmentationTrainer3d(unittest.TestCase):
    config = deepcopy(TEST_CONFIG_SEGM)

    def tearDown(self) -> None:
        shutil.rmtree(self.config.run_id.split("/")[0], ignore_errors=True)
        shutil.rmtree(self.config.model_dir, ignore_errors=True)
        shutil.rmtree(self.config.data.cache_dir, ignore_errors=True)
        super().tearDown()

    def test_in_order(self):
        self._test_one_epoch()
        self._test_folder_structure()

    def _test_one_epoch(self):
        trainer = SegmentationTrainer(config=self.config)
        trainer.run()

    def _test_folder_structure(self):
        """Check that all folders for logging have been created"""
        # check runs were created
        self.assertPathExists("runs")
        # check subdirs of runs (logs, output, trainlib, patch) were created
        self.assertPathExists("runs/name/logs")
        self.assertPathExists("runs/name/logs/train_logs.csv", is_file=True)

        self.assertPathExists("runs/name/output")
        self.assertPathExists("runs/name/output/preds")
        self.assertPathExists("runs/name/output/preds/image.pt", is_file=True)
        self.assertPathExists("runs/name/output/preds/label.pt", is_file=True)
        self.assertPathExists("runs/name/output/preds/pred_epoch_1.pt", is_file=True)
        for file in [
            "emissions.csv",
            "HausdorffDistance_raw.csv",
            "MeanDice_raw.csv",
            "metrics.csv",
            "SurfaceDistance_raw.csv",
            "val_mean_dice_raw.csv",
        ]:
            self.assertPathExists(f"runs/name/output/{file}", is_file=True)

        self.assertPathExists("runs/name/trainlib")
        # check config and requirements yaml were created
        self.assertPathExists("runs/name/config.yaml", is_file=True)
        self.assertPathExists("runs/name/requirements.txt", is_file=True)

        # check model dir was created
        self.assertPathExists("models")
        # check model dir is not empty
        self.assertIsNotNone(next(Path("models").iterdir(), None))

    def assertPathExists(self, fn: str, is_file: bool = False):  # noqa N802
        path = Path(fn)
        assert path.exists(), f"File does not exist: {fn}"
        if is_file and not path.is_file():
            raise AssertionError(f"{fn} exists but does not point towards a file.")


class TestSegmentationTrainer2d(unittest.TestCase):
    config = deepcopy(TEST_CONFIG_SEGM)
    config.data.train_csv = "../data/test_data_valid_2d_segm.csv"
    config.data.valid_csv = "../data/test_data_valid_2d_segm.csv"
    config.data.test_csv = "../data/test_data_valid_2d_segm.csv"
    config.ndim = 2
    config.data.dataset_type = "iterative"

    def tearDown(self) -> None:
        shutil.rmtree(self.config.run_id.split("/")[0], ignore_errors=True)
        shutil.rmtree(self.config.model_dir, ignore_errors=True)

    def test_one_epoch(self):
        trainer = SegmentationTrainer(config=self.config)
        trainer.run()


class TestClassificationTrainer3d(unittest.TestCase):
    config = deepcopy(TEST_CONFIG_CLF)

    def tearDown(self) -> None:
        shutil.rmtree(self.config.run_id.split("/")[0], ignore_errors=True)
        shutil.rmtree(self.config.model_dir, ignore_errors=True)
        shutil.rmtree(self.config.data.cache_dir, ignore_errors=True)
        super().tearDown()

    def test_one_epoch(self):
        trainer = ClassificationTrainer(config=self.config)
        trainer.run()


class TestClassificationTrainer2d(unittest.TestCase):
    config = deepcopy(TEST_CONFIG_CLF)
    config.data.train_csv = "../data/test_data_valid_2d_clf.csv"
    config.data.valid_csv = "../data/test_data_valid_2d_clf.csv"
    config.data.test_csv = "../data/test_data_valid_2d_clf.csv"
    config.ndim = 2
    config.data.dataset_type = "iterative"

    def tearDown(self) -> None:
        shutil.rmtree(self.config.run_id.split("/")[0], ignore_errors=True)
        shutil.rmtree(self.config.model_dir, ignore_errors=True)
        super().tearDown()

    def test_one_epoch(self):
        trainer = ClassificationTrainer(config=self.config)
        trainer.run()


if __name__ == "__main__":
    unittest.main()
