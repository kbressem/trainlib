import shutil
import unittest
from pathlib import Path

from test_utils import TEST_CONFIG

from trainlib.trainer import SegmentationTrainer


class TestSegmentationTrainer(unittest.TestCase):
    config = TEST_CONFIG

    def tearDown(self) -> None:
        shutil.rmtree(self.config.run_id.split("/")[0], ignore_errors=True)
        shutil.rmtree(self.config.model_dir, ignore_errors=True)
        shutil.rmtree(self.config.data.cache_dir, ignore_errors=True)
        ```
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


if __name__ == "__main__":
    unittest.main()
