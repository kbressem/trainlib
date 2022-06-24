import unittest
import multiprocessing
from trainlib.utils import num_workers


class TestNumWorkers(unittest.TestCase):
    def test_num_workers(self):
        "Test that num_workers returns an integer that is smaller than the max. amount of workers"
        workers = num_workers()
        self.assertIsInstance(workers, int)
        self.assertTrue(workers < multiprocessing.cpu_count())


if __name__ == "__main__":
    unittest.main()
