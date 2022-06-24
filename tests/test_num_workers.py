import unittest
import multiprocessing
from trainlib.utils import num_workers


class TestNumWorkers(unittest.TestCase):
    def test_num_workers(self):
        workers = num_workers()
        self.assertIsInstance(workers, int)
        self.assertTrue(workers < multiprocessing.cpu_count())


if __name__ == "__main__":
    unittest.main()
