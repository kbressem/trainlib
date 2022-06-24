import unittest

import tempfile
from trainlib.utils import import_patched


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
