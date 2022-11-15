""" Test if viewer can be executed. Does not verify output """

import unittest

import torch
from matplotlib import pyplot as plt
from parameterized import parameterized

from trainlib.viewer import BasicViewer, DicomExplorer, ListViewer

plt.ion()

TEST_IMAGE_3D = torch.randn(9, 9, 9)
TEST_IMAGE_2D = torch.randn(9, 9)
TEST_IMAGE_2D_RGB = torch.randn(3, 9, 9)

TEST_LABEL_3D = torch.rand(9, 9, 9).round()
TEST_LABEL_2D = torch.rand(9, 9).round()


TEST_CASE1 = [TEST_IMAGE_3D, TEST_LABEL_3D, None, None, None]
TEST_CASE2 = [TEST_IMAGE_2D, TEST_LABEL_2D, None, None, None]
TEST_CASE3 = [TEST_IMAGE_2D_RGB, TEST_LABEL_2D, None, None, "RGB"]

TEST_CASE4 = [TEST_IMAGE_3D, "Text Label", "Text Prediction", None, None]
TEST_CASE5 = [TEST_IMAGE_2D, "Text Label", "Text Prediction", None, None]
TEST_CASE6 = [TEST_IMAGE_2D_RGB, "Text Label", "Text Prediction", None, "RGB"]

TEST_CASE7 = [TEST_IMAGE_3D, "Text Label", 0.123, "Some Description", None]
TEST_CASE8 = [TEST_IMAGE_2D, 2, "Text Prediction", "Some Description", None]
TEST_CASE9 = [TEST_IMAGE_2D_RGB, 1, 1, "Some Description", "RGB"]

TEST_CASE10 = [TEST_IMAGE_3D, None, None, None, None]


TEST_CASES = [
    TEST_CASE1,
    TEST_CASE2,
    TEST_CASE3,
    TEST_CASE4,
    TEST_CASE5,
    TEST_CASE6,
    TEST_CASE7,
    TEST_CASE8,
    TEST_CASE9,
]


class TestViewer(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_basic_viewer(self, image, label, prediction, description, mode):
        viewer = BasicViewer(x=image, y=label, prediction=prediction, description=description, mode=mode)
        viewer.show()
        plt.close("all")

    @parameterized.expand(TEST_CASES)
    def test_dicom_explorer(self, image, label, prediction, description, mode):
        viewer = DicomExplorer(x=image, y=label, prediction=prediction, description=description, mode=mode)
        viewer.show()
        plt.close("all")

    @parameterized.expand(TEST_CASES)
    def test_list_viewer(self, image, label, prediction, description, mode):

        if label is not None:
            label = [label, label]
        if prediction is not None:
            prediction = [prediction, prediction]
        if description is not None:
            description = [description, description]

        viewer = ListViewer(x=[image, image], y=label, prediction=prediction, description=description, mode=mode)
        viewer.show()
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
