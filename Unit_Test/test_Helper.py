import unittest
import numpy as np
from Classification import Helper
import cv2
from PIL import Image


class Test_Helper(unittest.TestCase):
    #testing opencv to PIl Object converter
    def test_cv_to_pil(self):
        #E:/HTW berlin/SWE/sweproj/team4/fb_classifier/Unit_Test/
        example = cv2.imread('fb_classifier/Unit_Test/imagetest.png')
        actualresult = Helper.cv_to_pil(example)
        expectedresult = Helper.cv_to_pil(example)
        self.assertEqual(actualresult, expectedresult)

    #testing PIL to opencv Object converter
    def test_pil_to_cv(self):
        #E:/HTW berlin/SWE/sweproj/team4/fb_classifier/Unit_Test/
        example = Image.open('fb_classifier/Unit_Test/imagetest.png')
        actualresult = Helper.pil_to_cv(example)
        expectedresult = Helper.pil_to_cv(example)
        np.array_equal(actualresult, expectedresult)

    #testing 3d to 2d Array converter
    def test_convert_to_array2d(self):
        arr1 = np.array([[[0, 1, 2],
                          [3, 4, 5]],
                         [[6, 7, 8],
                          [9, 10, 11]]])
        arr2 = np.array([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]])
        result = Helper.convert_to_array2d(arr1)
        np.array_equal(result, arr2)


if __name__ == '__main__':
    unittest.main()