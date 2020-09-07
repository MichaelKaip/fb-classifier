import unittest
import numpy as np
from Classification import KMeans, FuzzyCMeans
import cv2

class test_ModuleCMeans(unittest.TestCase):
    #test fcm_train() method
    def test_fcm_train(self):
        #first test with 4d Array
        img = np.zeros((2,3,4,4))
        c = 2
        exp = 2
        error = 0.005
        maxiter = 1000
        with self.assertRaises(Exception): FuzzyCMeans.train_model(img, c, exp, error, maxiter)

        #second test with 2D Array
        img2 = np.array([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]])
        expectedresult = (np.array([[ 2.52539673,  6.52539673, 10.52539673],
                                    [ 0.47460478,  4.47460478,  8.47460478]]),
                          np.array([[0.03411386, 0.106052  , 0.89394727, 0.96588631],
                                    [0.96588614, 0.893948  , 0.10605273, 0.03411369]]),
                          np.array([[0.43687127, 0.48291163, 0.47264038, 0.53539893],
                                    [0.56312873, 0.51708837, 0.52735962, 0.46460107]]),
                          np.array([[4.37411544, 2.64206463, 0.91001382, 0.82203698],
                                    [0.82203958, 0.91001122, 2.64206203, 4.37411284]]),
                          np.array([7.45069328, 6.40432837, 3.8057625 , 2.88382394, 2.79072625,
                                    2.78613086]),
                            6,
                            0.8722447075074815,
                            6.509833333333178e-05)
        np.array_equal(FuzzyCMeans.train_model(img2, c, exp, error, maxiter), expectedresult)

        #third test with invalid error-parameter
        error2 = -10
        with self.assertRaises(Exception): FuzzyCMeans.train_model(img, c, exp, error2, maxiter)

        #fourth test with invalid c-parameter
        c2 = 1.34
        with self.assertRaises(Exception): FuzzyCMeans.train_model(img, c2, exp, error, maxiter)

        #fifth test with invalid exp-parameter
        exp2 = 'ten'
        with self.assertRaises(Exception): FuzzyCMeans.train_model(img, c, exp2, error, maxiter)

        #sixth test with invalid maxiter-parameter
        maxiter2 = 30136548.4572
        with self.assertRaises(Exception): FuzzyCMeans.train_model(img, c, exp, error, maxiter2)

    def test_build_model(self):
        #testing generate_fcm_model() Method using creating a list from examples
        img3 = np.array([[ 0,  1,  2,  3],[ 4,  5,  6,  7],[ 8,  9, 10, 11]])
        img4 = np.array([[ 12,  13,  14,  15],[ 16,  17,  18,  19],[ 20,  21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans.build_model(datalist)
        expectedresult = [([[ 8.52611378, 12.52611378, 16.52611378],
        [6.47390978, 10.47390978, 14.47390978]]), ([[0.03399879, 0.10621365, 0.89377492, 0.96600387],
        [0.96600121, 0.89378635, 0.10622508, 0.03399613]]), ([[0.60402916, 0.39790087, 0.54240983, 0.64458746],
        [0.39597084, 0.60209913, 0.45759017, 0.35541254]]), ([[4.37535742, 2.64330661, 0.9112558 , 0.82079501],
        [0.82083582, 0.91121499, 2.6432658 , 4.37531661]]), 6.5, 0.8722215612701494, 0.00318839999999998]
        np.array_equal(result, expectedresult)

    def test_get_mean_cntrs(self):
        #testing generate_get_mean_cntrs() Method using a list from examples
        img3 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        img4 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans._get_mean_cntrs(datalist)
        expectedresult = [[ 6.47334461, 10.47334461, 14.47334461],
                          [ 8.52661966, 12.52661966, 16.52661966]]
        np.array_equal(result, expectedresult)

    def test_get_mean_fcpm(self):
        #testing generate_get_mean_fcpm() Method using a list from examples
        img3 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        img4 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans._get_mean_fcpm(datalist)
        expectedresult = [[0.03393459, 0.10635908, 0.8937884, 0.9660312],
                          [0.96606541, 0.89364092, 0.1062116, 0.0339688]]
        np.array_equal(result,expectedresult)

    def test_get_mean_fcpm0(self):
        # testing generate_get_mean_fcpm0() Method using a list from examples
        img3 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        img4 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans._get_mean_fcpm0(datalist)
        expectedresult = [[0.66447874, 0.46096216, 0.62128557, 0.38143302],
                          [0.33552126, 0.53903784, 0.37871443, 0.61856698]]
        np.array_equal(result,expectedresult)

    def test_get_mean_dist(self):
        # testing generate_get_mean_dist() Method using a list from examples
        img3 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        img4 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans._get_mean_dist(datalist)
        expectedresult = [[2.59775656, 1.77829017, 1.77829017, 2.59839586],
                          [2.59835063, 1.77821274, 1.77821274, 2.59780179]]
        np.array_equal(result, expectedresult)

    def test_get_mean_iter(self):
        # testing generate_get_mean_iter() Method using a list from examples
        img3 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        img4 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans._get_mean_iter(datalist)
        expectedresult = FuzzyCMeans._get_mean_iter(datalist)
        self.assertEqual(result, expectedresult)

    def test_get_mean_fpc(self):
        # testing generate_get_mean_fpc() Method using a list from examples
        img3 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        img4 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans._get_mean_fpc(datalist)
        expectedresult = FuzzyCMeans._get_mean_fpc(datalist)
        self.assertEqual(result, expectedresult)

    def test_get_mean_runtime(self):
        # testing generate_get_mean_runtime() Method using a list from examples
        img3 = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
        img4 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
        data1 = FuzzyCMeans.train_model(img3, 2, 2, 0.005, 1000)
        data2 = FuzzyCMeans.train_model(img4, 2, 2, 0.005, 1000)
        datalist = [data1, data2]
        result = FuzzyCMeans._get_mean_runtime(datalist)
        expectedresult = FuzzyCMeans._get_mean_runtime(datalist)
        self.assertEqual(result, expectedresult)


class test_ModuleKMeans(unittest.TestCase):
    #test determinate_clusters() method
    def test_determinate_clusters(self):
        img = cv2.imread("fb_classifier/Unit_Test/imagetest.png")
        example = KMeans.determinate_clusters(img)
        img = cv2.imread('fb_classifier/Unit_Test/imagetest.png')
        result = KMeans.determinate_clusters(img)
        expectedresult = example
        np.array_equal(result,expectedresult)

if __name__ == '__main__':
    unittest.main()