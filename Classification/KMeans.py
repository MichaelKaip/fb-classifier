import numpy as np
from sklearn.cluster import KMeans
from Filters import HomomorphicFilter as filter
import cv2
from timeit import default_timer as timer
debug = False


class Filler:
    """
    Contains methods to remove a noise on an image

    """
    def __init__(self, img:np.ndarray, brush_color:float, size_threshold:int=30, threshold:int=256 // 2):
        """
        Creates an object of filler with special settings
        :param img:  an original image user wants to remove a noise on it.
        :param brush_color:  color will be colored a noise pixel
        :param size_threshold:  suggested  size of noise pixel.
                                If a pixel large than size, the algorithm
                                suggests it a front and does nothing with.
        :param threshold: The border of color. If a pixel color value is smaller than
                          threshold, the algorithms find it as backgroung. else
                          one is determinated as front pixel.
                          Default value is middle between black and white color value
                          (255 + 1) // 2 = 128.
        """

        if img is None:
            raise Exception('The image is None')


        if brush_color is None:
            raise Exception('The brush_color is None')
        if brush_color < 0:
            raise Exception('The brush_color is less than 0')
        if size_threshold < 0:
            raise Exception("The size threshold is less than 0")
        img2 = img.copy().astype(np.int)
        self.img = img2
        self.h, self.w = img2.shape
        self.res = img2.copy()
        self.tmp = img2.copy()
        self.size_threshold = size_threshold
        self.threshold = threshold
        self.brush_color = brush_color

    def get_fill_count(self, i, j):
        """
        Counts a neighborhoods pixel and unions them, if size of union less than
        size_threshold

        :param i: x coordinate of pixel
        :param j: y coordinate of pixel
        :return:  size of union of pixel.
        """
        current_value = self.tmp[i, j]
        cnt = 0
        if current_value < self.threshold:
            return 0
        neighborhoods = [(i, j)]
        self.tmp[i, j] = -1
        while len(neighborhoods) > 0:
            i2, j2 = neighborhoods.pop()
            cnt += 1
            tmp = [(i2 + 1, j2), (i2 - 1, j2), (i2, j2 + 1), (i2, j2 - 1)]
            for i3, j3 in tmp:
                if i3 < 0 or i3 >= self.h:
                    continue
                if j3 < 0 or j3 >= self.w:
                    continue
                if self.tmp[i3, j3] == current_value:
                    self.tmp[i3, j3] = -1
                    neighborhoods.append((i3, j3))
        return cnt

    def fill_with_black(self, i, j):
        """
        Sets a pixel by i, j a black color
        :param i:  pixel index
        :param j:  pixel index

        """
        if i < 0 or j < 0:
            raise Exception("Index of pixel is invalid")
        current_value = self.img[i, j]
        neighborhoods = [(i, j)]
        self.res[i, j] = self.brush_color
        while len(neighborhoods) > 0:
            i2, j2 = neighborhoods.pop()
            tmp = [(i2 + 1, j2), (i2 - 1, j2), (i2, j2 + 1), (i2, j2 - 1)]
            for i3, j3 in tmp:
                if i3 < 0 or i3 >= self.h:
                    continue
                if j3 < 0 or j3 >= self.w:
                    continue
                if self.res[i3, j3] == current_value:
                    self.res[i3, j3] = self.brush_color
                    neighborhoods.append((i3, j3))

    def remove_noise(self):
        """
        Removes a noise from an image.
        :return: an image without noise.
        """
        for i in range(self.h):
            for j in range(self.w):
                if self.tmp[i, j] > self.threshold:
                    cnt = self.get_fill_count(i, j)
                    if cnt == 0:
                        continue
                    if cnt < self.size_threshold:
                        self.fill_with_black(i, j)
        return self.res

#End Filler class

def img_show(img, title='Image'):
    """
    Shows an image using opencv library
    :param img:  an image
    :param title: a title of frame. Default value is "Image"
    :return:
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def determinate_clusters(img, width=-1, height=-1):
    """

    Returns the clussified image.
    Used Kmeans algorithm from sklearn.cluster
    Clussified an image on 2  clusters
    :param img:  origin image
    :param width: to resize image. Default -1. No modification
    :param height: to resize image. Default -1. No modification
    :return: clussified image
    """
    if img  is None:
        raise Exception('The image is None')

    if len(img.shape) != 3:
        raise Exception('The numpy array is invalid')
    if width > 0 and height > 0:
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = (img * 255).astype(np.uint8)
    homo_filter = filter.HomomorphicFilter(a=0.55, b=114.5)
    res = homo_filter.filter(I=img, filter_params=[15, 2], filter='gaussian' )
    res = cv2.bilateralFilter(res, 10, 10, 10)
    h, w = res.shape
    res2 = res.copy()  # np.zeros((h,w))
    res2 = cv2.normalize(res, res2, 0, 255, cv2.NORM_MINMAX)
    res2 = res2.reshape((h * w, 1))
    clusters = 2
    if debug:
        start: float = timer()
    kmeans = KMeans(n_clusters=clusters).fit(res2)
    if debug:
        print(kmeans.cluster_centers_)

    centers = kmeans.cluster_centers_.reshape(clusters)
    res2 = np.take(centers, kmeans.labels_).astype(np.uint8)
    res2 = res2.reshape((h, w))
    filler = Filler(res2, np.min(centers), size_threshold=120)
    res3 = filler.remove_noise().astype(np.uint8)
    filler = Filler(res3, np.min(centers), size_threshold=120)
    res3 = filler.remove_noise().astype(np.uint8)
    if debug:
        runtime: float = (timer() - start)
        print("Time ", runtime)
        img_show(np.hstack((img, res2)))
        img_show(np.hstack((img, res3)))
    return res2


