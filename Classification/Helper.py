
import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngImageFile as pil_file
import matplotlib.pyplot as plt
import re
import os
import string


def cv_to_pil(img: np.ndarray):
    """
    Converts an image from opencv to PIL object

    :param img: an origin image giving with opencv library
    :return: an image in PIL format or None if converting
             is not possible
    """
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pil_to_cv(img: pil_file):
    """
    Converts an image from PIL to opencv object

    :param img: an origin image giving with PIL library
    :return: an image in opencv format

    """

    return np.array(img.convert('RGB'))


def convert_to_array2d(img: np.ndarray):
    """
    Converts numpy ndarrays from 3 into 2 dimensions.
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------

    img:    3D numpy.ndarray
            Image data to be clustered. 2-dimensional arrays will be returned without
            any change and passing in arrays with less than 2 or more then 3 dimension
            will cause an exception.

    Returns:
    -----------------------------------------------------------------------------------
    img2d:  2D numpy.ndarray, size (M,N)
            Reshaped numpy array. M is the number of features and N the number of data
            points in the data set.
    """
    if len(img.shape) == 2:
        return img

    elif len(img.shape) < 2 | len(img.shape) > 3:
        raise Exception('Import array should not be less than 2 dimension, nor exceed '
                        '3 dimensions. The value of d was: {}'.format(len(img.shape)))

    else:
        img2d = img.reshape(img.shape[2], img.shape[0] * img.shape[1])

    return img2d


def capture_video_frames(source_path: string, dest_path: string, duration: float, fps: int = 24):
    """
    Used for preparing a data set. Loads a video, which is expected to be in the .mp4 format
    and stores the single frames of this video.

    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    source_path:            string
                            Path to the video which has to be captured, e.g.
                            .../.../myTestvid.mp4
    dest_path:              string
                            Destination of where the captured frames has to be stored to.
    duration:               float
                            Defines how much minutes of the video to be captured.
                            The number of frames to be captured is then calculated by
                            minutes * 60 * fps.
    fps:                    int
                            Frames per second. Default = 24.
    """

    vid_name = source_path.split('/')[3].split('.')[0]

    frame_counter = duration * 60 * fps

    vid_capture = cv2.VideoCapture(source_path)
    success, image = vid_capture.read()
    count = 0
    while success and count < frame_counter:
        cv2.imwrite(dest_path + '/' + vid_name + '_frame_%d.png' % count, image)  # save frame as PNG file
        success, image = vid_capture.read()
        print(count, ' Read a new frame: ', success)
        count += 1


def load_frames_as_array(dir_path):
    """
        Once a video or sequence of has been captured, this

        -----------------------------------------------------------------------------------

        Parameters:
        -----------------------------------------------------------------------------------
        source_path:            string
                                Path to the video which has to be captured, e.g.
                                .../.../myTestvid.mp4
        dest_path:              string
                                Destination of where the captured frames has to be stored to.
        minutes:                float
                                Defines how much minutes of the video to be captured.
                                The number of frames to be captured is calculated by
                                minutes * 60 * fps.
        fps:                    int
                                Frames per second. Default = 24.
        """
    data = []
    for img in sorted_aphanumeric(os.listdir(dir_path)):
        tmp = cv2.imread(dir_path + '/' + img, cv2.IMREAD_COLOR)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        tmp_1 = tmp.astype(np.float64)
        tmp_1 = tmp_1 / 255.0
        data.append(tmp_1)

    return data


def sorted_aphanumeric(data):
    """
    Used to sort a directory in alphanumeric order.
    From: https://stackoverflow.com/questions/4813061/non-alphanumeric-list-order-from-os-listdir
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


def convert_to_array_3d(data_1d: list):
    """
        Converts numpy ndarrays from 1 into 3 dimensions. It can be especially used to convert
        defuzzyfied prediction data, which is a 1d-array, into a 3d-array. It can be than used
        for visualize the results of prediction.
        -----------------------------------------------------------------------------------

        Parameters:
        -----------------------------------------------------------------------------------

        img:    3D numpy.ndarray
                Image data to be clustered. 2-dimensional arrays will be returned without
                any change and passing in arrays with less than 2 or more then 3 dimension
                will cause an exception.

        Returns:
        -----------------------------------------------------------------------------------
        img2d:  2D numpy.ndarray, size (M,N)
                Reshaped numpy array. M is the number of features and N the number of data
                points in the data set.
        """

    data_3d = []
    for i in range(len(data_1d)):
        tmp = np.reshape(data_1d[i], (2592, 1944))
        print(i)
        data_3d.append(tmp)

    return data_3d


def convert_to_gray_and_reshape(img, width, height):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

