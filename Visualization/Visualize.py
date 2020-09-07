import string
import numpy as np
import cv2
import re
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from Classification import KMeans as kmeans_cl

def kmeans_prediction_result(filename):

    cap = cv2.VideoCapture(filename)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = kmeans_cl.determinate_clusters(frame, width=640, height=480)
        width = 640
        height = 480
        dim = (width, height)
        gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('frame',  np.hstack((gray, result)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    kmeans_prediction_result("../../../data/20190626_13_14.mp4")

def cmeans_prediction_result(path_to_original: string, path_to_predicted: string):
    """
        Used to visualize the original video sequence and the prediction side-by-side in one window.

        Parameters:
        -----------------------------------------------------------------------------------
        path_to_original:               string
                                        The path to the original video sequence, on which prediction has been executed
                                        on.

        path_to_predicted:              string
                                        The path to the predicted video sequence, from prior running cmeans_predict.

        """
    pass
