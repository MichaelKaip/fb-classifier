import string
from timeit import default_timer as timer
import numpy as np
import skfuzzy as fuzz
from Classification import Helper
import pickle


def train_model(data: np.ndarray, c: int = 2, exp: float = 2, error: float = 0.005, maxiter: int = 1000,):
    """
    Classifies images using fuzzy c-means algorithm. It uses the scikit-fuzzy library on
    https://pythonhosted.org/scikit-fuzzy/_modules/skfuzzy/cluster/_cmeans.html
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               2D or 3D numpy.ndarray
                        Image data to be clustered.
    c:                  int
                        Desired number of clusters or classes. Default = 2, because previous research on different
                        images from sample data showed, that fpc is maximized using 2 clusters for each tested frame.
    exp:                float
                        Array exponentiation applied to the membership function u_old at each
                        iteration, where U_new = u_old ** m. Default = 2.
    error:              float
                        Stopping criterion; stops if the norm of (u[p] - u[p-1]) < error. Default = 0.005.
    maxiter:            int
                        Maximum number of iterations. Default = 1000.

    Returns:
    -----------------------------------------------------------------------------------
    cntrs:          2d array, size (S, c)
                    Cluster centers. Data for each center along each feature provided for
                    every cluster (of the c requested clusters).
    fcpm:           2d array, (S, N)
                    Final fuzzy c-partitioned matrix. Measures the probability of membership for each pixel and
                    different clusters.
    fcpm0:          2d array, (S, N)
                    Initial guess at fuzzy c-partitioned matrix.
    dist:           2d array, (S, N)
                    Final Euclidian distance matrix.
    obj_func_hist:  1d array, length P
                    Objective function history. The goal of the algorithm is to minimize this function which means,
                    that the similarity among all members of one cluster is maximal.
    iter:           int
                    Number of iterations run.
    fpc:            float
                    Final fuzzy partition coefficient. Measures the cluster validity in the
                    range between 0 and 1 (normalized). The higher this value - the better.
    runtime:        float
                    Runtime of the algorithm in seconds.
    """

    if len(data.shape) < 2 | len(data.shape) > 3:
        raise Exception('Import array should not be less than 2 dimension, nor exceed 3 dimensions. '
                        'The value of d was: {}'.format(len(data.shape)))

    elif len(data.shape) == 3:
        img2d = Helper.convert_to_array2d(data)
        start: float = timer()
        cmeans = fuzz.cluster.cmeans(img2d, c, exp, error, maxiter)
        runtime: float = (timer() - start)
        return cmeans + (runtime,)

    else:
        start: float = timer()
        cmeans: object = fuzz.cluster.cmeans(data, c, exp, error, maxiter)
        runtime: float = (timer() - start)
        return cmeans + (runtime,)


def build_model(data: list):
    """
    Calculates the average for each element from prior training c-means.
    First you have to run fcm_train on multiple sample frames. Afterwards this methods is used to build a ML-Model,
    based on the mean values. Most of them just returned for statistical reasons, whereas mean_cntrs and mean_fcpm
    are needed for predicting unseen images.
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List (ndarray, int)
                        Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    fcm_model:      List (ndarray, float)
                    Returns a Dictionary the same structure as the List returned from fcm_train, but with mean
                    values from all samples in the training set.

                    mean_cntrs:     2d array, size (S, c)
                                    Average Cluster center. Data for each center along each feature provided for
                                    every cluster (of the c requested clusters).
                    mean_fcpm:      2d array, (S, N)
                                    Average final fuzzy c-partitioned matrix.
                    mean_fcpm0:     2d array, (S, N)
                                    Average initial guess at fuzzy c-partitioned matrix.
                    mean_dist:      2d array, (S, N)
                                    Average final Euclidian distance matrix.
                    mean_iter:      float
                                    Average number of iterations run.
                    mean_fpc:       float
                                    Average final fuzzy partition coefficient
                    mean_runtime:   float
                                    Average runtime of the algorithm in seconds.
    """
    if len(data) == 1:

        return data

    fcm_model = [_get_mean_cntrs(data), _get_mean_fcpm(data), _get_mean_fcpm0(data),
                 _get_mean_dist(data), _get_mean_iter(data), _get_mean_fpc(data),
                 _get_mean_runtime(data)]

    return fcm_model


def build_average_model(batch_1: list, batch_2: list):
    """
        Used to build one model out of two batches. Takes two batches and returns the average of both.
        -----------------------------------------------------------------------------------

        Parameters:
        -----------------------------------------------------------------------------------
        batch_1:            List (ndarray, int)
                            Result from fcm_train().
        batch_2:            List (ndarray, int)
                            Result from fcm_train().

        Returns:
        -----------------------------------------------------------------------------------
        fcm_model:      List (ndarray, float)
                        Returns a Dictionary the same structure as the List returned from fcm_train, but with mean
                        values from all samples in the training set.

                        mean_cntrs:     2d array, size (S, c)
                                        Average Cluster center. Data for each center along each feature provided for
                                        every cluster (of the c requested clusters).
                        mean_fcpm:      2d array, (S, N)
                                        Average final fuzzy c-partitioned matrix.
                        mean_fcpm0:     2d array, (S, N)
                                        Average initial guess at fuzzy c-partitioned matrix.
                        mean_dist:      2d array, (S, N)
                                        Average final Euclidian distance matrix.
                        mean_iter:      float
                                        Average number of iterations run.
                        mean_fpc:       float
                                        Average final fuzzy partition coefficient
                        mean_runtime:   float
                                        Average runtime of the algorithm in seconds.
        """
    fcm_model = []

    for i in range(len(batch_1)):
        fcm_model.append((batch_1[i] / 2 + batch_2[i] / 2))

    return fcm_model


def predict(data: np.ndarray, cntrs_trained: np.ndarray, fcpm_trained: np.ndarray = None,
            exp: float = 2, error: float = 0.005, maxiter: int = 1000):
    """
    Predicts new (unseen) data, given a trained fuzzy c-means framework. It works by repeating the clustering
    with fixed centers (mean_cntrs from fcm_train) and then efficiently finds the fuzzy membership at all
    points.

    It uses the scikit-fuzzy library on https://pythonhosted.org/scikit-fuzzy/.
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:           2D or 3D numpy.ndarray
                    New independent data set to be predicted based on build_model().
    cntrs_trained:  2D numpy.ndarray
                    Location of trained centers from prior training cmeans (fcm_train) and build_model().
    fcpm_trained:   2D numpy.ndarray
                    Fuzzy c-partitioned matrix from prior training cmeans (fcm_train) and build_model().
                    Default = None.
    exp:            float
                    Array exponentiation applied to the membership function u_old at each
                    iteration, where U_new = u_old ** m. Default = 2.
    error:          float
                    Stopping criterion; stops if the norm of (u[p] - u[p-1]) < error. Default = 0.005.
    maxiter:        int
                    Maximum number of iterations. Default = 1000.

    Returns:
    -----------------------------------------------------------------------------------
    fcpm:           2d array, (S, N)
                    Final fuzzy c-partitioned matrix. Measures the probability of membership for each pixel and
                    different clusters.
    fcpm0:          2d array, (S, N)
                    Initial guess at fuzzy c-partitioned matrix.
    dist:           2d array, (S, N)
                    Final Euclidian distance matrix.
    obj_func_hist:  1d array, length P
                    Objective function history. The goal of the algorithm is to minimize this function which means,
                    that the similarity among all members of one cluster is maximal.
    iter:           int
                    Number of iterations run.
    fpc:            float
                    Final fuzzy partition coefficient. Measures the cluster validity in the
                    range between 0 and 1 (normalized). The higher this value - the better.
    runtime:        float
                    Runtime of the algorithm in seconds.
    """
    if len(data.shape) < 2 | len(data.shape) > 3:
        raise Exception('Import array should not be less than 2 dimension, nor exceed 3 dimensions. '
                        'The value of d was: {}'.format(len(data.shape)))

    elif len(data.shape) == 3:
        img2d = Helper.convert_to_array2d(data)
        start: float = timer()
        cmeans_predict: object = fuzz.cluster.cmeans_predict(img2d, cntrs_trained, exp, error,
                                                             maxiter, init=fcpm_trained)
        runtime: float = (timer() - start)
        return cmeans_predict + (runtime,)

    else:
        start: float = timer()
        cmeans_predict: object = fuzz.cluster.cmeans_predict(data, cntrs_trained, exp, error,
                                                             maxiter, init=fcpm_trained)
        runtime: float = (timer() - start)
        return cmeans_predict + (runtime,)


def defuzzify(data: np.ndarray):
    """
    Produces a defuzzified data set from prior predicting. The input data contains for each pixel the probability
    of cluster-membership.

    If "the probability of membership to cluster 0" > "the probability of membership to cluster 1":
        ==> it's a background pixel (per definition)

    If "the probability of membership to cluster 0" < "the probability of membership to cluster 1":
        ==> it's a foreground pixel (per definition)

    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:                   2d numpy.ndarray
                            Final fuzzy c-partitioned matrix from prior running
                            FuzzyCMeans.predict() => prediction_result[0]

    Returns:
    -----------------------------------------------------------------------------------
    defuzzed_prediction:    numpy.ndarray (tuple(r,g,b))
                            The defuzzified prediction result, where each pixel has been assigned to be either
                            foreground or background.

    """

    defuzzed_prediction_result = np.zeros(len(data[0]), dtype=tuple, order='C')

    for i in range(len(data[0])):

        if data[0][i] > data[1][i]:
            defuzzed_prediction_result[i] = 1  # white (foreground)

        elif data[0][i] == data[1][i]:
            defuzzed_prediction_result[i] = 0

        else:
            defuzzed_prediction_result[i] = 0  # black (background)

    return defuzzed_prediction_result.astype(np.uint8)


def _get_mean_cntrs(data: list):
    """
    Calculates the average center positions for each element from prior training c-means.

    -----------------------------------------------------------------------------------
    !!! Needed for prediction !!!
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List (ndarray, int)
                        Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    mean_cntrs:     2d array, size (S, c)
                    Average center positions
    """

    mean_cntrs = data[0][0] / len(data)  # Starting with the center matrix of the 1st image

    for i in range(1, len(data)):
        mean_cntrs += data[i][0] / len(data)  # Adding the center matrices from all other images

    return mean_cntrs


def _get_mean_fcpm(data: list):
    """
    Calculates the average fuzzy c-partitioned-matrix for each element from prior
    training c-means.

    -----------------------------------------------------------------------------------
    !!! For statistical evaluation !!!
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List (ndarray, int)
                        Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    mean_fcpm:      2d array, (S, N)
                    Average final fuzzy c-partitioned matrix.
    """

    mean_fcpm = data[0][1] / len(data)  # Starting with the fcpm matrix of the 1st image

    for i in range(1, len(data)):
        mean_fcpm += data[i][1] / len(data)  # Adding the fcpm matrices from all other images

    return mean_fcpm  # Returning the mean fcpm matrix


def _get_mean_fcpm0(data: list):
    """
    Calculates the average initial guess at fuzzy c-partitioned matrix.

    -----------------------------------------------------------------------------------
    !!! For statistical evaluation !!!
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List (ndarray, int)
                        Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    mean_fcpm0: 2d array, (S, N)
                Average initial guess at fuzzy c-partitioned matrix.
    """

    mean_fcpm0 = data[0][2] / len(data)  # Starting with the fcpm0 matrix of the 1st image

    for i in range(1, len(data)):
        mean_fcpm0 += data[i][2] / len(data)  # Adding the fcpm0 matrices from all other images

    return mean_fcpm0  # Returning the mean fcpm0 matrix


def _get_mean_dist(data: list):
    """
    Calculates the average euclidian distance matrix from prior training c-means.

    -----------------------------------------------------------------------------------
    !!! For statistical evaluation !!!
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List (ndarray, int)
                        Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    mean_dist:      2d array, (S, N)
                    Average final Euclidian distance matrix.
    """

    mean_dist = data[0][3] / len(data)  # Starting with the distance matrix of the 1st image

    for i in range(1, len(data)):
        mean_dist += data[i][3] / len(data)  # Adding the distance matrices from all other images

    return mean_dist  # Returning the mean distance matrix


def _get_mean_iter(data: list):
    """
    Calculates the average iterations run for clustering from prior training c-means.

    -----------------------------------------------------------------------------------
    !!! For statistical evaluation !!!
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:   List (ndarray, int)
            Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    mean_iter:  float
                Average number of iterations run.
    """

    mean_iter = 0

    for item in data:
        mean_iter += item[5] / len(data)

    return mean_iter


def _get_mean_fpc(data: list):
    """
    Calculates the average fuzzy partition coefficient from prior training c-means.

    -----------------------------------------------------------------------------------
    !!! For statistical evaluation !!!
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List (ndarray, int)
                        Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    mean_fpc:   float
                The average final fuzzy partition coefficient
    """

    mean_fpc = 0

    for item in data:
        mean_fpc += item[6] / len(data)

    return mean_fpc


def _get_mean_runtime(data: list):
    """
    Calculates the average runtime/image for clustering from prior training c-means.

    -----------------------------------------------------------------------------------
    !!! For statistical evaluation !!!
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List (ndarray, int)
                        Result from fcm_train().

    Returns:
    -----------------------------------------------------------------------------------
    runtime:        float
                    The average runtime of the algorithm in seconds.
    """

    mean_runtime = 0

    for item in data:
        mean_runtime += item[7] / len(data)

    return mean_runtime


def save_model(data, filename: string):
    """
    Used to serialize and save data. A trained model, for instance or orediction results
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    data:               List or numpy.ndarray

    """
    file = open('../../'+filename, 'wb')
    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def load_model(file_path: string):
    """
    Used to serialize an save a trained model, so it can be reused later on again.
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    file_path:               List (ndarray, int)
                             Path to a stored model from prior running save_model().

    Returns:
    -----------------------------------------------------------------------------------
    fcm_model:      List (ndarray, float)
                    The de-serialized model.
    """
    fcm_model = pickle.load(open(file_path, 'rb'))

    return fcm_model
