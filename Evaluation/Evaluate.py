import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def kmean_evaluate(gt_labeled: np.ndarray, result: np.ndarray):
    """
        Used to check the accuracy of prediction results from prior running MLModel_KMeans.determinate_clusters(img). It's getting done by
        calculating a confusion matrix and key figures (see returns!) to determine the result.
        -----------------------------------------------------------------------------------

        Parameters:
        -----------------------------------------------------------------------------------
        defuzzed_prediction result:     numpy.ndarray (tuple(float, float, float))
                                        Result from prior running FuzzyCMeans.defuzzify().
        gt_labeled:                     np.ndarray[][][]
                                        The labeled twin of the predicted image.

        Returns:
        -----------------------------------------------------------------------------------
        evaluation_result:      List(array(2, 2), tuple, float, )

                                confusion_matrix:               numpy.array(2, 2)

                                                                               Foreground        Background
                                                                             ----------------- -----------------
                                                                            |                 |                 |
                                                                 Foreground |  True Positives | False Positives |
                                                                            |                 |                 |
                                                                             ----------------- -----------------
                                                                            |                 |                 |
                                                                 Background |  False Negatives| True Negatives  |
                                                                            |                 |                 |
                                                                             ----------------- -----------------

                                accuracy:                       float
                                                                How often the classification was correct?
                                                                =(true_positives+true_negatives)/total
                                misclassification_rate:         float
                                                                How often the classification was wrong?
                                                                = (false_positives + false_negatives)/total
                                recall:                         float
                                                                Shows, how much classes have been predicted correctly out
                                                                of all possible classes. Should be as high as possible.
                                                                = true_positives/(true_positives+false_negatives)
                                precision:                      float
                                                                How many of all positive predicted classes, were actually
                                                                positive?
                                                                = true_positives/(true_positives + false_positives)
                                f_measure:                      float
                                                                Measures recall and precision at the same time using the
                                                                harmonic mean from both values.
                                                                = 2 * ((recall * precision)/(recall + precision))
        """
    # Calculate key figures
    evaluation_result = []
    confusion_matrix = _calculate_confusion_matrix(gt_labeled, result)
    evaluation_result.append(confusion_matrix)

    predicted_foreground = confusion_matrix[0][0] + confusion_matrix[1][0]
    predicted_background = confusion_matrix[0][1] + confusion_matrix[1][1]

    total = predicted_foreground + predicted_background

    # accuracy
    evaluation_result.append((confusion_matrix[0][0] + confusion_matrix[1][1]) / total)

    # Misclassification Rate
    evaluation_result.append((confusion_matrix[0][1] + confusion_matrix[1][0]) / total)

    # Recall
    recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    evaluation_result.append(recall)

    # Precision
    precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    evaluation_result.append(precision)

    # F-Measure
    evaluation_result.append(2 * ((recall * precision) / (recall + precision)))

    return evaluation_result


def fuzzy_cmeans(gt_labeled: np.ndarray, defuzzed_prediction_result: np.ndarray):
    """
    Used to check the accuracy of prediction results from prior running FuzzyCMeans.predict(). It's getting done by
    calculating a confusion matrix and key figures (see returns!) to determine the result.
    -----------------------------------------------------------------------------------

    Parameters:
    -----------------------------------------------------------------------------------
    defuzzed_prediction result:     numpy.ndarray (tuple(uint8))
                                    Result from prior running FuzzyCMeans.defuzzify().
    gt_labeled:                     np.ndarray[][][]
                                    The labeled twin of the predicted image.

    Returns:
    -----------------------------------------------------------------------------------
    evaluation_result:     List(array(2, 2), tuple, float, )

    cm:               numpy.array(2, 2)

                                                   Foreground        Background
                                                 ----------------- -----------------
                                                |                 |                 |
                                     Foreground |  True Positives | False Positives |
                                                |                 |                 |
                                                 ----------------- -----------------
                                                |                 |                 |
                                     Background |  False Negatives| True Negatives  |
                                                |                 |                 |
                                                 ----------------- -----------------

    accuracy:                       float
                                    How often the classification was correct?
                                    =(true_positives+true_negatives)/total
    misclassification_rate:         float
                                    How often the classification was wrong?
                                    = (false_positives + false_negatives)/total
    recall:                         float
                                    Shows, how much classes have been predicted correctly out
                                    of all possible classes. Should be as high as possible.
                                    = true_positives/(true_positives+false_negatives)
    precision:                      float
                                    How many of all positive predicted classes, were actually
                                    positive?
                                    = true_positives/(true_positives + false_positives)
    f_measure:                      float
                                    Measures recall and precision at the same time using the
                                    harmonic mean from both values.
                                    = 2 * ((recall * precision)/(recall + precision))
    """
    if len(gt_labeled.shape) != 3:
        raise Exception(' gt_labeled is expected to be a 3-dimensional array. But the actual dimension '
                        'is {}'.format(gt_labeled.shape) + '. Hint: Using matplotlib.pyplot.imread(gt_labeled_path) '
                                                           'provides the right format!!!')

    elif gt_labeled.shape[0] != 1944 or gt_labeled.shape[1] != 2592 or gt_labeled.shape[2] != 3:
        raise Exception('gt_labeled ist expected to be of size 1944 * 2592 px. Please provide a proper input format.')

    elif len(defuzzed_prediction_result.shape) != 1:
        raise Exception('defuzzed_prediction_result is expected to be a 1-dimensional array. But the actual dimension'
                        'is {}'.format(defuzzed_prediction_result.shape) + '. Hint: Running FuzzyCMeans.defuzzify will'
                                                                           'provide you a proper input here!!!')

    elif len(defuzzed_prediction_result) != 5038848:
        raise Exception('defuzzed_prediction_result is expected to be of length 5038848 (1944 x 2592 px). But the'
                        'actual length was {}'.format(len(defuzzed_prediction_result)))

    # Converting gt_labeled to a 1-dimensional array in order to make it comparable to defuzzed_prediction_result
    gt_labeled_1d = _convert_to_array1d(gt_labeled)

    # Calculate key figures
    evaluation_result = []

    # Confusion Matrix
    cm = _calculate_confusion_matrix(gt_labeled_1d, defuzzed_prediction_result)
    evaluation_result.append(cm)

    predicted_foreground = cm[0][0] + cm[1][0]
    predicted_background = cm[0][1] + cm[1][1]

    total = predicted_foreground + predicted_background

    # accuracy
    evaluation_result.append((cm[0][0] + cm[1][1]) / total)

    # Misclassification Rate
    evaluation_result.append((cm[0][1] + cm[1][0]) / total)

    # Recall
    recall = cm[0][0] / (cm[0][0] + cm[1][0])
    evaluation_result.append(recall)

    # Precision
    precision = cm[0][0] / (cm[0][0] + cm[0][1])
    evaluation_result.append(precision)

    # F-Measure
    evaluation_result.append(2 * ((recall * precision) / (recall + precision)))

    return evaluation_result


def _convert_to_array1d(gt_labeled: object):

    # Converting into 2-dimensional array
    gt_labeled_2d = gt_labeled.reshape(gt_labeled.shape[0] * gt_labeled.shape[1], gt_labeled.shape[2])

    gt_labeled_1d = np.zeros(len(gt_labeled_2d), dtype=tuple, order='C')

    for i in range(len(gt_labeled_2d)):

        if gt_labeled_2d[i][0] == 255:
            gt_labeled_1d[i] = 1
        continue

    return gt_labeled_1d.astype(np.uint8)


def _calculate_confusion_matrix(gt_labeled_1d: np.ndarray, defuzzed_prediction_result: np.ndarray):

    cm = np.zeros((2, 2))

    for i in range(len(defuzzed_prediction_result)):

        if defuzzed_prediction_result[i] == 1 and gt_labeled_1d[i] == 1:
            cm[0][0] += 1

        elif defuzzed_prediction_result[i] == 1 and gt_labeled_1d[i] == 0:
            cm[0][1] += 1

        elif defuzzed_prediction_result[i] == 0 and gt_labeled_1d[i] == 0:
            cm[1][1] += 1

        elif defuzzed_prediction_result[i] == 0 and gt_labeled_1d[i] == 1:
            cm[1][0] += 1

    return cm

