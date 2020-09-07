from Classification import FuzzyCMeans
from timeit import default_timer as timer
from Classification import Helper
import os
import cv2
from Evaluation import Evaluate

def main():

    ###########################################################################################################
    # 1) TRAINING AND EVALUATION ON GROUND TRUTH DATA #########################################################
    ###########################################################################################################

    #####################################################
    # 1.1) TRAINING THE MODEL IN BATCHES OF APPROX. 100 #
    #####################################################

    for folder in Helper.sorted_aphanumeric(os.listdir('../../training_data')):

        training_data = []
        cmeans_data = []

        for img in os.listdir('../../training_data/' + folder):
            path = '../../training_data/' + folder + '/' + img
            training_data.append(cv2.imread(path))

        print('Start running Cmeans on ' + folder)
        # Running c-means for frames per batch.
        for i in range(len(training_data)):
            cmeans_data.append(FuzzyCMeans.train_model(training_data[i]))
            print(i)

        print('Start building model')
        fcm_model = FuzzyCMeans.build_model(data=cmeans_data)
        FuzzyCMeans.save_model(fcm_model, filename='fcm_model_' + folder)

    ##################################################
    # 1.2) BUILDING MODEL BATCHES OF DIFFERENT SIZES #
    ##################################################

    batch_1 = FuzzyCMeans.load_model('../../trained_models/fcm_model_batch_1')
    batch_2 = FuzzyCMeans.load_model('../../trained_models/fcm_model_batch_2')
    batch_3 = FuzzyCMeans.load_model('../../trained_models/fcm_model_batch_3')
    batch_4 = FuzzyCMeans.load_model('../../trained_models/fcm_model_batch_4')
    batch_5 = FuzzyCMeans.load_model('../../trained_models/fcm_model_batch_5')
    batch_6 = FuzzyCMeans.load_model('../../trained_models/fcm_model_batch_4')

    # Creating new batches with different batch sizes and storing them for later usage
    batch_200_12 = FuzzyCMeans.build_average_model(batch_1, batch_2)
    FuzzyCMeans.save_model(batch_200_12, 'batch_12_size200')

    batch_300_123 = FuzzyCMeans.build_average_model(batch_200_12, batch_3)
    FuzzyCMeans.save_model(batch_300_123, 'batch_123_size300')

    batch_400_1234 = FuzzyCMeans.build_average_model(batch_300_123, batch_4)
    FuzzyCMeans.save_model(batch_400_1234, 'batch_1234_size400')

    batch_500_12345 = FuzzyCMeans.build_average_model(batch_400_1234, batch_5)
    FuzzyCMeans.save_model(batch_500_12345, 'batch_12345_size500')

    batch_600_123456 = FuzzyCMeans.build_average_model(batch_500_12345, batch_6)
    FuzzyCMeans.save_model(batch_600_123456, 'batch_123456_size600')

    ########################################################################
    # 1.3) PREDICTION ON GROUND TRUTH WITH MODELS OF DIFFERENT BATCH SIZES #
    ########################################################################

    gt_samples = []

    # Load sample images
    for i in Helper.sorted_aphanumeric(os.listdir('../../other_resources/GroundTruth/GT_Images/originals')):
        gt_samples.append(cv2.imread('../../other_resources/GroundTruth/GT_Images/originals/' + i))

    # For each model
    for model in Helper.sorted_aphanumeric(os.listdir('../../../trained_models/combined_batches')):
        loaded_model = FuzzyCMeans.load_model('../../../trained_models/combined_batches/' + model)

        prediction_results = []

        print('Start running prediction on {}'.format(model))

        # ... run prediction on each ground truth sample ...
        for j in range(len(gt_samples)):
            prediction_results.append(FuzzyCMeans.predict(gt_samples[j], loaded_model[0]))
            print(j)

        print('Start storing prediction results on {}'.format(model))

        # and store the results for later usage.
        FuzzyCMeans.save_model(prediction_results, 'prediction_results_{}'.format(model))

    ###########################################################################
    # 1.4) DEFUZZIFICATION AND EVALUATION OF PREDICTIONS ON GROUND TRUTH DATA #
    ###########################################################################

    print('Start loading labeled images')

    labeled_img = []

    # Load labeled ground truth images
    for img in Helper.sorted_aphanumeric(os.listdir('../../other_resources/GroundTruth/GT_Images/labeled')):
        labeled_img.append(cv2.imread('../../other_resources/GroundTruth/GT_Images/labeled/' + img))

    # Load predictions
    for result in Helper.sorted_aphanumeric(os.listdir('../../../prediction_results')):
        prediction_result = FuzzyCMeans.load_model('../../../prediction_results/' + result)

        print('Start defuzzifying ' + result)

        defuzzed_prediction_result = []

        # Defuzzify predictions
        for i in range(len(prediction_result)):
            defuzzed_prediction_result.append(FuzzyCMeans.defuzzify(prediction_result[i][0]))
            print(i)

        print('Start evaluating predictions on ' + result)

        evaluated_predictions = []

        # Evaluate predictions
        for i in range(len(defuzzed_prediction_result)):
            evaluated_predictions.append(Evaluate.fuzzy_cmeans(labeled_img[i], defuzzed_prediction_result[i]))
            print(i)

        # Save evaluated predictions
        FuzzyCMeans.save_model(evaluated_predictions, 'evaluated_' + result)

    #######################################################################
    # 1.5) INTERPRETATION OF CMEANS-PREDICTION AND COMPARISON WITH KMEANS #
    #######################################################################

    import xlsxwriter

    # Load data from file
    cmeans_data = []
    for evaluation in Helper.sorted_aphanumeric(os.listdir('../../evaluated_predictions')):
        cmeans_data.append(FuzzyCMeans.load_model('../../evaluated_predictions/' + evaluation))

    kmeans_data = FuzzyCMeans.load_model('../../kmeans_evaluation')

    # Create Workbook
    workbook = xlsxwriter.Workbook('_comparison_cmeans_vs_kmeans.xlsx')

    # Load cmeans and kmeans data
    worksheet_cmeans = workbook.add_worksheet('cmeans')
    worksheet_kmeans = workbook.add_worksheet('kmeans')

    # Write column names into the worksheets
    column = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    column_name = ['Model Batch Size', 'Image Name', 'True Positives', 'True Negatives', 'False Positives',
                   'False Negatives','Accuracy', 'Misclassification Rate', 'Recall', 'Precision', 'F1-Score']

    for i in range(len(column)):
        worksheet_cmeans.write(column[i]+'1', column_name[i])
        worksheet_kmeans.write(column[i]+'1', column_name[i])

    model_batch_size = [200, 300, 400, 500, 600]
    count = 1

    # Fill in cmeans data
    for i in range(len(cmeans_data)):  # 5

        for j in range(len(cmeans_data[0])):  # 31

            worksheet_cmeans.write(count, 0, model_batch_size[i])
            worksheet_cmeans.write(count, 1, 'picture_{}'.format(j))
            worksheet_cmeans.write(count, 2, cmeans_data[i][j][0][0][0])
            worksheet_cmeans.write(count, 3, cmeans_data[i][j][0][1][1])
            worksheet_cmeans.write(count, 4, cmeans_data[i][j][0][0][1])
            worksheet_cmeans.write(count, 5, cmeans_data[i][j][0][1][0])
            worksheet_cmeans.write(count, 6, cmeans_data[i][j][1])
            worksheet_cmeans.write(count, 7, cmeans_data[i][j][2])
            worksheet_cmeans.write(count, 8, cmeans_data[i][j][3])
            worksheet_cmeans.write(count, 9, cmeans_data[i][j][4])
            worksheet_cmeans.write(count, 10, cmeans_data[i][j][5])
            count += 1

    count = 1
    # Fill in kmeans data
    for i in range(len(kmeans_data)):  # 5

            worksheet_kmeans.write(count, 1, 'picture_{}'.format(i))
            worksheet_kmeans.write(count, 2, kmeans_data[i][0][0][0])
            worksheet_kmeans.write(count, 3, kmeans_data[i][0][1][1])
            worksheet_kmeans.write(count, 4, kmeans_data[i][0][0][1])
            worksheet_kmeans.write(count, 5, kmeans_data[i][0][1][0])
            worksheet_kmeans.write(count, 6, kmeans_data[i][1])
            worksheet_kmeans.write(count, 7, kmeans_data[i][2])
            worksheet_kmeans.write(count, 8, kmeans_data[i][3])
            worksheet_kmeans.write(count, 9, kmeans_data[i][4])
            worksheet_kmeans.write(count, 10, kmeans_data[i][5])
            count += 1

    workbook.close()

    ##############################################################################################################
    # 2.) PREDICTING AND EVALUATING UNSEEN DATA ##################################################################
    ##############################################################################################################

    # source: str = '../../../Testvid/Testvid.mp4'
    # destination: str = '../../../Testvid/single_frames'

    ####################################################################
    # 2.1) Preparing a test data set and run FuzzyCMeans.predict on it #
    ####################################################################

    start_1: float = timer()

    print('Start Capturing...')

    # Capture single frames from video
    Helper.capture_video_frames(source_path=source, dest_path=destination, duration=0.08)
    test_data = Helper.load_frames_as_array(destination)

    print('Start loading trained Model...')

    # Load cmeans model
    trained_model = FuzzyCMeans.load_model('../../../trained_models/combined_batches/batch_12_size200')

    print('Start running prediction on data set...')

    # Run prediction on unseen data and build a data structure containing the relevant prediction results
    prediction_result = []

    for i in range(len(test_data)):

        prediction_result.append(FuzzyCMeans.predict(test_data[i], trained_model[0]))

        print(i)

    print('Start storing the prediction results...')

    # Freeing memory
    trained_model = None
    test_data = None

    # Store prediction results
    FuzzyCMeans.save_model(prediction_result, 'prediction_results_test_final')

    runtime_1: float = timer() - start_1

    print('Job #2.1 finished in {} '.format(runtime_1/60) + 'minutes')

    ####################################
    # 2.2) Defuzzify prediction results #
    ####################################

    # Load prediction results from file
    pred_data = FuzzyCMeans.load_model('../../prediction_results_test_final')

    print('Start defuzzifying...')

    start_2: float = timer()

    # Defuzzify
    defuzzed_prediction_data = []
    for i in range(len(pred_data)):
        defuzzed_prediction_data.append(FuzzyCMeans.defuzzify(pred_data[i][0]))
        print(i)

    # Store defuzzyfied data
    FuzzyCMeans.save_model(defuzzed_prediction_data, 'defuzzed_prediction_results_test_final')

    runtime_2: float = timer() - start_2

    print('Job #2.2 finished in {} '.format(runtime_2/60) + 'minutes')

    print('debug')

    ##############################################################################
    # 2.3) Write video sequences from original and defuzzified video image frames #
    ##############################################################################

    # Load defuzzed prediction data
    pred_data = FuzzyCMeans.load_model('../../defuzzed_prediction_results_test_final')

    defuzzed_prediction = []

    for i in range(len(pred_data)):
        defuzzed_prediction.append(FuzzyCMeans.defuzzify(pred_data[i][0]))
        print(i)

    start_3: float = timer()

    print('Start reshaping data...')

    pred_data_3d = Helper.convert_to_array_3d(pred_data)

    print('Start writing video')

    # Video settings
    width, height = 2592, 1944
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # video codec for mp4-format
    fps = 24
    video_name = 'converted_video.mp4'

    # Write video
    out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
    for i in range(len(pred_data_3d)):
        cv2.imwrite('test_debug.png', pred_data_3d[i])
        out.write(cv2.cvtColor(pred_data_3d[i]))
    out.release()

    # Show original and predicted video side-by-side
    

    runtime_3: float = timer() - start_3

    print('Job #2.3 finished in {} '.format(runtime_3/60) + 'minutes')


if __name__ == '__main__':
    main()
