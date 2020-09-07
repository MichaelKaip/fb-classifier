# Documentation:

This is a Software Library which provides the functionality of classifying (segmenting) images into foreground and
background.

It includes 2 different approaches:

## Fuzzy C-Means and K-Means Algorithm

### The Fuzzy C-Means classifier contains the following libraries, methods and oyher helpful resources:

* **ComputerVisionPackageSetup.yml:** For easily setting up the environment and installing all necessary packages, you
can simply load this file into a new environment on anaconda.

#### Classification

* **FuzzyCMeans.py:**
  * .train_model: Used for training the model.
  * .build_model: Builds the model from training data calculating the mean
  * .build_average_model: Allows to combine different trained models.
  * .predict: Predicts on unseen data using a trained model.
  * .defuzzify: Defuzzifies prediction results
  <br/>
* **KMeans.py:**
  * ...
  <br/>
* **Helper.py:**
  * .cv_to_pil
  * .pil_to_cv
  * .convert_to_array_2d: Converts numpy ndarrays from 3 into 2 dimensions.
  * .convert_to_array_3d: Converts numpy ndarrays from 1 into 3 dimensions.
  * .capture_video_frames: Captures single video frames.
  * .load_frames_as_array
  * .sorted_alphanumeric: Sorts a directory in alphanumeric order
  <br/>

#### Evaluation

* **Evaluate.py**
  * .kmeans_evaluate: Evaluates KMeans algorithm on ground truth data
  * .fuzzy_cmeans: Evaluates a with FuzzyCMeans model on ground truth data

#### Filters
* **HomomorphicFilter.py**
  * .filter: A methods which applies a homomorphic filter (gaussian or butterworth) on an image

#### Code Examples
The aim of this section is to provide commented Code Examples for the user to gain comprehension on 
how to use this Toolbox in the right way.
* **FCMeansExamples.py**
  1) TRAINING AND EVALUATION ON GROUND TRUTH DATA
  2) PREDICTING AND EVALUATING UNSEEN DATA
  <br/>
* **KMeansExamples.py:**
   1) CLUSSIFIED AND EVALUATION ON GROUND TRUTH DATA 




