from Classification import KMeans, Helper
import cv2
from Evaluation import Evaluate

def main():
    """
    Provides an examples to use kmeans
    :return:
    """
    # Path to an original image
    original_img_path: str = '../../other_resources/GroundTruth/GT_Images/originals/bild30.png'
    # Path to ground_truth image
    ground_truth_img_path: str = '../../other_resources/GroundTruth/GT_Images/labeled/bild30GT.png'
    # Loads an original image
    original_img = cv2.imread(original_img_path)

    # Loads an ground_truth image
    ground_truth_img = cv2.imread(ground_truth_img_path)

    # Prints shape each
    print(original_img.shape, ground_truth_img.shape)
    # Starts clustering
    result_img = KMeans.determinate_clusters(original_img)
    print("Type of result image")
    print(type(result_img))
    # Prepares ground_thuth image to have a same dimension as clustered image
    # and color
    ground_truth_img = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2GRAY)

    # Information

    print("Type ground_truth_img")
    print(type(ground_truth_img))
    print("Shape of result image")
    print(result_img.shape)
    print("Shape of ground_truth_image")
    print(ground_truth_img.shape)

    # If you want to show an clusterfied image, please uncomments
    # Helper.img_show(ground_truth_img)

    cm = Evaluate.kmean_evaluate(ground_truth_img, result_img)

if __name__ == '__main__':
    main()
