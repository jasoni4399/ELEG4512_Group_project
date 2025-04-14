import cv2
import os
import numpy as np
from processors import *

path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@testing
@process_image(image_path="final_result")#file name to save the image
def main(image):
    # Enhance the edge of object by observing the contrast of it and the environment
    enhanced1 = gamma_clahe(image, 12, 32)
    enhanced2 = gamma_clahe(image, 8, 8)
    save_image(enhanced1, "enhanced1")
    save_image(enhanced2, "enhanced2")
    
    # Highlight area that reflect light + remove noise(e.g. glass reflection)
    output1 = generate_reflection_mask(enhanced1, kernel_size=5, threshold=120)
    param_ouput1_contour_area = 250
    filtered_output1 = filter_draw_contours(output1, min_contour_area=param_ouput1_contour_area*10, contour_thickness= 3)

    # Get object edge with high constract with the environemt
    param_canny_low = 150 
    param_canny_high = 350
    output2 = edge_based_segmentation(enhanced2, canny_low=param_canny_low, canny_high=param_canny_high, dilate_iterations=3)
    
    param_ouput2_contour_area = 50
    filtered_output2 = filter_draw_contours(output2, min_contour_area=param_ouput2_contour_area*10, contour_thickness= 3)

    # Combination of these two kinds of edges
    # 2. more about the edge between reflective surfaces and shadows
    # 1. more about the edge between object and environment
    combined1 = cv2.addWeighted(filtered_output2, 1, filtered_output1, 1, 0)

    # Change color of the edges
    output_img = image.copy()
    output_img[combined1 == 255] = [0, 255, 0] 

    return output_img

@process_image(image_path="k_mean_result")#file name to save the image
def k_mean_test(image):
    # Convert to float32 for k-means
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.2)
    K = 10
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back to uint8 and reshape to original image size
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    result_image = res.reshape((image.shape))


    return result_image



if __name__ == "__main__":
    main(image)