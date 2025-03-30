import cv2
import os
import numpy as np
from processors import *

path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="final_result")#file name to save the image
def main(image):
    # Enhance the edge of object by observing the contrast of it and the environment
    enhanced1 = gamma_clahe(image, 12, 32)
    enhanced2 = gamma_clahe(image, 8, 8)
    
    # Highlight area that reflect light + remove noise(e.g. glass reflection)
    output1 = generate_reflection_mask(enhanced1, kernel_size=5, threshold=120)
    filtered_output1 = filter_draw_contours(output1, min_contour_area=2500, contour_thickness= 3)

    # Get object edge with high constract with the environemt
    output2 = edge_based_segmentation(enhanced2, canny_low=150, canny_high=350, dilate_iterations=3)
    filtered_output2 = filter_draw_contours(output2, min_contour_area=500, contour_thickness= 3)

    # Combination of these two kinds of edges
    # 2. more about the edge between reflective surfaces and shadows
    # 1. more about the edge between object and environment
    combined1 = cv2.addWeighted(filtered_output2, 1, filtered_output1, 1, 0)

    # Change color of the edges
    output_img = image.copy()
    output_img[combined1 == 255] = [0, 255, 0] 

    return output_img

if __name__ == "__main__":
    main(image)