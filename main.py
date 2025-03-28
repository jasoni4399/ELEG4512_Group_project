import cv2
import os
import numpy as np
from processors import *
# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="frequency_domain_separation")#file name to save the image
def main(image):

    outputs = generate_reflection_mask(image, kernal_size=5, threshold=145)
    laplacian = laplacian_filter(outputs, kernel_size=5)
    gamma = gamma_correction(image, 10)
    outputs2 = generate_reflection_mask(gamma, kernal_size=5, threshold=120)
    outputs2 = histogram_normalization(outputs2, 0, 255)
    laplacian2 = laplacian_filter(outputs2, kernel_size=5)
    laplacian3 = laplacian + laplacian2
    dilation_kernel = np.ones((5,5), np.uint8)
    thick_edge = cv2.dilate(laplacian3, dilation_kernel, iterations=2)
    #thick_edge = cv2.erode(laplacian3, dilation_kernel, iterations=1)
    
    

    laplacian3_3channel = cv2.cvtColor(laplacian3, cv2.COLOR_GRAY2BGR)
    final = cv2.add(image, laplacian3_3channel)
    return outputs2

if __name__ == "__main__":
    main(image)