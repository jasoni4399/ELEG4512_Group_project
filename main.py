import cv2
import os
import numpy as np
from processors import frequency_domain_separation, generate_reflection_mask, multi_scale_inpainting, contrast_stretching, histogram_normalization, laplacian_filter, gamma_correction, contrast_stretching, average_filter, gaussian_blurring
from processors import process_image, median_filter_2d, dilation, erosion

# make a decorator to time the function
path = "inputs/old_image.jpg"
image = cv2.imread(path)

@process_image(image_path="frequency_domain_separation")#file name to save the image
def main(image):
    image_hist = histogram_normalization(image, min_out=20, max_out=240)
    image_gamma = gamma_correction(image_hist, gamma=2)

    edge_laplcian = laplacian_filter(image_gamma, kernel_size=5)
    image = cv2.cvtColor(edge_laplcian, cv2.COLOR_BGR2GRAY)
    print(image.shape)
    kernel = np.ones((3, 3), dtype=np.uint8)
    outputs = dilation(image, kernel)

    return outputs

if __name__ == "__main__":
    main(image)