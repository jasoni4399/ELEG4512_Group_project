__all__ = ["save_image", "process_image","gamma_correction", "laplacian_filter", "histogram_normalization", 
           "contrast_stretching", "average_filter"]

from .gamma_correction import gamma_correction
from .laplacian_filtering import laplacian_filter as laplacian_filter
from .histogram_normalization import histogram_normalization
from .contrast_stretching import contrast_stretching
from .average_filtering import average_filter

import cv2
import os

def save_image(image, path):
    output_path = os.path.join("outputs", path+".jpg")
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")

def process_image(image_path):
    def wrapper(func):
        def run(*args, **kwargs):
            processed_image = func(*args, **kwargs)
            save_image(processed_image, image_path)
            return processed_image
        return run
    return wrapper