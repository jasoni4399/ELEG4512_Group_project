__all__ = ["remove_reflection", "gamma_correction", "contrast_stretching", "histogram_normalization",
           "laplacian_filter", "save_image", "process_image"]

from .test import remove_reflection_test as remove_reflection
from .gamma_correction import gamma_correction as gamma_correction
from .contrast_stretching import contrast_stretching as contrast_stretching
from .histogram_normalization import histogram_normalization as histogram_normalization
from .laplacian_filtering import laplacian_filter as laplacian_filter

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