__all__ = ["remove_reflection",
            "save_image",
            "process_image",
            "gamma_correction",
            "laplacian_filter",
            "rgb_laplacian_filter",
            "histogram_equalization",
            "rgb_histogram_equalization",
            ]

from .remove_reflection import remove_reflection_test as remove_reflection
from .gamma_correction import gamma_correction
from .laplacian_filtering import laplacian_filter as laplacian_filter
from .laplacian_filtering import rgb_laplacian_filter as rgb_laplacian_filter
from .histogram_equalization import histogram_equalization as histogram_equalization
from .histogram_equalization import rgb_histogram_equalization as rgb_histogram_equalization

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
