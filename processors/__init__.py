__all__ = ["save_image", "process_image","gamma_correction", "laplacian", "histogram_normalization", 
           "contrast_stretching", "average_filter", "gaussian_blurring", "frequency_domain_separation",
           "generate_reflection_mask", "multi_scale_inpainting", "median_filter_2d", "dilation", "erosion",
           "sobel", "display", "LoG", "canny", "morphologyEx", "fft"] 

from .edge_detection_methods.laplacian_edge_detection import laplacian_edge_detection as laplacian
from .edge_detection_methods.sobel_edge_detection import sobel_edge_detection as sobel
from .edge_detection_methods.log_edge_detection import log_edge_detection as LoG
from .edge_detection_methods.canny_edge_detection import canny_edge_detection as canny
from .edge_detection_methods.morphological_gradient import morphological_gradient as morphologyEx
from .edge_detection_methods.fourier_highpass_filter import fourier_highpass_filter as fft

from .gamma_correction import gamma_correction

from .histogram_normalization import histogram_normalization
from .contrast_stretching import contrast_stretching
from .average_filtering import average_filter
from .gaussian_blurring import gaussian_blurring
from .freq_domain_separation import frequency_domain_separation as frequency_domain_separation
from .reflection_mask import generate_reflection_mask as generate_reflection_mask
from .multi_scale_inpainting import multi_scale_inpainting
from .median_filtering import median_filter_2d as median_filter_2d
from .dilation import dilation
from .erosion import erosion
from .utils.display import display

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