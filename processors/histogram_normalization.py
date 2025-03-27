import cv2
import numpy as np

def histogram_normalization(image, min_out=0, max_out=255):
    if len(image.shape) == 3:
        normalized_channels = []
        for channel in cv2.split(image):
            min_in = np.min(channel)
            max_in = np.max(channel)

            normalized = ((channel - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
            normalized = np.clip(normalized, min_out, max_out).astype(np.uint8)
            normalized_channels.append(normalized)
        normalized_image = cv2.merge(normalized_channels)
    else:
        min_in = np.min(image)
        max_in = np.max(image)
        normalized_image = ((image - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
        normalized_image = np.clip(normalized_image, min_out, max_out).astype(np.uint8)
    
    return normalized_image
