import cv2
import numpy as np

def contrast_stretching(image, lower_percent=2, upper_percent=98):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    lower = np.percentile(gray, lower_percent)
    upper = np.percentile(gray, upper_percent)
    
    result = np.clip((gray - lower) * (255.0 / (upper - lower)), 0, 255).astype(np.uint8)
    
    if len(image.shape) == 3:
        channels = []
        for channel in cv2.split(image):
            lower_ch = np.percentile(channel, lower_percent)
            upper_ch = np.percentile(channel, upper_percent)
            stretched_ch = np.clip((channel - lower_ch) * (255.0 / (upper_ch - lower_ch)), 0, 255)
            channels.append(stretched_ch.astype(np.uint8))
        result = cv2.merge(channels)
    
    return result