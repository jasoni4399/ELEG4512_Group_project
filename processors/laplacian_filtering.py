import cv2
import numpy as np

def laplacian_filter(image, kernel_size=3, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT):

    if len(image.shape) == 3:
        #RGB image
        gray = cv2.cvtColor(image,cv2.COLORR)
    else:
        gray = image.copy()
    
    laplacian = cv2.Laplacian(gray, ddepth=cv2.CV_64F, ksize=kernel_size, scale=scale, delta=delta, borderType=border_type)
    
    laplacian_abs = np.absolute(laplacian)
    laplacian_scaled = np.uint8(laplacian_abs / np.max(laplacian_abs) * 255)
    
    return laplacian_scaled

def rgb_laplacian_filter(image, kernel_size=3, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT):
    b, g, r = cv2.split(image)
    
    b_lap = laplacian_filter(b, kernel_size, scale, delta, border_type)
    g_lap = laplacian_filter(g, kernel_size, scale, delta, border_type)
    r_lap = laplacian_filter(r, kernel_size, scale, delta, border_type)
    
    return cv2.merge((b_lap, g_lap, r_lap))