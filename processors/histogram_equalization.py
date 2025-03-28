#histogram equalization
import cv2
import numpy as np
import os

def histogram_equalization(image):
    if len(image.shape) == 3:
        #RGB image
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    equalized = cv2.equalizeHist(gray/255.0) * 255.0
    
    return equalized

def rgb_histogram_equalization(image):
    b, g, r = cv2.split(image)
    
    b_eq = histogram_equalization(b)
    g_eq = histogram_equalization(g)
    r_eq = histogram_equalization(r)
    
    return cv2.merge((b_eq, g_eq, r_eq))

#adaptive histogram equalization
def adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    if len(image.shape) == 3:
        #RGB image
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
    else:
        l = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)
    
    if len(image.shape) == 3:
        return cv2.merge((cl, a, b))
    else:
        return cl
    
def rgb_adaptive_histogram_equalization(image, clip_limit=2.0, grid_size=(8, 8)):
    b, g, r = cv2.split(image)
    
    b_eq = adaptive_histogram_equalization(b, clip_limit, grid_size)
    g_eq = adaptive_histogram_equalization(g, clip_limit, grid_size)
    r_eq = adaptive_histogram_equalization(r, clip_limit, grid_size)
    
    return cv2.merge((b_eq, g_eq, r_eq))