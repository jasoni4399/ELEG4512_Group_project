import cv2
import os
import numpy as np
from processors import *

path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

def enhance_contrast(img, a: int = 8, b: int = 8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    gamma = a
    enhanced = np.power(gray / 255.0, gamma) * 255.0
    enhanced = enhanced.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=b, tileGridSize=(8, 8))
    normalized = clahe.apply(enhanced)    

    return normalized

def threshold_segmentation(img):
    img = cv2.GaussianBlur(img, (7,7), 0)
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 45, 16
    )
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closed

def edge_based_segmentation(img):
    # Gaussian blurring
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    
    # Sobel
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)
    gradient = cv2.magnitude(sobelx, sobely)
    gradient_enhanced = np.power(gradient / gradient.max(), 0.5) * 255
    gradient_enhanced = gradient_enhanced.astype(np.uint8)

    # Gaussian blurring
    blurred = cv2.GaussianBlur(gradient_enhanced, (5,5), 0)

    # Canny
    edges = cv2.Canny(blurred, 150, 350)

    # Gaussian blurring
    blurred = cv2.GaussianBlur(edges, (3,3), 0)

    # Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=3)
    
    return edges_dilated

def postprocess(mask):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed

@process_image(image_path="combined2")#file name to save the image
def main(image):
    enhanced1 = enhance_contrast(image, 12, 32)
    enhanced2 = enhance_contrast(image, 8, 8)
    
    output1 = generate_reflection_mask(enhanced1, kernel_size=5, threshold=120)
    contours, _ = cv2.findContours(output1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_output1 = np.zeros_like(output1)
    for cnt in contours:
        if cv2.contourArea(cnt) > 2500:
            cv2.drawContours(filtered_output1, [cnt], -1, 255, 3)
    output2 = edge_based_segmentation(enhanced2)
    contours, _ = cv2.findContours(output2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_output2 = np.zeros_like(output2)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(filtered_output2, [cnt], -1, 255, 3)
    combined1 = cv2.addWeighted(filtered_output2, 1, filtered_output1, 1, 0)

    output_img = image.copy()
    output_img[combined1 == 255] = [0, 255, 0] 

    return output_img


if __name__ == "__main__":
    main(image)