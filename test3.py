import cv2
import os
import numpy as np
from processors import *
# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="frequency_domain_separation")#file name to save the image
def main(image):

    outputs = generate_reflection_mask(image, kernal_size=3, threshold=145)
    test = cv2.cvtColor(outputs, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    L_channel = lab[:, :, 0]

    gamma = 2
    gamma2 = 32
    gamma_corrected = np.power(L_channel / 255.0, gamma) * 255
    gamma_corrected_2 = np.power(L_channel / 255.0, gamma2) * 255
    gamma_corrected = gamma_corrected.astype(np.uint8)
    gamma_corrected_2 = gamma_corrected_2.astype(np.uint8)

    filtered = cv2.bilateralFilter(gamma_corrected, 9, 75, 75)

    sobelx = cv2.Sobel(filtered, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(filtered, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)

    log_img = np.log1p(filtered.astype(np.float32))
    log_blur = cv2.GaussianBlur(log_img, (15, 15), 10)
    reflectance = cv2.normalize(log_img - log_blur, None, 0, 255, cv2.NORM_MINMAX)
    reflectance = reflectance.astype(np.uint8)

    combined1 = cv2.addWeighted(gamma_corrected, 0.4, gamma_corrected_2, 0.6, 0)
    combined2 = cv2.addWeighted(combined1, 0.7, reflectance, 0.3, 0)

    # 加強邊緣
    sobelx = cv2.Sobel(combined2, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(combined2, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobelx, sobely)

    # 非線性映射強化強邊緣
    gradient_enhanced = np.power(gradient / gradient.max(), 0.5) * 255
    gradient_enhanced = gradient_enhanced.astype(np.uint8)

    # 再融合一次進原圖
    combined3 = cv2.addWeighted(combined2, 1, gradient_enhanced, 1, 0)

    # 用新的圖再做 edge detection
    sobelx = cv2.Sobel(combined3, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(combined3, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)

    edges_norm = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    edges_uint8 = edges_norm.astype(np.uint8)
    _, thresh = cv2.threshold(edges_uint8, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_img = image.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(output_img, [cnt], 0, (0, 255, 0), 2)

    return output_img

if __name__ == "__main__":
    main(image)