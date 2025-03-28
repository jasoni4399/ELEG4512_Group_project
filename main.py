import cv2
import os
import numpy as np
from processors import *
# make a decorator to time the function
path = "inputs/blur_noisey_photo.jpg"
image = cv2.imread(path)

@process_image(image_path="frequency_domain_separation")#file name to save the image
def main(image):

    outputs = generate_reflection_mask(image, kernal_size=5, threshold=145)
    laplacian = laplacian_filter(outputs, kernel_size=5)
    gamma = gamma_correction(image, 10)
    canny = cv2.Canny(gamma, 150, 200)
    kernel_h = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    # 垂直方向連接 (強化垂直線)
    kernel_v = np.array([[1], [1], [1], [1], [1]], dtype=np.uint8)

    # 水平線強化
    dilated_h_1 = cv2.dilate(laplacian, kernel_h, iterations=1)
    dilated_h = cv2.dilate(canny, kernel_h, iterations=1)
    # 垂直線強化
    dilated_v_1 = cv2.dilate(laplacian, kernel_v, iterations=1)
    dilated_v = cv2.dilate(canny, kernel_v, iterations=1)
    # 合併結果
    connected_edges_1 = cv2.bitwise_or(dilated_h_1, dilated_v_1)
    kernal = np.ones((5,5), np.uint8)
    connected_edges_1 = cv2.dilate(connected_edges_1, kernal, iterations=1)
    connected_edges = cv2.bitwise_or(dilated_h, dilated_v)

    """
    #k means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    Z = outputs2.reshape((-1,3))
    Z = np.float32(Z)
    # apply kmeans clustering
    compactness,labels,centers = cv2.kmeans(Z, 2, None, criteria, 10, flags)
    print("compactness: ", compactness)
    print("labels: ", labels)
    print("centers: ", centers)
    center = np.uint8(centers)
    res = center[labels.flatten()]
    res2 = res.reshape((outputs2.shape))
    """
    #laplacian2 = laplacian_filter(res2, kernel_size=5)
    laplacian3 = connected_edges_1 + connected_edges
    #dilation_kernel = np.ones((5,5), np.uint8)
    #thick_edge = cv2.dilate(laplacian3, dilation_kernel, iterations=2)
    #thick_edge = cv2.erode(laplacian3, dilation_kernel, iterations=1)
    
    

    laplacian3_3channel = cv2.cvtColor(laplacian3, cv2.COLOR_GRAY2BGR)
    red_edges = np.zeros_like(laplacian3_3channel)
    red_edges[:,:,2] = 255  # BGR中的R通道設為255（紅色）

    # 3. 創建掩模（非黑像素的位置）
    mask = cv2.threshold(laplacian3, 1, 255, cv2.THRESH_BINARY)[1]  # 大於0的像素設為255

    # 4. 將紅色邊緣應用到原圖
    final = image.copy()
    final[mask > 0] = red_edges[mask > 0]

    processed = laplacian_filter(image, kernel_size=5)

    return processed

if __name__ == "__main__":
    main(image)