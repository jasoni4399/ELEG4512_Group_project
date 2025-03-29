import cv2
import os
import numpy as np
from processors import *
# make a decorator to time the function
path = "inputs/test3.jpg"
image = cv2.imread(path)

@process_image(image_path="frequency_domain_separation")#file name to save the image
def main(image):
    out = generate_reflection_mask(image, kernal_size=5, threshold=145)
    gamma = gamma_correction(image, 6)
    _sobel = sobel(gamma, ksize=3, dx=1, dy=1, normalize=False, uint8=True)

    kernel_h = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    kernel_v = np.array([[1], [1], [1], [1], [1]], dtype=np.uint8)
    dilated_h = cv2.dilate(_sobel, kernel_h, iterations=1)
    dilated_v = cv2.dilate(_sobel, kernel_v, iterations=1)
    connected_edges = cv2.bitwise_or(dilated_h, dilated_v)
    binary = np.where(connected_edges > 90, 255, 0).astype(np.uint8)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 4)

    # 創建遮罩：只保留面積大於閾值的區域
    min_area = 2000  # 最小像素面積
    mask = np.zeros_like(_sobel)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            mask[labels == i] = 255

    # 可選：用膨脹恢復部分形狀
    final1 = cv2.dilate(mask, kernel, iterations=1)


    laplacian3_3channel = cv2.cvtColor(final1, cv2.COLOR_GRAY2BGR)
    red_edges = np.zeros_like(laplacian3_3channel)
    red_edges[:,:,2] = 255  # BGR中的R通道設為255（紅色）

    # 3. 創建掩模（非黑像素的位置）
    mask = cv2.threshold(final1, 1, 255, cv2.THRESH_BINARY)[1]  # 大於0的像素設為255

    # 4. 將紅色邊緣應用到原圖
    final = image.copy()
    final[mask > 0] = red_edges[mask > 0]

    return final

if __name__ == "__main__":
    main(image)