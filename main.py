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
    lap = laplacian(image, ksize=5)
    gamma = gamma_correction(image, 10)
    laplacian_iamge = laplacian(image, ksize=5)
    time = cv2.subtract(gamma, outputs)
    canny = cv2.Canny(gamma, 150, 200)
    kernel_h = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    # 垂直方向連接 (強化垂直線)
    kernel_v = np.array([[1], [1], [1], [1], [1]], dtype=np.uint8)

    # 水平線強化
    dilated_h_1 = cv2.dilate(lap, kernel_h, iterations=1)
    dilated_h = cv2.dilate(canny, kernel_h, iterations=1)
    dilated_h_2 = cv2.dilate(laplacian_iamge, kernel_h, iterations=1)
    # 垂直線強化
    dilated_v_1 = cv2.dilate(lap, kernel_v, iterations=1)
    dilated_v = cv2.dilate(canny, kernel_v, iterations=1)
    dilated_v_2 = cv2.dilate(laplacian_iamge, kernel_h, iterations=1)
    # 合併結果
    connected_edges_1 = cv2.bitwise_or(dilated_h_1, dilated_v_1)
    kernal = np.ones((5,5), np.uint8)
    connected_edges_1 = cv2.dilate(connected_edges_1, kernal, iterations=1)
    connected_edges = cv2.bitwise_or(dilated_h, dilated_v)
    connected_edges_2 = cv2.bitwise_or(dilated_h_2, dilated_v_2)


    laplacian3 = cv2.add(connected_edges_1, connected_edges)
    laplacian3 = cv2.add(laplacian3, connected_edges_2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))  # 橢圓形核
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(laplacian3, 4)

    # 創建遮罩：只保留面積大於閾值的區域
    min_area = 100  # 最小像素面積
    mask = np.zeros_like(laplacian3)
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

    return laplacian3



if __name__ == "__main__":
    main(image)