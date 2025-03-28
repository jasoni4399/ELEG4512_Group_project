import numpy as np

def median_filter_2d(image, kernel_size=3):
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='constant')
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.median(window)
    
    return filtered_image