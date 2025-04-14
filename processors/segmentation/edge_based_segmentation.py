import cv2
import numpy as np

def edge_based_segmentation(
    img: np.ndarray,
    canny_low: int = 150,
    canny_high: int = 350,
    dilate_iterations: int = 3,
    output_dtype: type = np.uint8,
) -> np.ndarray:
    # Check input dtype and convert to uint8 if needed
    if img.dtype not in (np.uint8, np.float32, np.float64):
        raise ValueError("Input image must be uint8, float32, or float64.")
    
    if len(img.shape) != 2:
        raise ValueError("Input image must be grayscale (single-channel).")

    # Convert float inputs to uint8 for processing
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

    # Step 1: Gaussian blurring
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Step 2: Sobel edge detection
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)
    gradient = cv2.magnitude(sobelx, sobely)
    
    # Normalize and enhance gradient
    gradient_enhanced = np.power(gradient / gradient.max(), 0.5) * 255
    gradient_enhanced = gradient_enhanced.astype(np.uint8)

    # Step 3: Gaussian blurring
    blurred = cv2.GaussianBlur(gradient_enhanced, (5, 5), 0)
    # Step 4: Canny edge detection (requires uint8 input)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    # Step 5: Gaussian blurring
    blurred = cv2.GaussianBlur(edges, (3, 3), 0)
    # Step 6: Dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=dilate_iterations)

    # Convert output to desired dtype
    if output_dtype == np.uint8:
        return edges_dilated
    elif output_dtype in (np.float32, np.float64):
        return edges_dilated.astype(output_dtype) / 255.0
    else:
        raise ValueError("output_dtype must be uint8, float32, or float64.")
    