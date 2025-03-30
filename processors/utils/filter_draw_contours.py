import cv2
import numpy as np

def filter_draw_contours(
    binary_image: np.ndarray,
    min_contour_area: int = 500,
    contour_thickness: int = 3,
    output_dtype: type = np.uint8
) -> np.ndarray:
    
    # Check if the input is a binary image
    if binary_image.dtype != np.uint8:
        raise ValueError("The input must be a binary image of type uint8")
    
    if len(binary_image.shape) != 2:
        raise ValueError("The input must be a single-channel binary image")

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    filtered_image = np.zeros_like(binary_image)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(filtered_image, [cnt], -1, 255, contour_thickness)

    # Change to desired data type
    if output_dtype == np.uint8:
        return filtered_image
    elif output_dtype in (np.float32, np.float64):
        return filtered_image.astype(output_dtype) / 255.0
    else:
        raise ValueError("Unsupported output type, please use uint8/float32/float64")