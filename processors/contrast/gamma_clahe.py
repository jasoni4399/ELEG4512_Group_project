import cv2
import numpy as np
from .gamma_correction import gamma_correction

def gamma_clahe(
    img: np.ndarray,
    gamma: int = 8,
    cliplimit: int = 8,
    gamma_output_dtype: type = np.uint8,
) -> np.ndarray:
    
    # Convert to grayscale if needed
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()

    # Apply gamma correction
    enhanced = gamma_correction(gray, gamma=gamma, output_dtype=gamma_output_dtype)

    # CLAHE requires uint8 input
    if enhanced.dtype != np.uint8:
        enhanced = (enhanced * 255.0).clip(0, 255).astype(np.uint8)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(8, 8))
    normalized = clahe.apply(enhanced)

    return normalized