import numpy as np

def gamma_correction(
    image: np.ndarray, 
    gamma: float, 
    output_dtype: type = np.uint8
    ) -> np.ndarray:

    # Normalize input to [0, 1] based on dtype
    if image.dtype == np.uint8:
        normalized = image.astype(np.float64) / 255.0
    elif image.dtype == np.uint16:
        normalized = image.astype(np.float64) / 65535.0
    elif image.dtype in (np.float32, np.float64):
        normalized = image.copy()
        if np.max(normalized) > 1.0 or np.min(normalized) < 0.0:
            raise ValueError("Float image must be in range [0, 1].")
    else:
        raise ValueError(f"Unsupported input dtype: {image.dtype}. Expected uint8/uint16/float.")

    # Apply gamma correction
    corrected = np.power(normalized, gamma)

    # Convert to desired output dtype
    if output_dtype == np.uint8:
        return (corrected * 255.0).clip(0, 255).astype(np.uint8)
    elif output_dtype in (np.float32, np.float64):
        return corrected.astype(output_dtype)
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}. Expected uint8/float32/float64.")