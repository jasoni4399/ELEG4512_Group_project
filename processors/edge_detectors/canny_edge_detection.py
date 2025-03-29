import cv2
import numpy as np
from ..utils.display import display

def canny_edge_detection(
    img, 
    threshold1: int = 50, 
    threshold2: int = 150, 
    apertureSize: int = 3,
    normalize: bool = True,
    show_result: bool = False,
    window_size: tuple[int, int] = (10, 6),
    dpi: int = 100 
) -> np.ndarray:

    #if len(img.shape) == 3:
    #    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(img, threshold1, threshold2, apertureSize=apertureSize)
    edges = np.abs(edges)

    if normalize:
        edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    edges = edges.astype(np.uint8)

    if show_result:
        display(edges, title='Laplacian Edge Detection', window_size=window_size, dpi=dpi)
        
    return edges