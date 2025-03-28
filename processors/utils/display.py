import matplotlib.pyplot as plt
import cv2
import numpy as np

def display(img,  
            title: str = 'Image',
            normalize: bool = True,
            window_size: tuple[int, int] = (10, 6),
            dpi: int = 100) -> None:

    plt.figure(figsize=window_size, dpi=dpi)
    if normalize:
        plt.imshow(img, cmap='gray')
    else:
        abs_edges = np.absolute(img)
        vmax = np.percentile(abs_edges, 99)
        plt.imshow(abs_edges, cmap='gray', vmin=0, vmax=vmax)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()