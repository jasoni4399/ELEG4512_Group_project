import cv2
import os
from processors import remove_reflection

path = "inputs/blur_noisey_photo.jpg"
def process_image(image_path):
    image = cv2.imread(image_path)
    
    processed_image = remove_reflection(image)
    
    os.makedirs("outputs", exist_ok=True)
    
    output_path = os.path.join("outputs", "output.jpg")
    cv2.imwrite(output_path, processed_image)
    
    print(f"Processed image saved to: {output_path}")
    return processed_image

if __name__ == "__main__":
    process_image(path)