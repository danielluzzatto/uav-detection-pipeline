"""
All the train images provided are grayscale, while the testing video has 3 channels. 
In this file, i convert the pictures to 3 channels, and use clahe, gaussian blur to make the features pop.
Run this file from the root of the folder
"""

import cv2
from pathlib import Path

directory = Path("data/test/images")
output_dir = Path("data/augmented_data/test/images")
output_dir.mkdir(exist_ok=True)
image_paths = [f for f in directory.iterdir() if f.is_file() and f.suffix in ['.jpg', '.png', '.jpeg']]

for i in range(len(image_paths)):
    if not image_paths:
        print(f"No images found in {directory.absolute()}")
    else:
        img_path_obj = image_paths[i]
        gray_image = cv2.imread(str(img_path_obj), cv2.IMREAD_GRAYSCALE)

        if gray_image is None:
            print(f"Failed to load image:")
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_clahe = clahe.apply(gray_image)
            gaussian_blur = cv2.GaussianBlur(gray_clahe, (0, 0), 3)
            sharpened = cv2.addWeighted(gray_clahe, 1.5, gaussian_blur, -0.5, 0)
            final_preprocessed = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            save_path = str(output_dir / img_path_obj.name)
            cv2.imwrite(save_path, final_preprocessed)

 
