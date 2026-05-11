"""
All the train images provided are grayscale, while the testing video has 3 channels. 
In this file, i convert the pictures to 3 channels.
Run this file from the root of the folder
"""

import cv2
from pathlib import Path

directory = Path("data/valid/images")
output_dir = Path("data/data3channels/valid/images")
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
            bgr_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            save_path = str(output_dir / img_path_obj.name)
            cv2.imwrite(save_path, bgr_image)

 
