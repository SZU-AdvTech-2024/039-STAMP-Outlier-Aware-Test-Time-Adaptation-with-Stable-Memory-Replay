import os
import numpy as np
from PIL import Image
from imagecorruptions import corrupt


input_path = "/home/ubuntu/stamp_ln/0149.jpg"
output_dir = "/home/ubuntu/stamp_ln/corruptions"  


os.makedirs(output_dir, exist_ok=True)


original_image = Image.open(input_path)
original_image = original_image.resize((32, 32)) 
image_array = np.array(original_image)


corruption_types = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 
    'snow', 'frost', 'fog', 'brightness', 'contrast', 
    'elastic_transform', 'pixelate', 'jpeg_compression'
]
severity_levels = [1, 2, 3, 4, 5]


for corruption in corruption_types:
    for severity in severity_levels:
        corrupted_image_array = corrupt(image_array, corruption_name=corruption, severity=severity)
        corrupted_image = Image.fromarray(corrupted_image_array)
        

        output_path = os.path.join(output_dir, f"{corruption}_s{severity}.jpg")
        corrupted_image.save(output_path)

print(f"所有 corruption 图片已保存到 {output_dir}")
