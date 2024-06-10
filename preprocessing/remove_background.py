from rembg import remove
from tqdm import tqdm
import os
from PIL import Image, UnidentifiedImageError

input_dir = '/home/kdt-admin/data/images/'
output_dir = '/home/kdt-admin/data/modified_images/'

file_list = [filename for filename in os.listdir(input_dir) if filename.endswith(".jpg") or filename.endswith(".png")]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for filename in tqdm(file_list, desc="Processing"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            input_img = Image.open(input_path)
            output_img = remove(input_img)
            output_img.save(output_path)
        except UnidentifiedImageError:
            print(f"Error: Unable to identify image file '{input_path}'. Skipping.")
            continue