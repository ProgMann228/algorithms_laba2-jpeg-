import numpy as np
from PIL import Image
import os
from bitarray import bitarray
import pickle
from compressor import compress
from decompressor import decompress
from YCbCr import Img


image_paths = [
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/Lenna_(test_image).png",
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/gray_Lenna.png",
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/bw_dithered_Lenna.png",
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/bw_nodither_Lenna.png",
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048.jpg",
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_gray.png",
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_bw_dithered.png",
    "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_bw_nodither.png",
]

qualities = [0, 20, 40, 60, 80, 100]

output_folder = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/results_pic"
os.makedirs(output_folder, exist_ok=True)

for image_path in image_paths:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    for q in qualities:
        print(f"\nCOMPRESSION HAS STARTED for {image_name} with quality {q}")
        outfile = "temp_output.bin"
        compress(image_path, q, outfile)

        print(f"\nDECOMPRESSION HAS STARTED for {image_name} with quality {q}")
        restored = os.path.join(output_folder, f"{image_name}_q{q}.png")
        decompress(restored,outfile)

        print(f"Saved: {restored}")


# RGB np.ndarray - Grayscale PIL.Image
def to_grayscale(rgb_array):
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError("Ожидается RGB-массив с формой (H, W, 3)")
    img_rgb = Image.fromarray(rgb_array.astype('uint8'), 'RGB')
    return img_rgb.convert("L")

# Grayscale - BW с дизерингом (встроенный Floyd–Steinberg)
def to_bw_dithered(img_gray):
    return img_gray.convert("1")  # дизеринг включён по умолчанию

# Grayscale - BW без дизеринга (пороговая)
def to_bw_threshold(img_gray, threshold=127):
    return img_gray.point(lambda p: 255 if p > threshold else 0)
"""
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/Lenna_(test_image).png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/gray_Lenna.png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/bw_dithered_Lenna.png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/bw_nodither_Lenna.png"

image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048.jpg"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_gray.png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_bw_dithered.png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_bw_nodither.png"

#image_path = "C:/Users/User/OneDrive/Документы/алгоритмы вуз/homework1/IMG_3405.jpg"

outfile="output.bin"

qual=100
print("\nCOMPRESSION HAS STARTED")
compressed=compress(image_path,qual,outfile)

print("\nDECOMPRESSION HAS STARTED")
decompressed=decompress(outfile)

img_mas=Img(image_path)

# Преобразуем
gray = to_grayscale(img_mas)
bw_dithered = to_bw_dithered(gray)
bw_threshold = to_bw_threshold(gray)

# Сохраняем
gray.save("gray.png")
bw_dithered.save("bw_dithered.png")
bw_threshold.save("bw_nodither.png")
"""