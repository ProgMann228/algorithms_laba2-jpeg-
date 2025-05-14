import os
import pickle
import matplotlib.pyplot as plt
from compressor import compress

# Путь к исходному изображению
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/Lenna_(test_image).png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/gray_Lenna.png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/bw_dithered_Lenna.png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/bw_nodither_Lenna.png"

#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048.jpg"
#1image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_gray.png"
#image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_bw_dithered.png"
image_path = "D:/новая папка/PyCharm 2025.1/alg_sem2laba2/2048x2048_bw_nodither.png"

# Уровни качества с шагом 5
quality_levels = range(0, 101, 5)

# Результаты: качество → размер (в байтах)
compression_results = {}

# Временный файл для pickle-сжатия
temp_outfile = "temp_output.bin"

for quality in quality_levels:
    # Сжимаем изображение во временный файл
    compress(image_path, qual=quality, outfile=temp_outfile)

    # Получаем размер временного файла
    file_size = os.path.getsize(temp_outfile)
    compression_results[quality] = file_size

    print(f"Качество: {quality}%, Размер: {file_size} байт")

# Удаляем временный файл после завершения
if os.path.exists(temp_outfile):
    os.remove(temp_outfile)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(compression_results.keys(), compression_results.values(), 'bo-')
plt.xlabel("Качество сжатия (0-100)")
plt.ylabel("Размер сжатого файла (байты)")
plt.title("Зависимость размера файла от уровня качества")
plt.grid(True)
plt.tight_layout()

# Сохраняем график в файл (например, в формате PNG)
output_graph_path = "compression_quality_vs_size.png"
plt.savefig(output_graph_path)
print(f"График сохранён в файл: {output_graph_path}")

# Показываем график (опционально, можно убрать если не нужно)
#plt.show()

# Завершаем программу
exit()
