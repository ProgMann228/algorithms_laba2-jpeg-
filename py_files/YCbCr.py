import numpy as np
from PIL import Image
import os


def RGB_to_YCbCr(pixels):
    ycbcr = np.empty_like(pixels)
    height, width = pixels.shape[0], pixels.shape[1] #размерности матрицы(ширина,вытсота)

    for i in range(height):  # По строкам (ось i)
        for j in range(width):  # По столбцам (ось j)
            pixel = pixels[i, j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]
            rr, gg, bb = float(r), float(g), float(b)

            y = int(round(0.299 * rr + 0.587 * gg + 0.114 * bb))
            cb = int(round(-0.168736 * rr - 0.331264 * gg + 0.5 * bb + 128))
            cr = int(round(0.5 * rr - 0.418688 * gg - 0.081312 * bb + 128))

            # Ограничиваем значения и записываем
            ycbcr[i, j] = (
                max(0, min(255, y)),
                max(0, min(255, cb)),
                max(0, min(255, cr))
            )
    return ycbcr

def YCbCr_to_RGB(y: int, cb: int, cr: int):
    y,cb,cr = float(y),float(cb),float(cr)

    r = int(round(y + 1.402*(cr - 128)))
    g = int(round(y - 0.34414*(cb - 128) - 0.71414*(cr - 128)))
    b = int(round(y + 1.772*(cb - 128)))

    # Ограничиваем диапазон
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return r, g, b


def Img(image_path):
    img = Image.open(image_path)

    if img.mode != "RGB":
        img = img.convert("RGB")  # Приведение к стандартному цветному режиму
    #print(f"Исходный режим изображения: {img.mode}")
    mas = np.array(img)

    # Размер несжатых данных в байтах:
    uncompressed_size_bytes = mas.size * mas.itemsize
    print(f"Размер несжатых данных: {uncompressed_size_bytes} байт")
    return mas


def to_blocks(matrix, N):
    h = matrix.shape[0]
    w = matrix.shape[1]

    if h % N != 0:
        new_h = h + (N - h % N)
    else:
        new_h = h

    if w % N != 0:
        new_w = w + (N - w % N)
    else:
        new_w = w

    padded = np.pad(
        matrix,
        pad_width=((0, new_h - h), (0, new_w - w)),
        mode="constant",
        constant_values=0
    )
    blocks = []
    for y in range(0, new_h, N):
        for x in range(0, new_w, N):
            block = padded[y:(y+N), x:(x+N)]
            blocks.append((y, x, block)) #координаты начала блока и сам срез

    print(f"количество блоков:",len(blocks))
    return blocks,new_h,new_w


def from_blocks(blocks, height, width,padded_h, padded_w, N):
    #padded_h = ((height + N - 1) // N) * N
    #padded_w = ((width + N - 1) // N) * N
    full = np.zeros((padded_h, padded_w), dtype=blocks[0].dtype)
    print(f"\nsize blocks: ", len(blocks))
    print(height, width,padded_h, padded_w)
    idx = 0
    for y in range(0, padded_h, N):
        for x in range(0, padded_w, N):
            full[y:(y+N), x:(x+N)] = blocks[idx]
            idx += 1

    return full[:height, :width] #обрезаем padding

"""
def downsampling(pixels,N):
    height = pixels.shape[0]
    width = pixels.shape[1]
    channels = pixels.shape[2]

    if height % N != 0:
        new_h = height + (N - height % N)
    else:
        new_h = height

    if width % N != 0:
        new_w = width + (N - width % N)
    else:
        new_w = width

    # Дополняем изображение нулями (по высоте и ширине, каналы не трогаем)
    padded = np.pad(
        pixels,
        pad_width=((0, new_h - height), (0, new_w - width), (0, 0)),
        mode="constant",
        constant_values=0
    )

    # Создаём массив для результата (уменьшенное изображение)
    downsampled = np.zeros((new_h, new_w, channels), dtype=np.uint8)

    # Для Y компоненты просто копируем значения
    downsampled[:, :, 0] = padded[:, :, 0]

    # Перебор блоков
    blocks = to_blocks(padded, N)
    for y, x, block in blocks:
        #компонента Cb
        avg_cb = np.mean(block[:, :, 1])
        downsampled[y:y + N, x:x + N, 1] = avg_cb
        # компонента Cr
        avg_cr = np.mean(block[:, :, 2])
        downsampled[y:y + N, x:x + N, 2] = avg_cr

    return downsampled
"""

def downsampling(pixels,N):
    Y=pixels[:, :, 0]
    height, width = Y.shape

#Cb канал
    Cb_matrix = pixels[:, :, 1]
    Cb_blocks,Cb_pad_h,Cb_pad_w = to_blocks(Cb_matrix, 2)  # делим на блоки 2*2 пикселя
    padded_height = ((height + 1) // 2) * 2  # округление вверх до ближайшего кратного 2
    padded_width = ((width + 1) // 2) * 2
    Cb = np.zeros((padded_height // 2, padded_width // 2), dtype=np.uint8)

    for y, x, block in Cb_blocks:
        sum_pix = np.sum(block)
        avg = sum_pix / 4
        Cb[y // 2, x // 2] = avg  # сохраняем в уменьшенную матрицу

# Cr канал#
    Cr_matrix = pixels[:, :, 2]
    Cr_blocks,Cr_pad_h,Cr_pad_w = to_blocks(Cr_matrix, 2)
    Cr = np.zeros((padded_height // 2, padded_width // 2), dtype=np.uint8)

    for y, x, block in Cr_blocks:
            sum_pix = np.sum(block)
            avg = sum_pix / 4
            Cr[y // 2, x // 2] = avg  # сохраняем в уменьшенную матрицу

    Cb_new_h, Cb_new_w = Cb.shape
    Cr_new_h, Cr_new_w = Cr.shape
    return Y, Cb,Cr, Cb_new_h, Cb_new_w,Cr_new_h, Cr_new_w


def upsampling(Y, Cb, Cr):
    height, width = Y.shape

    # Восстанавливаем Cb
    Cb_upsampled = np.repeat(np.repeat(Cb, 2, axis=0), 2, axis=1)
    Cb_h, Cb_w = Cb_upsampled.shape

    # Восстанавливаем Cr
    Cr_upsampled = np.repeat(np.repeat(Cr, 2, axis=0), 2, axis=1)
    Cr_h, Cr_w = Cr_upsampled.shape

    # Обрезаем до исходных размеров, если нужно
    Cb_cropped = Cb_upsampled[:height, :width]
    Cr_cropped = Cr_upsampled[:height, :width]

    # Собираем обратно в одно изображение
    pixels = np.stack((Y, Cb_cropped, Cr_cropped), axis=2)

    return pixels


def bits_to_bytes(bit_string):
    # Добавляем нули в конец, чтобы длина делилась на 8
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += '0' * padding

    # Разбиваем на байты
    byte_list = []
    for i in range(0, len(bit_string), 8):
        byte = bit_string[i:i+8]
        byte_list.append(int(byte, 2))

    return bytes(byte_list), padding

def bytes_to_bits(byte_data, padding=0):
    bits = ''.join(f'{byte:08b}' for byte in byte_data)
    if padding > 0:
        bits = bits[:-padding]  # Удаляем доп. биты, добавленные при кодировании
    return bits



