import numpy as np

#Для одного канала
def quantize(block, Q):
    return np.round(block / Q).astype(np.int32)

def dequantize(block, Q):
    return block * Q

"""
#для 3х каналов
def quantize(block, Q_Y,Q_C): #block - матрица коэффициентов DCT
    result = np.zeros_like(block, dtype=np.int32)

    Y_matrix = block[:, :, 0]
    result[:, :, 0] = np.round(Y_matrix / Q_Y).astype(np.int32)

    Cb_matrix = block[:, :, 1]
    result[:, :, 1] = np.round(Cb_matrix / Q_C).astype(np.int32)

    Cr_matrix = block[:, :, 2]
    result[:, :, 2] = np.round(Cr_matrix / Q_C).astype(np.int32)
    return result


def dequantize(block, Q):
    result = np.zeros_like(block, dtype=np.float32)

    Y_matrix = block[:, :, 0]
    result[:, :, 0] = Y_matrix * Q

    Cb_matrix = block[:, :, 1]
    result[:, :, 1] = Cb_matrix * Q

    Cr_matrix = block[:, :, 2]
    result[:, :, 2] = Cr_matrix * Q
    return result
"""

def scale_quant_matrix(Q, quality):
    quality=np.clip(quality,1,100)
    if quality < 50:
        scale = 5000 / quality
    elif quality==50:
        return Q
    else:
        scale = 200 - 2 * quality
    Q_scaled = np.round((Q * scale + 50) / 100).clip(1,255)
    return Q_scaled.astype(np.uint8)
