import numpy as np

def create_dct_matrix(N):
    C = np.zeros((N, N))
    for k in range(N):
        for n in range(N):
            if k == 0:
                C[k, n] = 1 / np.sqrt(N)
            else:
                C[k, n] = np.sqrt(2 / N) * np.cos(np.pi * (2*n + 1) * k / (2 * N))
    return C

def DCT_block(block):
    N = block.shape[0]
    C = create_dct_matrix(N)
    return C @ block @ C.T  # DCT = C * block * C^T

def iDCT_block(block):
    N = block.shape[0]
    C = create_dct_matrix(N)
    return C.T @ block @ C  # inverse DCT = C^T * block * C

def DCT_general(blocks):
    return [(y, x, DCT_block(block)) for y, x, block in blocks]

def iDCT_general(blocks):
    return [np.clip(np.round(iDCT_block(block)), 0, 255).astype(np.uint8) for block in blocks]


"""
def DCT_block(block): #для одного канала
    N = block.shape[0] #shape - вернёт кортеж (N,N)
    result = np.zeros_like(block, dtype=np.float32)
    for k2 in range(N):
        for k1 in range(N):
            if k1 == 0:
                c1 = 1 / np.sqrt(N)
            elif k1 > 0:
                c1 = np.sqrt(2 / N)
            if k2 == 0:
                c2 = 1 / np.sqrt(N)
            elif k2 > 0:
                c2 = np.sqrt(2 / N)
            sum = 0
            for y in range(N):
                for x in range(N):
                    sum += block[y, x] * np.cos(np.pi * (x + 0.5) * k1/N) * np.cos(np.pi * (y + 0.5) * k2/N)
            result[k2, k1] = c1 * c2 * sum
    return result

#для одноканальной матрицы
def DCT_general(blocks):
    DCT_blocks = []

    for y, x, block in blocks:
        DCT = DCT_block(block)
        DCT_blocks.append((y, x, DCT))

    return DCT_blocks


def iDCT_block(block):
    N = block.shape[0]  # shape - вернёт кортеж (N,N)
    result = np.zeros_like(block, dtype=np.float32)

    for y in range(N):
        for x in range(N):
            sum = 0.0
            for k2 in range(N):
                for k1 in range(N):
                    if k1 == 0:
                        c1 = 1 / np.sqrt(N)
                    elif k1 > 0:
                        c1 = np.sqrt(2 / N)
                    if k2 == 0:
                        c2 = 1 / np.sqrt(N)
                    elif k2 > 0:
                        c2 = np.sqrt(2 / N)
                    sum += c1 * c2 * block[k2, k1] * np.cos(np.pi * (x + 0.5) * k1/N) * np.cos(np.pi * (2*y + 1) * k2/(2*N))

            result[y, x] =  sum
    return result

#для одноканальной матрицы
def iDCT_general(blocks):
    iDCT_blocks = []
    for block in blocks:  # block — это уже 2D DCT-матрица
        iDCT = iDCT_block(block)
        iDCT = np.clip(np.round(iDCT), 0, 255).astype(np.uint8) #округляем
        iDCT_blocks.append((iDCT))
    return iDCT_blocks

"""
"""
#для 3х канальной матрицы
def DCT_general(blocks):
    DCT_blocks = []
    for y, x, block in blocks:

        # Y канал из блока
        Y_block = block[:, :, 0] #[y,x,канал]
        DCT_block_Y = DCT_block(Y_block)

        # Cb канал из блока
        Cb_block = block[:, :, 1]
        DCT_block_Cb = DCT_block(Cb_block)

        # Cr канал из блока
        Cr_block = block[:, :, 2]
        DCT_block_Cr = DCT_block(Cr_block)

        # Собираем 3D-массив обратно
        DCT_block_all = np.stack((DCT_block_Y, DCT_block_Cb, DCT_block_Cr), axis=2)
        DCT_blocks.append((y, x, DCT_block_all))
    return DCT_blocks
"""


"""
#для 3х канальной матрицы
def iDCT_general(blocks):
    iDCT_blocks = []
    for y, x, block in blocks:  # block — это уже 2D DCT-матрица
        Y_block = iDCT_block(block[:, :, 0])
        Cb_block = iDCT_block(block[:, :, 1])
        Cr_block = iDCT_block(block[:, :, 2])

        # Собрать обратно в 3D-блок
        restored_block = np.stack((Y_block, Cb_block, Cr_block), axis=2)  # (N, N, 3)
        iDCT_blocks.append((y, x, restored_block))
    return iDCT_blocks
"""


