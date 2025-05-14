import numpy as np

def DC(channel_zig):
    dc_coeffs=[]

    #список dc всех блоков
    for (y, x, zig_list) in channel_zig:
        dc=zig_list[0] #dc коэф - сред яркость блока
        dc_coeffs.append(dc)

    diffs = [dc_coeffs[0]]
    #список разностей
    for i in range(1, len(dc_coeffs)):
        diffs.append(dc_coeffs[i] - dc_coeffs[i - 1])

    return diffs


def iDC(diffs):
    restored = [diffs[0]]
    for i in range(1, len(diffs)):
        restored.append(restored[-1] + diffs[i])
    return restored


def AC_DC_combine(diffs, ac_blocks):
    dc_list = iDC(diffs)
    channel_zig=[]
    i=0
    for (ACmas) in ac_blocks:
        channel_zig.append(([dc_list[i]]+ACmas)) #[] нужны чтобы dc_list[i] был списком а не элементом
        i+=1
    return channel_zig


def DC_coding(list):
    coded=[]
    for a in list:
        if a == 0:
            size = 0
            value = ''
        else:
            size = len(bin(abs(a))) - 2 #-2 потому что 0b в начале#value
            if a>0:
                value = bin(a)[2:]
            else:
                # инвертируем строку битов от abs(a)
                bits = bin(abs(a))[2:].zfill(size)  # дополняем до size
                value = ''.join('1' if b == '0' else '0' for b in bits)
        coded.append((size, value))

    return coded


def iDC_coding(coded):
    list=[]
    for (size, value) in coded:
        if size == 0 and value == '':
            a = 0
        else:
            if value[0] == '1':
                a = int(value, 2)
            else:
                inverted = ''.join('1' if b == '0' else '0' for b in value)
                a = -int(inverted, 2)
        list.append(a)
    return list


def AC_coef_blocks(channel_zig):
    AC_blocks=[]
    # список ac для каждого блока отдельно
    for (y, x, zig_list) in channel_zig:
        ACmas=zig_list[1:]
        AC_blocks.append((y, x, ACmas))

    return AC_blocks


def AC_coding(AC_blocks):
    AC_coded_blocks=[]
    for (y, x, ACmas) in AC_blocks:
        coded = []
        for a in ACmas:
            if a == 0:
                size = 0
                value = ''
            else:
                size = len(bin(abs(a))) - 2  # -2 потому что 0b в начале#value
                if a > 0:
                    value = bin(a)[2:]
                else:
                    # инвертируем строку битов от abs(a)
                    bits = bin(abs(a))[2:].zfill(size)  # дополняем до size
                    value = ''.join('1' if b == '0' else '0' for b in bits)
            coded.append((size, value))
        AC_coded_blocks.append((y, x, coded))
    return AC_coded_blocks


def iAC_coding(AC_coded_blocks):
    AC_blocks=[]
    for (coded) in AC_coded_blocks:
        ACmas=[]
        for (size,value) in coded:
            if size == 0 and value == '':
                a = 0
            else:
                if value[0] == '1':
                    a = int(value, 2)
                else:
                    inverted = ''.join('1' if b == '0' else '0' for b in value)
                    a = -int(inverted, 2)
            ACmas.append(a)
        AC_blocks.append((ACmas))

    return AC_blocks


def RLE_AC(blocks):

    all_rle = []
    for x,y, block in blocks:
        rle_block = []
        run = 0

        for size, value in block:
            if size == 0 and value == '':
                run += 1
                if run == 16:
                    rle_block.append((15, 0, ''))  # ZRL
                    run = 0
            else:
                rle_block.append((run, size, value))
                run = 0

        if run > 0:
            rle_block.append((0, 0, ''))  # EOB

        all_rle.append(rle_block)
    return all_rle


def iRLE_AC(rle_blocks):

    all_blocks = []

    for rle_block in rle_blocks:
        block = []
        count = 0

        for run, size, value in rle_block:
            if run == 0 and size == 0:  # EOB
                break
            if run == 15 and size == 0:  # ZRL
                block.extend([(0, '')] * 16)
                count += 16
            else:
                block.extend([(0, '')] * run)
                block.append((size, value))
                count += run + 1

        while count < 63:
            block.append((0, ''))
            count += 1

        all_blocks.append(block)
    return all_blocks


"""
def split_ac_rle_to_blocks(ac_rle_flat, N):
    block_size = N * N - 1  # 63 для 8x8
    blocks = []
    current = []
    count = 0

    for item in ac_rle_flat:
        current.append(item)
        if item == (0, 0, ''):
            blocks.append(current)
            current = []
            count = 0
        else:
            count += 1
            if count >= block_size:
                blocks.append(current)
                current = []
                count = 0

    if current:
        blocks.append(current)
    return blocks


def RLE_AC(AC_coded):
    RLE_all_blocks = []
    for (y,x,ACmas) in AC_coded:
        RLEmas=[]
        k=0
        for size, value in ACmas:
            if size == 0:
                k += 1
            else:
                while k > 15:
                    RLEmas.append((15, 0, ''))  # ZRL
                    k -= 16
                RLEmas.append((k, size, value))
                k = 0  # сбрасываем после ненулевого
        if k > 0:
            # если после последнего ненулевого значения остались нули,
            # они должны быть заменены на (0, 0)
            RLEmas.append((0, 0, ''))
        RLE_all_blocks.append((y, x, RLEmas))

    return RLE_all_blocks


def iRLE_AC(RLE_all_blocks,N,padded_h,padded_w):
    AC_coded=[]
    ex_size=(padded_h//N)*(padded_w//N)
    for RLEmas in RLE_all_blocks:
        ACmas=[]

        for(k, size, value) in RLEmas:

            if k==0 and size==0 and value=='': #если встретили маркер нулей в конце
                kol=N*N-len(ACmas)
                for i in range(kol):
                    ACmas.append((0, ''))
            else:
                for _ in range(k):
                    ACmas.append((0,''))
                ACmas.append((size, value))

        AC_coded.append((ACmas))

    print("AC_coded, size:", len(AC_coded),ex_size)

    if len(AC_coded)<ex_size:
        extra = N*N-1
        for i in range(ex_size - len(AC_coded)):
            ACmas = []
            for j in range(extra):
                ACmas.append((0, ''))
            AC_coded.append((ACmas))

    elif len(AC_coded)>ex_size:
        AC_coded = AC_coded[:ex_size]  # просто обрезаем лишние

    return AC_coded



def split_ac_rle_to_blocks(ac_rle_flat, N):
    block_size = N*N - 1
    blocks = []
    current = []
    coeff_count = 0  # Счетчик реальных коэффициентов в текущем блоке

    for item in ac_rle_flat:
        current.append(item)

        # Обновляем счетчик коэффициентов
        if item == (0, 0, ''):  # EOB
            # Заполняем оставшиеся коэффициенты нулями
            remaining = block_size - coeff_count
            if remaining > 0:
                current.extend([(0, 0, '')] * remaining)
            coeff_count = block_size  # Помечаем блок как полный
        else:
            # Для обычных RLE элементов: run нулей + 1 коэффициент
            run = item[0]
            coeff_count += (run+1)

        # Если блок заполнен (63 коэффициента)
        if coeff_count >= block_size:
            blocks.append(current)
            current = []
            coeff_count = 0

    # Добавляем последний неполный блок, если есть
    if current:
        # Заполняем нулями до конца блока
        remaining = block_size - coeff_count
        if remaining > 0:
            current.extend([(0, 0, '')] * remaining)
        blocks.append(current)

    print(f"Разделено на {len(blocks)} блоков (ожидается {len(ac_rle_flat) // 63})")
    return blocks
    
def iRLE_AC(RLE_all_blocks,N):
    AC_coded=[]

    for RLEmas in RLE_all_blocks:
        ACmas=[]

        for(k, size, value) in RLEmas:

            if k==0 and size==0 and value=='': #если встретили маркер нулей в конце
                kol=N*N-len(ACmas)
                for i in range(kol):
                    ACmas.append((0, ''))
            else:
                for _ in range(k):
                    ACmas.append((0,''))
                ACmas.append((size, value))

        AC_coded.append((ACmas))
    return AC_coded

def iRLE_AC(RLE_all_blocks,N):
    AC_coded=[]

    for RLEmas in RLE_all_blocks:
        ACmas=[]

        for(k, size, value) in RLEmas:

            if k==0 and size==0 and value=='': #если встретили маркер нулей в конце
                kol=N*N-len(ACmas)
                for i in range(kol):
                    ACmas.append((0, ''))
            else:
                for _ in range(k):
                    ACmas.append((0,''))
                ACmas.append((size, value))

        AC_coded.append((ACmas))
    return AC_coded

"""