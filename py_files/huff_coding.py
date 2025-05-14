import numpy as np
from huffman_tables import DC_luminance_table,DC_chrominance_table,AC_chrominance_table,AC_luminance_table

def DC_huff_codes(channel,channel_list):
    codes = []
    for el in channel_list:
        category = el[0]
        if channel=='Y':
            codeword = DC_luminance_table[category]
        else:
            codeword = DC_chrominance_table[category]
        code = codeword + el[1]
        codes.append(code)
    return codes


def DC_huff_decodes(bitstream, channel, DC_luminance_table, DC_chrominance_table):
        if channel == 'Y':
            table = DC_luminance_table
        else:
            table=DC_chrominance_table

        # Делаем обратную таблицу: код - категория
        reverse_table = {}
        for category in table:
            code = table[category]
            reverse_table[code] = category

        decoded = []
        i = 0  # Текущая позиция в битовой строке
        while i < len(bitstream):
            matched = False  # Флаг, нашли ли мы код

            # Пробуем взять кусочки разной длины, чтобы найти подходящий код
            for length in range(2, 12):
                code_candidate = bitstream[i:i + length]

                if code_candidate in reverse_table:
                    category = reverse_table[code_candidate]  # Нашли категорию
                    i += length
                    size = int(category)

                    if size > 0:
                        amplitude_bits = bitstream[i:i + size]
                        i += size
                    else:
                        amplitude_bits = ''

                    decoded.append((category, amplitude_bits))
                    matched = True
                    break

            if not matched:
                print("Ошибка: не удалось найти подходящий код начиная с позиции", i)
                break
        return decoded


def AC_huff_codes(rle_blocks, huffman_table):

    print('HUFFMAN')
    result = []
    for rle_block in rle_blocks:
        bitstream = ""
        for run, size, value in rle_block:
            key = f"{run}/{size}"
            huff_code = huffman_table.get(key, '')

            print(huff_code,end=', ')
            bitstream += huff_code + value  # value может быть '', и это нормально
        result.append(bitstream)
    return result


def AC_huff_decodes(bitstream, huffman_table, max_coeffs=63):

    reverse_table = {v: k for k, v in huffman_table.items()}
    result = []
    pos = 0
    total_bits = len(bitstream)

    while pos < total_bits:
        block = []
        count = 0

        while count < max_coeffs:
            if pos >= total_bits:
                break

            # Поиск следующего кода Хаффмана
            found = False
            for length in range(1, 17):
                if pos + length > total_bits:
                    break
                code = bitstream[pos:pos+length]
                if code in reverse_table:
                    run_size = reverse_table[code]
                    run, size = map(int, run_size.split('/'))
                    pos += length
                    found = True
                    break

            if not found:
                if pos >= total_bits and not block:
                    break  # Нормальное завершение, если нет данных
                raise ValueError(f"Invalid Huffman code at position {pos}")

            # Обработка специальных кодов
            if run == 0 and size == 0:  # EOB
                block.append((0, 0, ''))
                break
            elif run == 15 and size == 0:  # ZRL
                block.append((15, 0, ''))
                count += 16
            else:
                if pos + size > total_bits:
                    raise ValueError("Unexpected end of stream during value bits")
                value_bits = bitstream[pos:pos+size]
                pos += size
                block.append((run, size, value_bits))
                count += run + 1

        # Добавляем блок только если он не пустой или это не конец данных
        if block or pos < total_bits:
            result.append(block)
            print('HUFFMAN',block)

    return result

"""
def AC_huff_codes(channel,channel_list):
    codes = []
    for (y,x,block) in channel_list:
        for el in block:
            run = el[0]
            size = el[1]
            #amplitude_bits = el[2] if len(el) > 2 else ''
            category = f"{run}/{size}"
            if channel == 'Y':
                codeword = AC_luminance_table[category]
            else:
                codeword = AC_chrominance_table[category]
            code = codeword + el[2]
            codes.append(code)
    return codes


def AC_huff_decodes(bitstream, channel, AC_luminance_table, AC_chrominance_table):
    if channel == 'Y':
        table = AC_luminance_table
    else:
        table = AC_chrominance_table

    reverse_table = {}
    for key, codeword in table.items():
        reverse_table[codeword] = key  # key это строка вида 'run/size'

    decoded = []
    i = 0
    k=0
    while i < len(bitstream):
        matched = False
        # Пробуем найти код в таблице по префиксу бит
        for length in range(1, 17):
            if i + length > len(bitstream):
                break  # Не хватает бит для кода

            code_candidate = bitstream[i:i + length]

            if code_candidate in reverse_table:
                run_size = reverse_table[code_candidate]
                run, size = map(int, run_size.split('/'))
                i += length

                if run == 0 and size == 0:
                    # Это EOB (End of Block) — специальный код, нет дополнительных бит
                    amplitude_bits = ''
                else:
                    # Считываем амплитудные биты, их количество = size
                    amplitude_bits = bitstream[i:i + size]
                    i += size

                decoded.append((run, size, amplitude_bits))
                matched = True
                if (k <= 10):
                    print(code_candidate)
                    k += 1
                break

        if not matched:
            print(f"Не найден AC код по битам начиная с позиции {i}")
            break
    return decoded
    """
