import numpy as np
from PIL import Image
import os
from bitarray import bitarray
import pickle
from YCbCr import RGB_to_YCbCr, YCbCr_to_RGB,Img,downsampling,to_blocks,bits_to_bytes,bytes_to_bits
from DCT import DCT_block, DCT_general,iDCT_block,iDCT_general
from quantization import quantize, dequantize,scale_quant_matrix
from ZigZag import ZigZag,iZigZag
from AC_DC import DC,AC_DC_combine,DC_coding,iDC_coding,AC_coef_blocks,AC_coding,RLE_AC,iRLE_AC,iAC_coding
from huff_coding import DC_huff_codes,AC_huff_codes
from huffman_tables import DC_luminance_table, DC_chrominance_table, AC_chrominance_table, AC_luminance_table
from visualize import visualize_stage

N=8

def compress(image_path,qual,outfile="output.bin"):
    pixels = Img(image_path)

    #pixels = pixels[:50, :50, :]  # Маленькое изображение
    height, width = pixels.shape[:2]
    #print(f"pixels RGB:", pixels[:5, :5])

    # переводим из rgb в YCbCr
    ycbcr = RGB_to_YCbCr(pixels)
   # print(f"YCbCr: ", ycbcr[:5, :5])

    Y_downsampled, Cb_downsampled, Cr_downsampled, Cb_new_h, Cb_new_w, Cr_new_h, Cr_new_w = downsampling(ycbcr, N)

    """
    if qual!=100:
        Y_downsampled, Cb_downsampled, Cr_downsampled,Cb_new_h, Cb_new_w,Cr_new_h, Cr_new_w = downsampling(ycbcr, N)
    else:
        Y_downsampled = ycbcr[:, :, 0]
        Cb_downsampled = ycbcr[:, :, 1]
        Cr_downsampled = ycbcr[:, :, 2]
        Cb_new_h, Cb_new_w, Cr_new_h, Cr_new_w = height, width,height, width
    """

    print(f"Y downsampling:", Y_downsampled[:15, :15])
    print(f"size:", len(Y_downsampled))
    print(f"Cb downsampling:", Cb_downsampled[:15, :15])
    print(f"size:", len(Cb_downsampled))
    print(f"Cr downsampling:", Cr_downsampled[:15, :15])
    print(f"size:", len(Cr_downsampled))


    # применяем dct
    Y_blocks, Y_padded_h, Y_padded_w = to_blocks(Y_downsampled, N)
    #print(f"Y поделился на блоки: ", Y_blocks)
    print(f"size Y_blocks", len(Y_blocks))
    Y_DCTed = DCT_general(Y_blocks)
    print(f"Y after DCT:", Y_DCTed[:2])
    visualize_stage(Y_DCTed, "after_dequantize", 'Y',"compress_Y_after_deDCT",False)


    Cb_blocks,Cb_padded_h, Cb_padded_w = to_blocks(Cb_downsampled, N)
   # print(f"Cb поделился на блоки: ", Cb_blocks)
    print(f"size Cb_blocks", len(Cb_blocks))
    Cb_DCTed = DCT_general(Cb_blocks)
    print(f"Cb after DCT:", Cb_DCTed[:2])
    visualize_stage(Cb_DCTed, "after_dequantize", 'Cb',"compress_Cb_after_dequant.png",True)


    Cr_blocks,Cr_padded_h, Cr_padded_w = to_blocks(Cr_downsampled, N)
   # print(f"Cr поделился на блоки: ", Cr_blocks)
    print(f"size Cr_blocks", len(Cr_blocks))
    Cr_DCTed = DCT_general(Cr_blocks)
    print(f"Cr after DCT:", Cr_DCTed[:2])
    visualize_stage(Cr_DCTed, "after_dequantize", 'Cr',"compress_Cr_after_dequant.png",True)


    Q_Y = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    Q_C = np.array([
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99]
    ])


    Q_Y_scaled = scale_quant_matrix(Q_Y, qual)
    Q_C_scaled = scale_quant_matrix(Q_C, qual)

    """
    qualities = [1, 20, 40, 60, 80, 100]
    for q in qualities:
        Q_Y_scaled = scale_quant_matrix(Q_Y, q)
        Q_C_scaled = scale_quant_matrix(Q_C, q)
        print("q, Q_Y_scaled",q,Q_Y_scaled)
        print("q, Q_C_scaled",q, Q_C_scaled)

    """

    # квантование матрицы dct
    Y_Q_DCT = [(y, x, quantize(block, Q_Y_scaled)) for (y, x, block) in Y_DCTed]
    print(f"квантование Y DCT:", Y_Q_DCT[:2])

    Cb_Q_DCT = [(y, x, quantize(block, Q_C_scaled)) for (y, x, block) in Cb_DCTed]
    print(f"квантование Cb DCT:", Cb_Q_DCT[:2])

    Cr_Q_DCT = [(y, x, quantize(block, Q_C_scaled)) for (y, x, block) in Cr_DCTed]
    print(f"квантование Cr DCT:", Cr_Q_DCT[:2])


    # зиг заг обход матрицы
    Y_zigzag = []
    Cb_zigzag = []
    Cr_zigzag = []
    for (y, x, block) in Y_Q_DCT:
        Y_z = ZigZag(block, N)
        Y_zigzag.append((y, x, Y_z))

    for (y, x, block) in Cb_Q_DCT:
        Cb_z = ZigZag(block, N)
        Cb_zigzag.append((y, x, Cb_z))

    for (y, x, block) in Cr_Q_DCT:
        Cr_z = ZigZag(block, N)
        Cr_zigzag.append((y, x, Cr_z))
    
    print(f"size Y_zigzag", len(Y_zigzag))
    print(f"size Cb_zigzag", len(Cb_zigzag))
    print(f"size Cr_zigzag", len(Cr_zigzag))

    # Разностное кодирование dc
    Y_DC = DC(Y_zigzag)
    #print(f"\nРазности DC Y:", Y_DC[:5])
    Cb_DC = DC(Cb_zigzag)
    #print(f"Разности DC Y:", Cb_DC[:5])
    Cr_DC = DC(Cr_zigzag)
    #print(f"Разности DC Y:", Cr_DC[:5])

    # Переменное кодирование разностей DC коэффициентов
    Y_DC_coded = DC_coding(Y_DC)
    print(f"\nПеременное кодирование DC Y:", Y_DC_coded[:2])
    Cb_DC_coded = DC_coding(Cb_DC)
    print(f"Переменное кодирование DC Cb:", Cb_DC_coded[:2])
    Cr_DC_coded = DC_coding(Cr_DC)
    print(f"Переменное кодирование DC Cr:", Cr_DC_coded[:2])

    # Переменное кодирование AC коэффициентов
    Y_AC_blocks = AC_coef_blocks(Y_zigzag)
    print(f"\nAC коэффициенты для Y:", Y_AC_blocks[:2])
    #print(f"size:", len(Y_AC_blocks))
    Y_AC_coded = AC_coding(Y_AC_blocks)
    print(f"Переменное кодирование AC Y:", Y_AC_coded[:2])
    #print(f"size Y_AC_coded:", len(Y_AC_coded))

    Cb_AC_blocks = AC_coef_blocks(Cb_zigzag)
    print(f"\nAC коэффициенты для Cb:", Cb_AC_blocks[:2])
    Cb_AC_coded = AC_coding(Cb_AC_blocks)
    print(f"Переменное кодирование AC Cb:", Cb_AC_coded[:2])
    print(f"size Cb_AC_coded:", len(Cb_AC_coded))

    Cr_AC_blocks = AC_coef_blocks(Cr_zigzag)
    print(f"\nAC коэффициенты для Cr:", Cr_AC_blocks[:2])
    Cr_AC_coded = AC_coding(Cr_AC_blocks)
    print(f"Переменное кодирование AC Cr:", Cr_AC_coded[:2])
    print(f"size Cr_AC_coded:", len(Cr_AC_coded))


    # RLE кодирование AC
    Y_RLE = RLE_AC(Y_AC_coded)
    print(f"\nRLE кодирование AC Y:", Y_RLE)
    print(f"size Y_RLE", len(Y_RLE))
    Cb_RLE = RLE_AC(Cb_AC_coded)
    print(f"RLE кодирование AC Cb:", Cb_RLE)
    print(f"size Cb_RLE", len(Cb_RLE))
    Cr_RLE = RLE_AC(Cr_AC_coded)
    print(f"RLE кодирование AC Cr:", Cr_RLE)
    print(f"size Cr_RLE", len(Cr_RLE))

    # Коды Хаффмана по таблице для DC
    Y_DC_huf_codes = DC_huff_codes('Y', Y_DC_coded)
    #print(f"\nHuffman coded DC Y:", Y_DC_huf_codes[:10])
    print(f"size Y_DC_huf_codes", len(Y_DC_huf_codes))
    Cb_DC_huf_codes = DC_huff_codes('Cb', Cb_DC_coded)
    #print(f"Huffman coded DC Cb:", Cb_DC_huf_codes[:10])
    print(f"size Cb_DC_huf_codes", len(Cb_DC_huf_codes))
    Cr_DC_huf_codes = DC_huff_codes('Cr', Cr_DC_coded)
    #print(f"Huffman coded DC Cr:", Cr_DC_huf_codes[:10])
    print(f"size Cr_DC_huf_codes", len(Cr_DC_huf_codes))

    # Коды Хаффмана по таблице для AC
    Y_AC_huf_codes = AC_huff_codes(Y_RLE,AC_luminance_table)
    print(f"\nHuffman coded AC Y:", Y_AC_huf_codes)
    print(f"size Y_AC_huf_codes", len(Y_AC_huf_codes))
    Cb_AC_huf_codes = AC_huff_codes(Cb_RLE,AC_chrominance_table)
    print(f"\nHuffman coded AC Cb:", Cb_AC_huf_codes)
    print(f"size Cb_AC_huf_codes", len(Cb_AC_huf_codes))
    Cr_AC_huf_codes = AC_huff_codes(Cr_RLE,AC_chrominance_table)
    print(f"\nHuffman coded AC Cr:", Cr_AC_huf_codes)
    print(f"size Cr_AC_huf_codes", len(Cr_AC_huf_codes))

    Y_channel = Y_DC_huf_codes + Y_AC_huf_codes
    Cb_channel = Cb_DC_huf_codes + Cb_AC_huf_codes
    Cr_channel = Cr_DC_huf_codes + Cr_AC_huf_codes

    Y_bits = ''.join(Y_channel)
    Cb_bits = ''.join(Cb_channel)
    Cr_bits = ''.join(Cr_channel)

    # Переводим в байты
    Y_bitstream, Y_padding = bits_to_bytes(Y_bits)
    Cb_bitstream, Cb_padding = bits_to_bytes(Cb_bits)
    Cr_bitstream, Cr_padding = bits_to_bytes(Cr_bits)
    print(f"size Y_bitstream", len(Y_bitstream))
    print(f"size Cb_bitstream", len(Cb_bitstream))
    print(f"size Cr_bitstream", len(Cr_bitstream))


    Y_dc_len = sum(len(code) for code in Y_DC_huf_codes)
    Cb_dc_len = sum(len(code) for code in Cb_DC_huf_codes)
    Cr_dc_len = sum(len(code) for code in Cr_DC_huf_codes)
    print(f"size Y_dc_len", Y_dc_len)
    print(f"size Cb_dc_len", Cb_dc_len)
    print(f"size Cr_dc_len", Cr_dc_len)

    Y_ac_len = sum(len(code) for code in Y_AC_huf_codes)
    Cb_ac_len = sum(len(code) for code in Cb_AC_huf_codes)
    Cr_ac_len = sum(len(code) for code in Cr_AC_huf_codes)
    print(f"size Y_ac_len", Y_ac_len)
    print(f"size Cb_ac_len", Cb_ac_len)
    print(f"size Cr_ac_len", Cr_ac_len)

    # Метаданные
    output = {
        "size": {
            "Y": {
                "height": height,
                "width": width,
            },
            "Cb": {
                "height": Cb_new_h,
                "width": Cb_new_w,
            },
            "Cr": {
                "height": Cr_new_h,
                "width": Cr_new_w,
            }
        },
        "padded_sizes": {
            "Y": [Y_padded_h, Y_padded_w],
            "Cb": [Cb_padded_h, Cb_padded_w],
            "Cr": [Cr_padded_h, Cr_padded_w]
        },
        "block_size": 8,
        "color_space": "YCbCr",
        "quant_table_Y": Q_Y_scaled,
        "quant_table_C": Q_C_scaled,
        "huffman_tables": {
            "DC_Y": DC_luminance_table,
            "AC_Y": AC_luminance_table,
            "DC_C": DC_chrominance_table,
            "AC_C": AC_chrominance_table
        },
        "compressed_data": {
            "Y": {
                "bitstream": Y_bitstream,
                "padding_bits": Y_padding,
                "DC_length": Y_dc_len
            },
            "Cb": {
                "bitstream": Cb_bitstream,
                "padding_bits": Cb_padding,
                "DC_length": Cb_dc_len
            },
            "Cr": {
                "bitstream": Cr_bitstream,
                "padding_bits": Cr_padding,
                "DC_length": Cr_dc_len
            }
        }
    }

    # Запись в файл
    with open(outfile, "wb") as f:
        pickle.dump(output, f)
    

