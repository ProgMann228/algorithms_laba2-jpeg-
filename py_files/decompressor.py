import numpy as np
from PIL import Image
import os
from bitarray import bitarray
import pickle
from YCbCr import RGB_to_YCbCr, YCbCr_to_RGB,Img,downsampling,to_blocks,bits_to_bytes,bytes_to_bits,from_blocks,upsampling
from DCT import DCT_block, DCT_general,iDCT_block,iDCT_general
from quantization import quantize, dequantize,scale_quant_matrix
from ZigZag import ZigZag,iZigZag
from AC_DC import DC,AC_DC_combine,DC_coding,iDC_coding,AC_coef_blocks,AC_coding,RLE_AC,iRLE_AC,iAC_coding
from huff_coding import DC_huff_codes,DC_huff_decodes,AC_huff_codes,AC_huff_decodes
from huffman_tables import DC_luminance_table, DC_chrominance_table, AC_chrominance_table, AC_luminance_table
from visualize import visualize_stage


# ДЕКОМПРЕССИЯ

def decompress(restoredfile,outfile="output.bin"):

    with open(outfile, "rb") as f:
        output = pickle.load(f)

    Y_bitstream = output["compressed_data"]["Y"]["bitstream"]
    Y_padding = output["compressed_data"]["Y"]["padding_bits"]

    Cb_bitstream = output["compressed_data"]["Cb"]["bitstream"]
    Cb_padding = output["compressed_data"]["Cb"]["padding_bits"]

    Cr_bitstream = output["compressed_data"]["Cr"]["bitstream"]
    Cr_padding = output["compressed_data"]["Cr"]["padding_bits"]

    print(f"size Y_bitstream", len(Y_bitstream))
    print(f"size Cb_bitstream", len(Cb_bitstream))
    print(f"size Cr_bitstream", len(Cr_bitstream))

    Y_height = output["size"]["Y"]["height"]
    Y_width = output["size"]["Y"]["width"]
    Cb_height = output["size"]["Cb"]["height"]
    Cb_width = output["size"]["Cb"]["width"]
    Cr_height = output["size"]["Cr"]["height"]
    Cr_width = output["size"]["Cr"]["width"]

    print(f"Y_height",Y_height)
    print(f"Y_width",Y_width)
    print(f"Cb_height",Cb_height)
    print(f"Cb_width",Cb_width)
    print(f"Cr_height",Cr_height)
    print(f"Cr_width",Cr_width)

    Y_padded_h, Y_padded_w = output["padded_sizes"]["Y"]
    Cb_padded_h, Cb_padded_w = output["padded_sizes"]["Cb"]
    Cr_padded_h, Cr_padded_w = output["padded_sizes"]["Cr"]
    print(f"padded h,w",Y_padded_h, Y_padded_w)
    print(f"padded h,w",Cb_padded_h, Cb_padded_w)
    print(f"padded h,w",Cr_padded_h, Cr_padded_w)

    N = output["block_size"]
    color_space = output["color_space"]
    Q_Y = output["quant_table_Y"]
    Q_C = output["quant_table_C"]
    DC_Y = output["huffman_tables"]["DC_Y"]
    AC_Y = output["huffman_tables"]["AC_Y"]
    DC_C = output["huffman_tables"]["DC_C"]
    AC_C = output["huffman_tables"]["AC_C"]

    Y_bits = bytes_to_bits(Y_bitstream, Y_padding)
    Cb_bits = bytes_to_bits(Cb_bitstream, Cb_padding)
    Cr_bits = bytes_to_bits(Cr_bitstream, Cr_padding)

    print(f"size Y_dc_len", output["compressed_data"]["Y"]["DC_length"])
    print(f"size Cb_dc_len", output["compressed_data"]["Cb"]["DC_length"])
    print(f"size Cr_dc_len", output["compressed_data"]["Cr"]["DC_length"])

# Отдельно получаем DC и AC битстримы
    Y_dc_bits = Y_bits[:output["compressed_data"]["Y"]["DC_length"]]
    Y_ac_bits = Y_bits[output["compressed_data"]["Y"]["DC_length"]:]
    print(f"size получаем DC битстрим", len(Y_dc_bits))
    print(f"size получаем AC битстрим", len(Y_ac_bits))

    Cb_dc_bits = Cb_bits[:output["compressed_data"]["Cb"]["DC_length"]]
    Cb_ac_bits = Cb_bits[output["compressed_data"]["Cb"]["DC_length"]:]
    print(f"size получаем DC битстрим", len(Cb_dc_bits))
    print(f"size получаем AC битстрим", len(Cb_ac_bits))

    Cr_dc_bits = Cr_bits[:output["compressed_data"]["Cr"]["DC_length"]]
    Cr_ac_bits = Cr_bits[output["compressed_data"]["Cr"]["DC_length"]:]
    print(f"size получаем DC битстрим", len(Cr_dc_bits))
    print(f"size получаем AC битстрим", len(Cr_ac_bits))

# Декодируем
    Y_blocks_expected=(Y_padded_h // N) * (Y_padded_w // N)
    Y_dc_decoded = DC_huff_decodes(Y_dc_bits, 'Y', DC_Y, DC_C)
    Y_ac_rle = AC_huff_decodes(Y_ac_bits,AC_Y)
    print(f"\nY DC Huffman decodes",len(Y_dc_decoded))
    print(f"Y AC Huffman decodes", len(Y_ac_rle))
    print(f"Y AC RLE", Y_ac_rle)

    Cb_blocks_expected=(Cb_padded_h // N) * (Cb_padded_w // N)
    Cb_dc_decoded = DC_huff_decodes(Cb_dc_bits, 'Cb', DC_Y, DC_C)
    Cb_ac_rle = AC_huff_decodes(Cb_ac_bits, AC_C)
    print(f"\nCb DC Huffman decodes", len(Cb_dc_decoded))
    print(f"Cb AC Huffman decodes", len(Cb_ac_rle))
    print(f"Cb AC RLE", Cb_ac_rle)

    Cr_blocks_expected=(Cr_padded_h // N) * (Cr_padded_w // N)
    Cr_dc_decoded = DC_huff_decodes(Cr_dc_bits, 'Cr', DC_Y, DC_C)
    Cr_ac_rle = AC_huff_decodes(Cr_ac_bits, AC_C)
    print(f"\nCr DC Huffman decodes", len(Cr_dc_decoded))
    print(f"Cr AC Huffman decodes", len(Cr_ac_rle))
    print(f"Cr AC RLE", Cr_ac_rle)
    """
#разбиваем сплошную rle строку на блоки, как это было во время коспрессии:
# RLE кодирование AC Cr: [(0, 0, [(0, 0, '')]), (0, 8, [(0, 5, '11100')

    Y_ac_rle_blocks = split_ac_rle_to_blocks(Y_ac_rle, N)
    print(f"\nРазбили Y AC RLE на блоки",Y_ac_rle_blocks[:2])

    Cb_ac_rle_blocks = split_ac_rle_to_blocks(Cb_ac_rle, N)
    print(f"Разбили Cb AC RLE на блоки", Cb_ac_rle_blocks[:2])

    Cr_ac_rle_blocks = split_ac_rle_to_blocks(Cr_ac_rle, N)
    print(f"Разбили Cr AC RLE на блоки", Cr_ac_rle_blocks[:2])
    """
# RLE декодирование AC
    Y_AC_coded = iRLE_AC(Y_ac_rle)
    #print(f"\nRLE декодирование AC Y:", Y_AC_coded[:-1])
    print(f"size Y RLE декодирование AC", len(Y_AC_coded))

    Cb_AC_coded = iRLE_AC(Cb_ac_rle)
    #print(f"RLE декодирование AC Cb:", Cb_AC_coded[:-1])
    print(f"size Cb RLE декодирование AC", len(Cb_AC_coded))

    Cr_AC_coded = iRLE_AC(Cr_ac_rle)
   # print(f"RLE декодирование AC Cr:", Cr_AC_coded[:-1])
    print(f"size Cr RLE декодирование AC", len(Cr_AC_coded))
    print(f"size Cr RLE block AC", len(Cr_AC_coded[len(Cr_AC_coded)-1]))


# Переменное декодирование AC коэффициентов
    Y_AC_blocks = iAC_coding(Y_AC_coded)
    #print(f"Y AC переменное декодирование", Y_AC_blocks[:-3])
    print(f"\nsize Переменное декодирование AC Y:", len(Y_AC_blocks))

    Cb_AC_blocks = iAC_coding(Cb_AC_coded)
    #print(f"Cb AC переменное декодирование", Cb_ac_rle[:-3])
    print(f"size Переменное декодирование AC Cb:", len(Cb_AC_blocks))

    Cr_AC_blocks = iAC_coding(Cr_AC_coded)
    #print(f"Cb AC переменное декодирование", Cb_ac_rle[:-3])
    print(f"size Переменное декодирование AC Cr:",len(Cr_AC_blocks))

# Переменное декодирование разностей DC коэффициентов
    Y_DC = iDC_coding(Y_dc_decoded)
    print(f"\nsize Переменное декодирование DC Y:", len(Y_DC))
    Cb_DC = iDC_coding(Cb_dc_decoded)
    print(f"\nsize Переменное декодирование DC Cb:", len(Cb_DC))
    Cr_DC = iDC_coding(Cr_dc_decoded)
    print(f"\nsize Переменное декодирование DC Cr:", len(Cr_DC))

# объединение AC и DC
    Y_zigzag = AC_DC_combine(Y_DC, Y_AC_blocks)
    print(f"size Y zigzag list:", len(Y_zigzag))
    Cb_zigzag = AC_DC_combine(Cb_DC, Cb_AC_blocks)
    print(f"size Cb zigzag list:", len(Cb_zigzag))
    Cr_zigzag = AC_DC_combine(Cr_DC, Cr_AC_blocks)
    print(f"size Cr zigzag list:", len(Cr_zigzag))

# Обратный зигзаг
    Y_Q_DCT = []
    Cb_Q_DCT = []
    Cr_Q_DCT = []
    for (block) in Y_zigzag:
        Y_dct = iZigZag(block, N)
        Y_Q_DCT.append((Y_dct))
   # print(f"после обратного зигзага вернулись к Y Q_DCT матрице:", Y_Q_DCT[:2])

    for (block) in Cb_zigzag:
        Cb_dct = iZigZag(block, N)
        Cb_Q_DCT.append((Cb_dct))
   # print(f"после обратного зигзага вернулись к Cb Q_DCT матрице:", Cb_Q_DCT[:2])

    for (block) in Cr_zigzag:
        Cr_dct = iZigZag(block, N)
        Cr_Q_DCT.append((Cr_dct))
   # print(f"после обратного зигзага вернулись к Cr Q_DCT матрице:", Cr_Q_DCT[:2])


    #Обратное квантование матрицы dct
    Y_DCTed = [dequantize(block.astype(np.float32), Q_Y.astype(np.float32)) for block in Y_Q_DCT]
    print(f"size деквантование Y DCT:", len(Y_DCTed))
    #print(f"деквантование Y DCT:", Y_DCTed[:2])
    visualize_stage(Y_DCTed, "after_dequantize", 'Y', "decomp_Y_after_dequant.png",False)

    Cb_DCTed = [dequantize(block.astype(np.float32), Q_C.astype(np.float32)) for block in Cb_Q_DCT]
    print(f"size деквантование Cb DCT:", len(Cb_DCTed))
    #print(f"деквантование Cb DCT:", Cb_DCTed[:2])
    visualize_stage(Cb_DCTed, "after_dequantize", 'Cb',"decomp_Cb_after_dequant.png",True)

    Cr_DCTed = [dequantize(block.astype(np.float32), Q_C.astype(np.float32)) for block in Cr_Q_DCT]
    print(f"size деквантование Cb DCT:", len(Cr_DCTed))
    #print(f"деквантование Cr DCT:", Cr_DCTed[:2])
    visualize_stage(Cr_DCTed, "after_dequantize", 'Cr',"decomp_Cr_after_dequant.png",True)

    #Обратное DCT
    Y_blocks = iDCT_general(Y_DCTed)
    print(f"size Y_blocks",len(Y_blocks))
    print(f"\nY блоки после обратного DCT:", Y_blocks[:2])
    Y_downsampled=from_blocks(Y_blocks,Y_height, Y_width,Y_padded_h,Y_padded_w, N)
   # print(f"\nвернулись к Y downsampling:", Y_downsampled[:2, :2])
    print(f"size Y_downsampled",len(Y_downsampled))

    Cb_blocks = iDCT_general(Cb_DCTed)
    print(f"size Cb_blocks",len(Cb_blocks))
    print(f"\nCb блоки после обратного DCT:", Cb_blocks[:2])
    Cb_downsampled = from_blocks(Cb_blocks, Cb_height, Cb_width, Cb_padded_h,Cb_padded_w,N)
    #print(f"\nвернулись к Cb downsampling:", Cb_downsampled[:2, :2])
    print(f"size Cb_downsampled", len(Cb_downsampled))

    Cr_blocks = iDCT_general(Cr_DCTed)
    print(f"size Cr_blocks",len(Cr_blocks))
    print(f"\nCr блоки после обратного DCT:", Cr_blocks[:2])
    Cr_downsampled = from_blocks(Cr_blocks, Cr_height, Cr_width,Cr_padded_h,Cr_padded_w, N)
    #print(f"\nвернулись к Cr downsampling:", Cr_downsampled[:2, :2])
    print(f"size Cr_downsampled", len(Cr_downsampled))

#Обратный даунсэмплинг

    ycbcr=upsampling(Y_downsampled, Cb_downsampled, Cr_downsampled)

    #print(f"\nВернулись к YCbCr",ycbcr[:2, :2])

# переводим из YCbCr в RGB
    height, width = ycbcr.shape[:2]
    rgb_image = np.empty((height, width, 3), dtype=np.uint8)
    ycbcr = np.clip(ycbcr, 0, 255)

    for i in range(height):
        for j in range(width):
            y, cb, cr = ycbcr[i, j]
            r, g, b = YCbCr_to_RGB(y, cb, cr)
            rgb_image[i, j] = (r, g, b)

    image = Image.fromarray(rgb_image, 'RGB')

    image.save(restoredfile)
    image.show()


