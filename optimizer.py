# from numba import jit, autojit
import argparse
import os
import math
import numpy as np
from utils import *
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree
import datetime
import warnings
import sys


def img2arr(image):

    # tobytes() will provide stream of bytes
    # np.fromString() converts these bytes to an 1D-array of strings
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)

    # image.size[0] will provide the height of image
    # image.size[0] will provide the width of image
    # print("Dimensions: ",image.size[0],"\t",image.size[1])

    # Finding the number of channels of the image. For RGB = 3.
    dep = (int) (im_arr.size / (image.size[0] * image.size[1]))

    # Converting the 1D-array to 2D-array 
    if dep > 1:
        im_arr = im_arr.reshape((image.size[1], image.size[0], dep))
    else:
        im_arr = im_arr.reshape((image.size[1], image.size[0]))

    return im_arr
        
# Loading the quantization table
# component can be luminence or chrominence.
def quantize(block, component):
    q = load_quantization_table(component)
    return (block / q).round().astype(np.int32)

# Getting the zigzag order of the image
# The points given be generated function, are then converted into 1D array
def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])

# @autojit
def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    # values are binary representations of array elements using SIZE bits
    values = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values


# @autojit
# Recreating the block from the zigzag order
def zigzag_to_block(zigzag):
    # assuming that the width and the height of the block are equal
    rows = cols = int(math.sqrt(len(zigzag)))

    if rows * cols != len(zigzag):
        raise ValueError("length of zigzag should be a perfect square")

    block = np.empty((rows, cols), np.int32)

    for i, point in enumerate(zigzag_points(rows, cols)):
        block[point] = zigzag[i]

    return block


def dequantize(block, component):
    q = load_quantization_table(component)
    return block * q


def main():

    warnings.filterwarnings("ignore")

    #To mark the start time 
    start = datetime.datetime.now()

    #TO accept Command Line Arguments we are creating an instance of ArgumentParser from argparse module
    parser = argparse.ArgumentParser()

    #input variable will contain the absolute address from the commandline arguments
    parser.add_argument("input", help="path to the input image directory")

    ## parser.add_argument("output", help="path to the output image")

    #parse_args is used to convert commandline arguments int o python object such as string,integer etc..
    args = parser.parse_args()
    fileDir = os.listdir(args.input)

    #creating an output directory which contains all the compressed images
    outputDir = args.input+"\compressed"
    os.mkdir(outputDir)

    ## Determining the length of the address and unused
    # tole = len(input_file)

    # creating a dataframe to store all the metrics
    df = pd.DataFrame(columns=['Image Name','PSNR', 'MSE', 'RMSE','Compression Ratio','Original Size','Compressed Size'], dtype=np.float32)

    for input_file in fileDir:

        block_index = -1

        block_index1 = -1

        input_file = os.path.join(args.input,input_file)

        print("**************************************************")

        #  "D:\ImageProcessing\image-optimizer\imgs\p1.jpeg"
        # Determine the extension of the image, the flag here is '.', it indicates the next part is extension
        poi = 0
        for i in input_file:
            if i != ".":
                poi += 1
            else:
                break
            
        # Determined the index of ".", extract it using slice operator
        exte = input_file[poi + 1:]
        print("exte : ", exte)

        # This is a lazy operation; this function identifies the file, but the file remains open 
        # and the actual image data is not read from the file until you try to process the data
        image = Image.open(input_file)

        # Stored the address by excluding .extension
        # Example "D:\ImageProcessing\image-optimizer\imgs\p1.jpeg" ==> "D:\ImageProcessing\image-optimizer\imgs\p1"
        input_file = input_file[:poi]

        imageName = input_file.split("\\")[-1]
        print('Executing ', imageName)

        # Obtained 2D-array of the image
        or_img = img2arr(image)
        print("original image shape : ", or_img.shape)

        # Converting it into ycbcr color space
        ycbcr = image.convert('YCbCr')

        # Converting into a numpy array
        npmat = np.array(ycbcr, dtype=np.uint8)

        # Rounding of the dimensions to factors of 8 because block size: 8x8
        rows, cols = npmat.shape[0], npmat.shape[1]
        orows, ocols = rows, cols
        print("old shape : ", orows, " * ", ocols)
        rows = int(rows / 8) * 8
        cols = int(cols / 8) * 8

        ## npmat.reshape((rows, cols, 3)) WRONG+
        npmat = npmat[0:rows, 0:cols, :]
        print("new shape : ", npmat.shape[0]," * ", npmat.shape[1])

        # block size: 8x8
        """
        if rows % 8 == cols % 8 == 0:
            blocks_count = rows // 8 * cols // 8
        else:
        	if rows % 8 != 0 and cols % 8 != 0:
        		blocks_count = int(rows / 8) * int(cols / 8)
        """

        # Determining the count of blocks in the image
        # print(rows / 8, cols / 8, int(rows / 8), int(cols / 8))


        blocks_count = int(rows / 8) * int(cols / 8)
        print("blocks_count : ", blocks_count)

        # dc is the top-left cell of the block, ac are all the other cells

        # Total number of dc components in the image
        dc = np.empty((blocks_count, 3), dtype=np.int32)

        # Total number of ac components in the image
        ac = np.empty((blocks_count, 63, 3), dtype=np.int32)

        print("rows", rows, " cols ", cols)

        #Iterating every piece of block
        for i in range(0, rows, 8):
            for j in range(0, cols, 8):
                try:
                    block_index += 1
                except NameError:
                    block_index = 0

                for k in range(3):
                    # split 8x8 block and center the data range on zero
                    # [0, 255] --> [-128, 127]
                    block = npmat[i:i+8, j:j+8, k] - 128

                    # Generated dct 
                    dct_matrix = fftpack.dct(block, norm='ortho')

                    # Quantizing the matrix
                    quant_matrix = quantize(dct_matrix,
                                            'lum' if k == 0 else 'chrom')

                    zz = block_to_zigzag(quant_matrix)

                    if block_index == 0 and k==0:
                        print("\n.............................................")
                        print("DCT Co-effecient Matrix: \n",quant_matrix)
                        print("\n.............................................")
                        print("Zig zag Traversal: \n",zz)

                    dc[block_index, k] = zz[0]
                    ac[block_index, :, k] = zz[1:]
                    
        # print("ENCODING_Outer")

        # Calling number of bits required on each DC_Y component.
        H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
        H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))

        H_AC_Y = HuffmanTree(
                flatten(run_length_encode(ac[i, :, 0])[0]
                        for i in range(blocks_count)))
        H_AC_C = HuffmanTree(
                flatten(run_length_encode(ac[i, :, j])[0]
                        for i in range(blocks_count) for j in [1, 2]))

        tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
                  'ac_y': H_AC_Y.value_to_bitstring_table(),
                  'dc_c': H_DC_C.value_to_bitstring_table(),
                  'ac_c': H_AC_C.value_to_bitstring_table()}

        # print("B")
        print("Encoded file size",(sys.getsizeof(tables)+sys.getsizeof(dc)+sys.getsizeof(ac))/1024,"kb")
        print("ENCODING DONE................")
        print("time passed : ", ((datetime.datetime.now() - start).seconds) / 60, " minutes")
        # write_to_file(output_file, dc, ac, blocks_count, tables)
        # print("C")
        # assuming that the block is a 8x8 square
        block_side = 8

        # assuming that the image height and width are equal
        # image_side = int(math.sqrt(blocks_count)) * block_side
        # rows = 672
        # cols = 1200

        # blocks_per_line = image_side // block_side

        npmat = np.empty(or_img.shape, dtype=np.uint8)

        """
        for block_index in range(blocks_count):
            i = block_index // blocks_per_line * block_side
            j = block_index % blocks_per_line * block_side

            for c in range(3):
                zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
                quant_matrix = zigzag_to_block(zigzag)
                dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom')
                block = fftpack.idct(dct_matrix, norm='ortho')
                npmat[i:i+8, j:j+8, c] = block + 128
        """
        i, j = 0, 0
        print("rows : ", rows, " cols : ", cols)
        for i in range(0, rows, 8):
            # print("DECODING_Outer")
            for j in range(0, cols, 8):
                try:
                    block_index1 += 1
                except NameError:
                    block_index1 = 0

                for c in range(3):
                    zigzag = [dc[block_index1, c]] + list(ac[block_index1, :, c])
                    quant_matrix = zigzag_to_block(zigzag)
                    dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom')
                    block = fftpack.idct(dct_matrix, norm='ortho')
                    npmat[i:i+8, j:j+8, c] = block + 128
        block_index1 = 0
        image = Image.fromarray(npmat, 'YCbCr')
        image = image.convert('RGB')
        npmat[-(orows - rows):,-(ocols - cols):,:] = or_img[-(orows - rows):,-(ocols - cols):,:]
        # image.show()
        print("DONE. Time passed : ", ((datetime.datetime.now() - start).seconds) / 60, " minutes")
        output_file = outputDir+ "\\" + imageName + "RSP." + exte
        image.save(output_file)
        
        # Getting the metrics information 
        df = metric(df,input_file+"."+exte,output_file,imageName)
        print("**************************************************")
        print()

    
    df.set_index(['Image Name','PSNR','MSE','RMSE','Compression Ratio','Original Size','Compressed Size'], inplace=True)  # set PSNR as the index
    df.to_csv('Metrics.csv')

if __name__ == "__main__":
    main()
