import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt


def jpegCompress(image, quantmatrix):
    '''
        Compress(imagefile, quanmatrix simulates the lossy compression of 
        baseline JPEG, by quantizing the DCT coefficients in 8x8 blocks
    '''
    # Return compressed image in result

    H = np.size(image, 0)
    W = np.size(image, 1)

    # Number of 8x8 blocks in the height and width directions
    h8 = H / 8
    w8 = W / 8

    # TODO If not an integer number of blocks, pad it with zeros
    if int(h8) != h8:
        h8 += 1
    if int(w8) != w8:
        w8 += 1
    H_padded = h8 * 8
    W_padded = w8 * 8

    if H_padded != H or W_padded != W:
        top = (H_padded - H) / 2
        bottom = H_padded - (H_padded - H) / 2
        left = (W_padded - W) / 2
        right = W_padded - (W_padded - W) / 2
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    # TODO Separate the image into blocks, and compress the blocks via quantization DCT coefficients
    result = np.zeros(image.shape, dtype=np.float32)
    for x in range(int(h8)):
        for y in range(int(w8)):
            F = np.float32(image[x * 8:(x + 1) * 8, y * 8: (y + 1) * 8, :])
            F_b, F_g, F_r = cv2.split(F)
            F_b /= 255.0
            F_g /= 255.0
            F_r /= 255.0
            G_b = np.zeros(F_b.shape, dtype=np.float32)
            G_g = np.zeros(F_g.shape, dtype=np.float32)
            G_r = np.zeros(F_r.shape, dtype=np.float32)
            G_b = cv2.dct(F_b, G_b)
            G_g = cv2.dct(F_g, G_g)
            G_r = cv2.dct(F_r, G_r)
            B_b = np.float32(np.divide(G_b, quantmatrix))
            B_g = np.float32(np.divide(G_g, quantmatrix))
            B_r = np.float32(np.divide(G_r, quantmatrix))
            D_b = np.zeros(B_b.shape, dtype=np.float32)
            D_g = np.zeros(B_g.shape, dtype=np.float32)
            D_r = np.zeros(B_r.shape, dtype=np.float32)
            D_b = cv2.idct(B_b, D_b)
            D_g = cv2.idct(B_g, D_g)
            D_r = cv2.idct(B_r, D_r)
            decoded = np.zeros(F.shape, dtype=np.float32)
            decoded = cv2.merge(mv=[D_b, D_g, D_r], dst=decoded)
            #   decoded *= 255.0
            result[int(x * 8):int((x + 1) * 8), int(y * 8):int((y + 1) * 8), :] = decoded
    #   de-padding
    if H_padded != H or W_padded != W:
        result = result[top:-bottom, left:-right, :]
    #   normalize compressed image in the intensity scope of [0, 255] and transform the type to uint8
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return result


if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp')
    im.astype('float')

    quantmatrix = sio.loadmat('./misc/quantmatrix.mat')['quantmatrix']

    out = jpegCompress(im, quantmatrix)
    #   cv2.imshow('output', im)
    #   cv2.waitKey(0)
    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or:
    cv2.imshow('output', out)
    cv2.waitKey(0)
    #   result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB, result)
    #   plt.imshow(out)
    #   plt.show()
    exit(0)
