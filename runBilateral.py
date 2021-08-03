import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt


def bilateral(im, sigma_s=5, sigma_r=0.1):
    # TODO Write "bilateral filter" function based on the illustration in specification.
    # Return filtered result image
    pad_w = int(3 * sigma_s / 2)
    I = np.pad(im, ((pad_w, pad_w), (pad_w, pad_w), (0, 0)), 'edge')
    I /= 255.0
    B, G, R = cv2.split(I)
    result_b = np.zeros([im.shape[0], im.shape[1]], dtype=np.float32)
    result_g = np.zeros([im.shape[0], im.shape[1]], dtype=np.float32)
    result_r = np.zeros([im.shape[0], im.shape[1]], dtype=np.float32)
    #   for each pixel p in image, matrix G_s never change
    G_s = np.zeros([3 * sigma_s, 3 * sigma_s], dtype=np.float32)

    for u in range(3 * sigma_s):
        for v in range(3 * sigma_s):
            spatial_dist = np.square(u - pad_w) + np.square(v - pad_w)
            G_s[u][v] = np.exp(-spatial_dist / np.square(sigma_s))
    #   As G_r is relevant with the intensity of pixel, matrix G_r varies on different pixel
    G_r_b = np.zeros([3 * sigma_s, 3 * sigma_s], dtype=np.float32)
    G_r_g = np.zeros([3 * sigma_s, 3 * sigma_s], dtype=np.float32)
    G_r_r = np.zeros([3 * sigma_s, 3 * sigma_s], dtype=np.float32)
    spatial_kernel = np.ones([3 * sigma_s, 3 * sigma_s], dtype=np.float32)
    spatial_kernel[pad_w][pad_w] = 0
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for u in range(3*sigma_s):
                for v in range(3*sigma_s):
                    #   noted that im[x][y] == I[x + padw][y + pad_w]
                    intensity_dist_b = np.square(I[x + pad_w][y + pad_w][0] - I[x + u][y + v][0])
                    intensity_dist_g = np.square(I[x + pad_w][y + pad_w][1] - I[x + u][y + v][1])
                    intensity_dist_r = np.square(I[x + pad_w][y + pad_w][2] - I[x + u][y + v][2])
                    G_r_b[u][v] = np.exp(-intensity_dist_b / np.square(sigma_r))
                    G_r_g[u][v] = np.exp(-intensity_dist_g / np.square(sigma_r))
                    G_r_r[u][v] = np.exp(-intensity_dist_r / np.square(sigma_r))
            intensity_kernel_b = I[x:x + 3 * sigma_s, y:y + 3 * sigma_s, 0:1]
            intensity_kernel_g = I[x:x + 3 * sigma_s, y:y + 3 * sigma_s, 1:2]
            intensity_kernel_r = I[x:x + 3 * sigma_s, y:y + 3 * sigma_s, 2:]
            intensity_kernel_b = np.squeeze(intensity_kernel_b, axis=-1)
            intensity_kernel_g = np.squeeze(intensity_kernel_g, axis=-1)
            intensity_kernel_r = np.squeeze(intensity_kernel_r, axis=-1)
            intensity_kernel_b[pad_w][pad_w] = 0
            intensity_kernel_g[pad_w][pad_w] = 0
            intensity_kernel_r[pad_w][pad_w] = 0
            G_b = np.multiply(G_s, G_r_b)    #   np.multiply: hadamard (elementwise) product
            G_g = np.multiply(G_s, G_r_g)
            G_r = np.multiply(G_s, G_r_r)
            result_b[x][y] = filter2D(G_b, intensity_kernel_b, 3 * sigma_s, 3 * sigma_s) / filter2D(G_b, spatial_kernel, 3 * sigma_s, 3 * sigma_s)
            result_g[x][y] = filter2D(G_g, intensity_kernel_g, 3 * sigma_s, 3 * sigma_s) / filter2D(G_g, spatial_kernel, 3 * sigma_s, 3 * sigma_s)
            result_r[x][y] = filter2D(G_r, intensity_kernel_r, 3 * sigma_s, 3 * sigma_s) / filter2D(G_r, spatial_kernel, 3 * sigma_s, 3 * sigma_s)
    result = cv2.merge([result_b, result_g, result_r])
    #   result = cv2.bilateralFilter(src=im, d=15, sigmaColor=sigma_r, sigmaSpace=sigma_s)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return result


def filter2D(src, kernel, H, W):
    result = np.float32(0.0)
    for x in range(H):
        for y in range(W):
            result += src[x][y] * kernel[x][y]
    return result


if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp').astype(np.float32)
    
    sigma_s = 5
    sigma_r = 0.1
    result = bilateral(im, sigma_s, sigma_r)
    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    cv2.imshow('output',result)
    cv2.waitKey(0)
    #   plt.imshow(result);
    #   plt.show()