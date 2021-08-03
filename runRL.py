import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt


def rlDeconv(B, PSF):
    # TODO Implement rl_deconv function based on spec.
    # Change based on your experiment
    maxIters = 36

    pad_w = 30
    # Pad border to avoid artifacts
    I = np.pad(B, ((pad_w, pad_w), (pad_w, pad_w), (0, 0)), 'edge')
    B = np.pad(B, ((pad_w, pad_w), (pad_w, pad_w), (0, 0)), 'edge')
    I_b, I_g, I_r = cv2.split(I)
    B_b, B_g, B_r = cv2.split(B)
    I_channels = [I_b, I_g, I_r]
    B_channels = [B_b, B_g, B_r]
    PSF_flipped = np.zeros(PSF.shape, dtype=np.float32)
    PSF_flipped = cv2.flip(src=PSF, dst=PSF_flipped, flipCode=-1)
    for i in range(0, maxIters):
        for c in range(3):
            denominator = cv2.filter2D(src=I_channels[c], ddepth=-1, kernel=PSF)
            frac = np.divide(B_channels[c], denominator)
            corr = cv2.filter2D(src=frac, ddepth=-1, kernel=PSF_flipped)
            I_channels[c] = np.multiply(I_channels[c], corr)
    I = cv2.merge(mv=I_channels, dst=I)
    I = I[pad_w: -pad_w, pad_w: -pad_w]
    return I


if __name__ == '__main__':
    gt = cv2.imread('./misc/lena_gray.bmp').astype('double')
    gt = gt / 255.0

    # You can change to other PSF
    PSF = sio.loadmat('./misc/psf.mat')['PSF']
    # Generate blur image
    B = cv2.filter2D(gt, -1, PSF)

    # Show image, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('B', B)
    #   cv2.waitKey(0)
    plt.imshow(B[:, :, [2, 1, 0]]) # for color image
    #   plt.imshow(B)
    plt.show()

    # Deconvolve image using RL
    I = rlDeconv(B, PSF)

    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('I',I)
    #   cv2.waitKey(0)
    plt.imshow(I)
    plt.show()
