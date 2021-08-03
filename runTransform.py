import scipy.io as sio
import numpy as np
import cv2

from matplotlib import pyplot as plt


def image_t(im, scale=1.0, rot=45):
    """

    :param im: source image
    :param scale: scale ratio, default=1.0
    :param rot: rotation angle, in degree, default=45
    :return: affine transformed image
    """
    # TODO Write "image affine transformation" function based on the illustration in specification.
    # Return transformed result image
    radian = np.pi * rot / 180
    height, width, channel = im.shape
    mat = np.float32([[scale*np.cos(radian), scale*np.sin(radian), 0],
              [-scale*np.sin(radian), scale*np.cos(radian), 0],
              [0, 0, 1]])
    src1 = np.float32([0, 0, 1])
    src2 = np.float32([height/2-1, 0, 1])
    src3 = np.float32([0, width/2-1, 1])
    dest1 = np.dot(mat, src1)
    dest2 = np.dot(mat, src2)
    dest3 = np.dot(mat, src3)
    src_pts = np.float32([[height/2, width/2] + src1[0:2], [height/2, width/2] + src2[0:2], [height/2, width/2] + src3[0:2]])
    dest_pts = np.float32([[height/2, width/2] + dest1[0:2], [height/2, width/2] + dest2[0:2], [height/2, width/2] + dest3[0:2]])
    M = cv2.getAffineTransform(src_pts, dest_pts)
    result = cv2.warpAffine(im, M, (height, width), flags=cv2.INTER_LINEAR)
    return result


if __name__ == '__main__':
    im = cv2.imread('./misc/lena_gray.bmp')

    scale = 1.0
    rot = 45
    result = image_t(im, scale, rot)
    # Show result
    # Or: 
    cv2.imshow('output', result)
    cv2.waitKey(0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB, result)
    plt.imshow(result)
    plt.show()
