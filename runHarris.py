import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt


def harrisdetector(image, k, t):
    # TODO Write harrisdetector function based on the illustration in specification.
    # Return corner points x-coordinates in result[0] and y-coordinates in result[1]
    ptr_x = []
    ptr_y = []
    #   color image to grayscale image
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    H, W = image.shape
    w = np.ones([2 * k + 1, 2 * k + 1], dtype=np.uint8)
    horizontal_kernel = np.array([[0, 0, 0],
                                  [0, -1, 1],
                                  [0, 0, 0]])
    vertical_kernel = np.array([[0, 0, 0],
                                [0, -1, 0],
                                [0, 1, 0]])

    I_x = cv2.filter2D(src=image, ddepth=-1, kernel=horizontal_kernel)
    I_y = cv2.filter2D(src=image, ddepth=-1, kernel=vertical_kernel)
    I_x2 = np.multiply(I_x, I_x)
    I_xy = np.multiply(I_x, I_y)
    I_y2 = np.multiply(I_y, I_y)
    A_x2 = cv2.filter2D(src=I_x2, ddepth=-1, kernel=w)
    A_xy = cv2.filter2D(src=I_xy, ddepth=-1, kernel=w)
    A_y2 = cv2.filter2D(src=I_y2, ddepth=-1, kernel=w)
    A = np.zeros([image.shape[0], image.shape[1], 2, 2], dtype=np.float32)
    l = np.zeros([2], dtype=np.float32)
    for u in range(H):
        for v in range(W):
            A[u][v] = np.array([[A_x2[u][v], A_xy[u][v]],
                               [A_xy[u][v], A_y2[u][v]]])
            cv2.eigen(src=A[u][v], eigenvalues=l)
            if l[0] >= t and l[1] >= t:
                ptr_x.append(v)
                ptr_y.append(u)
    result = [ptr_x, ptr_y]
    return result


if __name__ == '__main__':
    k = 3  # change to your value
    t = 0.46  # change to your value

    I = cv2.imread('./misc/corner_gray.png')

    fr = harrisdetector(I, k, t)

    # Show input, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('output',out)
    #   cv2.waitKey(0)
    plt.imshow(I)
    # plot harris points overlaid on input image
    plt.scatter(x=fr[0], y=fr[1], c='r', s=40)

    # show
    plt.show()
