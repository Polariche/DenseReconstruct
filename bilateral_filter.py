# https://github.com/anlcnydn/bilateral/blob/master/bilateral_filter.py

import numpy as np
import cv2
import sys
import math

def gaussian(x, sigma):
    return (1.0 / (2 * np.pi * (sigma ** 2))) * np.exp(- x / (2 * sigma ** 2))


def bilateral_filter(img, sigma_c, sigma_s):
    h, w = img.shape

    n = h*w

    u = np.array([[int(i/w), i%w] for i in range(n)]).T
    u_ = np.hstack([u.T, np.ones(h*w).reshape(-1,1)]).T

    # compute D with bilateral filter
    q = np.repeat(u.reshape(2, 1, -1), n, axis=1)
    u_rep = np.repeat(u.reshape(2, -1, 1), n, axis=2)
    u_q = np.sum((u_rep - q)**2, axis=0)

    R_u = img[u[0], u[1]]
    R_q = np.repeat(R_u.reshape(1, -1), n, axis=0)
    R_u = np.repeat(R_u.reshape(-1, 1), n, axis=1)
    R_u_q = (R_u - R_q)**2

    W = np.sum(gaussian(u_q, sigma_s) * gaussian(u_q, sigma_c), axis=1).reshape(h,w)
    D = np.sum(gaussian(u_q, sigma_s) * gaussian(u_q, sigma_c) * R_q, axis=1).reshape(h,w)

    return D/W
