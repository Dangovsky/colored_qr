import cv2
import os
import numpy as np

# Gaussian-distributed noise.


def gauss_noise(image, mean=0, sigma=1):
    row, col, ch = image.shape

    gauss = np.round(np.random.normal(mean, sigma, (row, col, ch)))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss.astype(np.int16)

    return np.clip(noisy, 0, 255).astype(np.uint8)


# Replaces random pixels with 0 or 1.
def s_p_noise(image, s_vs_p=0.5, amount=0.004):
    row, col, ch = image.shape
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    out[tuple(coords)] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    out[tuple(coords)] = 0
    return out.astype(np.uint8)


# Poisson-distributed noise generated from the data.
def poisson_noise(image, peak=0.5):
    #noisy = np.random.poisson(image / 255.0 * peak) / peak * 255
    #vals = len(np.unique(image))
    #vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image / 255.0 * peak) / peak * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)


# Multiplicative noise using out = image + n*image, where
# n is uniform noise with specified mean & sigma.
def speckle_noise(image, mean=0, sigma=0.1):
    row, col, ch = image.shape

    gauss = sigma * np.random.randn(row, col, ch) + mean
    gauss = gauss.reshape(row, col, ch).astype(np.uint16)
    noisy = image + image * gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)
