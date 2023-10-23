import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def read_image():
    return cv2.imread('templates/kitten.jfif')


def add_noise(img, noise_type='gaussian', mean=0, sigma=50,):
    if noise_type == 'gaussian':
        row, col, ch = img.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np.clip(img + gauss, 0, 255)
        return noisy.astype(np.uint8)


def default_filtering(img_noised):
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / kernel_size ** 2
    return cv2.filter2D(img_noised, -1, kernel)


def filtering(img_noised):
    kernel = np.array([
        [-2, -1, 0, -1, -2],
        [-1, 1, -1, 1, -1],
        [0, 1, 8, 1, 0],
        [-1, 1, -1, 1, -1],
        [-2, -1, 0, -1, -2],
    ])
    return cv2.filter2D(img_noised, -1, kernel)


def showing(img, img_noised, img_defaultfiltered, img_filtered):
    plt.figure(figsize=(20, 5))
    plt.subplot(141), plt.imshow(img), plt.title('Original')
    plt.subplot(142), plt.imshow(img_noised), plt.title('Gaussian Noise')
    plt.subplot(143), plt.imshow(img_defaultfiltered), plt.title('Averaging default')
    plt.subplot(144), plt.imshow(img_filtered), plt.title('Averaging')
    plt.show()


if __name__ == "__main__":
    image = read_image()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_noised = add_noise(image)
    image_DefaultFiltered = default_filtering(image_noised)
    image_filtered = filtering(image_noised)
    showing(image, image_noised, image_DefaultFiltered, image_filtered)
