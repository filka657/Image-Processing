import cv2
import random
import numpy as np
import matplotlib.pyplot as plt


def read_image():
    return cv2.imread('templates/kitten.jfif')


def add_noise(img, noise_type='gaussian', mean=0, sigma=25,):
    if noise_type == 'gaussian':
        row, col, ch = img.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np.clip(img + gauss, 0, 255)
        return noisy.astype(np.uint8)


def gauss_filter(img_noised, apert):
    return cv2.GaussianBlur(img_noised, apert, 0)


def filter_canny(img):
    return cv2.Canny(img, 100, 200, L2gradient=False)


def calcAndDrawHist(img_hsv):
    h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
    hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
    hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])

    for hist in [hist_h, hist_s, hist_v]:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
        print(type(hist), ':')
        print('minval:', minVal)
        print('maxval:', maxVal)
        print('minLoc:', minLoc)
        print('maxLoc:', maxLoc)
    plt.plot(hist_h, color='r', label="h")
    plt.plot(hist_s, color='g', label="s")
    plt.plot(hist_v, color='b', label="v")
    plt.legend()
    plt.show()


def showing(img, img_noised, img_gauss, img_canny, img_SuperBlured, img_final):
    plt.figure(figsize=(12, 12))
    plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Original')
    plt.subplot(232), plt.imshow(img_noised, cmap='gray'), plt.title('Gaussian Noise')
    plt.subplot(233), plt.imshow(img_gauss), plt.title('Gaussian Blur')
    plt.subplot(234), plt.imshow(img_canny, cmap='gray'), plt.title('Canny')
    plt.subplot(235), plt.imshow(img_SuperBlured), plt.title('Gaussian SuperBlur')
    plt.subplot(236), plt.imshow(img_final), plt.title('Final result')
    plt.show()


if __name__ == "__main__":
    image = read_image()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_HSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    image_noised = add_noise(image)
    image_gauss = gauss_filter(image_noised, apert=(5, 5))

    image_canny = filter_canny(image_gauss)
    calcAndDrawHist(image_HSV)
    image_cat = cv2.inRange(image_HSV, np.array((100, 10, 200), np.uint8), np.array((110, 30, 220), np.uint8))
    cv2.imshow('hsv_cat', image_cat)
    cv2.waitKey()
    cv2.destroyAllWindows()

    image_SuperBlured = gauss_filter(image_noised, apert=(49, 49))
    image_final = cv2.subtract(image_noised, image_SuperBlured)

    showing(image, image_noised, image_gauss, image_canny, image_SuperBlured, image_final)
