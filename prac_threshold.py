import cv2
import numpy as np
from matplotlib import pyplot as plt

if __name__ == "__main__":
    print("Hello, World!")
    img = cv2.imread("190307-122.JPG", 0) # read as a grayscale
    ret, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    plt.figure(figsize=(24,6))
    plt.subplot(1,3, 1)
    plt.imshow(thresh1, 'gray')
    plt.subplot(1,3, 2)
    plt.imshow(thresh2, 'gray')
    plt.subplot(1,3,3)
    plt.imshow(thresh3, 'gray')
    plt.show()