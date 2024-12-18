import numpy as np
import cv2
import math

def calculate_hs_histogram(img, bin_size):
    height, width, _ = img.shape
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    max_h = 179
    max_s = 255
    hs_hist = np.zeros((math.ceil(max_h+1/bin_size), math.ceil(max_s+1/bin_size)))
    for i in range(height):
        for j in range(width):
            h = img_hsv[i, j, 0]
            s = img_hsv[i, j, 1]
            hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] += 1
    hs_hist /= hs_hist.sum()
    return hs_hist

def color_segmentation(img, hs_hist, bin_size, threshold):
    height, width, _ = img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = np.zeros((height, width, 1))
    for i in range(height):
        for j in range(width):
            h = hsv[i, j, 0]
            s = hsv[i, j, 1]
            if hs_hist[math.floor(h/bin_size), math.floor(s/bin_size)] > threshold:
                mask[i, j, 0] = 1
    return mask


# Training
img_train = cv2.imread("training_image.png")

bin_size = 20
hs_hist = calculate_hs_histogram(img_train, bin_size)

# Testing
img_test = cv2.imread("testing_image.bmp")

threshold = 0.03
mask = color_segmentation(img_test, hs_hist, bin_size, threshold)

img_seg = img_test * mask

cv2.imshow("Input", img_test)
cv2.imshow("Mask", (mask*255).astype(np.uint8))
cv2.imshow("Segmentation", img_seg.astype(np.uint8))
cv2.waitKey()
