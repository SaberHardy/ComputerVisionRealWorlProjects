import numpy as np
import cv2

img = cv2.imread('leafs.jpg')
img = cv2.resize(img, (600, 400))
# To Detect any Color, we need to transform the image into HSV
# Hue: modeled as an angular dimension that encodes color
# Saturation: Encodes the intensity of the color
# Value: represents the amount of colors is mixed with the black

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# low and high of green color
lower_bound = np.array([50, 20, 20])
upper_bound = np.array([100, 255, 255])

mask = cv2.inRange(img_hsv, lower_bound, upper_bound)

# Step 3: Remove noise
kernel = np.ones((7, 7), np.uint8)

# Step 3.a - Remove the noise from the image

# Step 3.b - Remove black noise from white region
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Step 4.b - Remove white noise from black region
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

segmented_img = cv2.bitwise_and(img, img, mask=mask)

contours, hierarchy = cv2.findContours(mask.copy(),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

output = cv2.drawContours(segmented_img,
                          contours,
                          -1,
                          (0, 0, 255),
                          3)

cv2.imshow('Original', img)
# cv2.imshow('HSV', img_hsv)
# cv2.imshow('segmented_img', segmented_img)
cv2.imshow('output', output)
cv2.waitKey(0)
