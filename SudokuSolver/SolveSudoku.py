import os
import cv2
import numpy as np

from ToolsModels.tools import *
import sudokuSolver

image_path = 'sudoku.jpeg'
img_height = 400
img_width = 400

# model = initializePredictionModel()

img = cv2.imread(image_path)
img = cv2.resize(img, (img_width, img_height))
blank_img = np.zeros((img_height, img_width, 3), np.uint8)
img_thresh = pre_process(img)
img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_BGR2GRAY)

# Find the contours in the image
contours_img = img.copy()
big_contours_img = img.copy()
contours, hierarchy = cv2.findContours(img_thresh,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contours_img, contours, -1, (0, 200, 0), 2)

# Find the maximum contours and use it in the sudoku
biggest, max_area = biggest_contours(contours)
wrap_colored_img = np.ndarray

if biggest.size != 0:
    biggest = reorder(biggest)

    cv2.drawContours(big_contours_img, biggest, -1, (0, 0, 255), 10)
    pt1 = np.float32(biggest)
    pt2 = np.float32([[0, 0], [img_width, 0], [0, img_height], [img_width, img_height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    wrap_colored_img = cv2.warpPerspective(img, matrix, (img_width, img_height))
    detect_digits = blank_img.copy()


    # Add some changes
    # wrap_colored_img = cv2.cvtColor(wrap_colored_img, cv2.COLOR_BGR2GRAY)

array_images = np.concatenate([img, contours_img, wrap_colored_img], axis=1)

cv2.imshow('Image', array_images)
cv2.waitKey(0)
cv2.destroyAllWindows()
