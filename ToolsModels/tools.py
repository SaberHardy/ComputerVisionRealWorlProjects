import cv2
import numpy as np


# from keras.models import load_model


# def initialize_prediction_model():
#     model = load_model('sudoku_model.h5')
#     return model


def pre_process(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 1)
    thresh_img = cv2.adaptiveThreshold(blur_img,
                                       255,
                                       1,
                                       1,
                                       11,
                                       2)

    thresh_img_3d = np.stack((thresh_img,) * 3, axis=-1)

    return thresh_img_3d


def biggest_contours(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            arc_length = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * arc_length, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest, max_area


def reorder(points):
    points = points.reshape((4, 2))

    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    sum_points = points.sum(1)

    new_points[0] = points[np.argmin(sum_points)]
    new_points[3] = points[np.argmax(sum_points)]

    difference = np.diff(points, axis=1)

    new_points[1] = points[np.argmin(difference)]
    new_points[2] = points[np.argmax(difference)]

    return new_points
