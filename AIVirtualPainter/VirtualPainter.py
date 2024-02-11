import cv2
import numpy as np
import time
import os
import hand_tracking as hdm

folder_name = "../Resources/images/painterTools"
list_image_names = os.listdir(folder_name)
overlay_list = []

for im_path in list_image_names:
    image = cv2.imread(f"{folder_name}/{im_path}")
    overlay_list.append(image)

header = overlay_list[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = hdm.HandDetector()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    frame = detector.findHands(frame)

    # setup the image bar
    frame[0:125, 0:1280] = header

    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
