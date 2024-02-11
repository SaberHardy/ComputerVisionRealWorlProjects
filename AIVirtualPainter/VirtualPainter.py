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
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)

    lmList, bbox = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        # print(lmList)
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. check which finger is up
        fingers = detector.fingersUp()

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection Mode")

            cv2.rectangle(img, (x1, y1 - 25),
                          (x2, y2 + 25),
                          (255, 0, 200),
                          cv2.FILLED)

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1),
                       20,
                       (255, 0, 200),
                       cv2.FILLED)
            print("Drawing mode!!!!.>>>")

    # set up the image bar
    img[0:125, 0:1280] = header

    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
