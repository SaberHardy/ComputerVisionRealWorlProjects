import cv2
import numpy as np
import time
import os
import hand_tracking as hdm

folder_name = "../Resources/images/painterTools"
list_image_names = os.listdir(folder_name)
overlay_list = []
img_canvas = np.zeros((720, 1280, 3), np.uint8)

brushThickness = 15
eraserThickness = 100

xp, yp = 0, 0

for im_path in list_image_names:
    image = cv2.imread(f"{folder_name}/{im_path}")
    overlay_list.append(image)

header = overlay_list[0]
draw_color = (225, 0, 225)

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
            if y1 < 125:
                if 250 < x1 < 450:
                    header = overlay_list[0]
                    draw_color = (225, 0, 225)
                elif 550 < x1 < 750:
                    header = overlay_list[1]
                    draw_color = (225, 0, 0)
                elif 800 < x1 < 950:
                    header = overlay_list[2]
                    draw_color = (225, 225, 0)
                elif 1050 < x1 < 1200:
                    header = overlay_list[3]
                    draw_color = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25),
                          (x2, y2 + 25),
                          (255, 0, 200),
                          cv2.FILLED)

        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1),
                       20,
                       (255, 0, 200),
                       cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), draw_color, brushThickness)
            cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brushThickness)
            if draw_color == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), draw_color, eraserThickness)

                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), draw_color, brushThickness)
                cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 225, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, img_canvas)

    # setting the header image
    img[0:125, 0:1280] = header
    img = cv2.addWeighted(img, (0.5), img_canvas, 0.5, 0)
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
