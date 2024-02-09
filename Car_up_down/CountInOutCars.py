import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from ToolsModels.tracker import *

model = YOLO('../Resources/files/yolov8s.pt')


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colors_bgr = [x, y]
        # print(colors_bgr)


cv2.namedWindow('Cars')
cv2.setMouseCallback('Cars', click_event)

cap = cv2.VideoCapture('../Resources/Vids/cars_high_way.mp4')

my_file = open("../Resources/files/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0
vehicle_down = {}
vehicle_up = {}
count_up = {}

tracker = Tracker()

cy1 = 326
cy2 = 386
offset = 6

counter = []
counter1 = []

count_cars = set()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))

    if not ret:
        break

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a)

    list_rect_coordinates = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        detected_object_id = int(row[5])
        class_list_id = class_list[detected_object_id]

        if 'car' in class_list_id:
            list_rect_coordinates.append([x1, y1, x2, y2])

    bbox_ids = tracker.update(list_rect_coordinates)

    for bbox in bbox_ids:
        x3, y3, x4, y4, item_id = bbox

        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Cars going Down
        if (cy + offset) > cy1 > (cy - offset):
            vehicle_down[item_id] = cy
            # cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        if item_id in vehicle_down:
            if (cy + offset) > cy2 > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame,
                            str(item_id),
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0),
                            2)
                if counter.count(item_id) == 0:
                    counter.append(item_id)

        if (cy + offset) > cy2 > (cy - offset):
            vehicle_up[item_id] = cy

        if item_id in vehicle_up:
            if (cy + offset) > cy1 > (cy - offset):
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame,
                            str(item_id),
                            (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2)
                if counter1.count(item_id) == 0:
                    counter1.append(item_id)

    cv2.line(frame, (140, cy1), (683, cy1), (0, 255, 0), 2)
    cv2.line(frame, (100, cy2), (736, cy2), (0, 255, 0), 2)

    cv2.putText(frame,
                "Cars IN: " + str(len(counter)),
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                3)

    cv2.putText(frame,
                "Cars Out: " + str(len(counter1)),
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                3)

    cv2.imshow("Cars", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
