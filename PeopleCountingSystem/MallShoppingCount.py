import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from ToolsModels.tracker import *

model = YOLO('../Resources/files/yolov8s.pt')


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        colors_bgr = [x, y]
        print(colors_bgr)


cv2.namedWindow('ShoppingMall')
cv2.setMouseCallback('ShoppingMall', click_event)

# cap = cv2.VideoCapture('shoppingmall.mp4')
cap = cv2.VideoCapture('../Resources/Vids/BigShopping.mp4')

my_file = open("../Resources/files/coco.txt", "r")

data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0

tracker = Tracker()

area = [(720, 157), (727, 233), (789, 233), (783, 158)]
area2 = [(715, 274), (720, 355), (786, 355), (779, 282)]

count_cars = set()

# people_enter = {}
# counter1 = []
#
# people_exit = {}
# counter2 = []
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))

    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a)

    """
    0           1           2           3         4     5
    0   636.590759  207.587875  743.014526  455.269836  0.879535   0.0
    1   214.904907  231.706772  273.182190  378.770264  0.817533   0.0
    2   540.017395  249.995422  586.014771  367.956909  0.790075   0.0
    """
    list_rect_coordinates = []

    for index, row in px.iterrows():
        """
        print(f"Row data ====> {row}")
        This will return for example one object, with this coordinates:
        0    636.590759 => x1
        1    207.587875 => y1
        2    743.014526 => x2
        3    455.269836 => y2
        4      0.879535 =>
        5      0.000000 => object_id
        Name: 0, dtype: float32
        """

        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])

        detected_object_id = int(row[5])
        class_list_id = class_list[detected_object_id]

        if 'person' in class_list_id:
            list_rect_coordinates.append([x1, y1, x2, y2])

    bbox_ids = tracker.update(list_rect_coordinates)

    for bbox in bbox_ids:
        x3, y3, x4, y4, item_id = bbox

        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        results = cv2.pointPolygonTest(
            np.array(area, np.int32),
            (cx, cy),
            False)

        results2 = cv2.pointPolygonTest(
            np.array(area2, np.int32),
            (cx, cy),
            False)

        if results2 >= 0:  # or results2 >= 0:
            cv2.rectangle(frame,
                          (x3, y3), (x4, y4),
                          (0, 255, 0),
                          2)

            cv2.putText(frame, str(item_id),
                        (x3 + 10, y3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1)

            cv2.circle(frame, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            count_cars.add(item_id)

    cv2.putText(frame, "Number of Persons: " + str(len(count_cars)),
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 255),
                3)

    cv2.polylines(frame,
                  [np.array(area, np.int32)],
                  True,
                  (255, 88, 0), 2
                  )
    cv2.polylines(frame,
                  [np.array(area2, np.int32)],
                  True,
                  (255, 88, 0), 2)
    cv2.imshow("ShoppingMall", frame)

    if cv2.waitKey() & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
