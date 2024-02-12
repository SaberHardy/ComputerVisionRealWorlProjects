import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.7)

start_distance = None
scale = 0
cx, cy = 200, 200

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    image_to_zoom = cv2.imread("../Resources/images/Readable.jpg")
    if len(hands) == 2:
        hand1 = hands[0]
        hand2 = hands[1]

        hand1_fingers = detector.fingersUp(hand1)
        hand2_fingers = detector.fingersUp(hand2)
        if hand1_fingers == [1, 1, 0, 0, 0] and hand2_fingers == [1, 1, 0, 0, 0]:
            landmarks_list1 = hand1['lmList']
            landmarks_list2 = hand2['lmList']

            if start_distance is None:
                length, info, img = detector.findDistance(hand1['center'], hand2['center'], img)
                start_distance = length

            length, info, img = detector.findDistance(hand1['center'], hand2['center'], img)

            scale = int((length - start_distance) // 2)
            cx, cy = info[4:]
        else:
            start_distance = None

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
