import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.models import load_model

mp_hands = mp.solutions.hands  # perform the hand recognition algorithm
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils  # draw the detected key points automatically

model = load_model('mp_hand_gesture')
f = open('gesture.names', 'r')
class_names = f.read().split('\n')
# Class names: ['okay', 'peace', 'thumbs up', 'thumbs down',
#               'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

f.close()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, frame = cap.read()
    x, y, s = frame.shape
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Get hands prediction
    result = hands.process(frame_rgb)

    class_name = ''
    # print(result.multi_hand_landmarks)
    # landmark { x: 0.5720219016075134, y: 0.5286498665809631, z: -0.0007330788066610694}

    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmark in result.multi_hand_landmarks:
            # This line is for drawing the points in the hand
            # mp_draw.draw_landmarks(frame, hand_landmark)
            for lm in hand_landmark.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)
        # Add prediction to the frame
        prediction = model.predict([landmarks])
        # [[1.0422494e-14 3.1154030e-03 2.0712767e-12 8.2382339e-01 1.7304973e-01
        #   4.4705446e-33 9.0228349e-09 2.2103661e-08 4.4902349e-10 1.1362285e-05]]

        class_id = np.argmax(prediction)  # This will return the max val in the list
        print(class_id)  # this will return a number from 1 to .9.

        class_name = class_names[class_id]

        cv2.putText(frame, f"Prediction: {class_name}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255),
                    2, cv2.LINE_AA)

    cv2.imshow("Output image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
