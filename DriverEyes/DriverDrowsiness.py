import cv2

eye_cascPath = 'haarcascade_eye_tree_eyeglasses.xml'  # eye detect model
face_cascPath = 'haarcascade_frontalface_alt.xml'  # face detect model
faceCascade = cv2.CascadeClassifier(face_cascPath)
eyeCascade = cv2.CascadeClassifier(eye_cascPath)

cap = cv2.VideoCapture(0)

while 1:
    ret, frame = cap.read()
    if ret:
        # frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        # print("Found {0} faces!".format(len(faces)))
        if len(faces) > 0:
            eyes = eyeCascade.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30))

            if len(eyes) == 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame,
                                  (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255),
                                  2)

                    cv2.putText(frame, "Driver's eyes closed!",
                                (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 255),
                                2)
            else:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame,
                                  (x, y),
                                  (x + w, y + h),
                                  (0, 255, 0),
                                  2)
                    cv2.putText(frame, "Driver's eyes Opened!",
                                (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0),
                                2)

            frame_tmp = cv2.resize(frame, (640, 400), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Face Recognition', frame)

        wait_key = cv2.waitKey(1)
        if wait_key == ord('q') or wait_key == ord('Q'):
            cv2.destroyAllWindows()
            break
