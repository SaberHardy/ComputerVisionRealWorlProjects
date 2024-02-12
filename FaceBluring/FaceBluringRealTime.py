import cv2

cascade_path = 'haarcascade_frontalface_alt.xml'
cascade_face = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        faces = cascade_face.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(30, 30))
        print(faces)
        for x, y, w, h in faces:
            face_part = frame[y:y + h, x:x + w]
            roi = cv2.GaussianBlur(face_part, (25, 25), 30)
            frame[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

            # cv2.rectangle(frame,
            #               (x, y),
            #               (x + w, y + h),
            #               (255, 255, 255),
            #               -1)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
