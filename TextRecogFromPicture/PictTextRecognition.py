import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr

# read image
image = cv2.imread('../Resources/images/Readable.jpg')

# Instance text detector
reader = easyocr.Reader(['en'], gpu=False)

# Detect text on image
text = reader.readtext(image)

print(text)

for item in text:
    bbox, text_, score = item

    cv2.rectangle(image,
                  (int(bbox[0][0]), int(bbox[0][1])),
                  (int(bbox[2][0]), int(bbox[2][1] - 10)),
                  (0, 200, 200),
                  2)

    cv2.putText(image,
                text_,
                (int(bbox[0][0]), int(bbox[0][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 200),
                2)

    print(text_)

cv2.imshow("Window", image)
cv2.waitKey(0)
