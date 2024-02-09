import cv2
import numpy as np

# Variables to store lower and upper Hue bounds
lower_hue = [181, 166, 66]
upper_hue = [255, 255, 240]  # Maximum value for Hue is 179 in OpenCV


def get_hue_value(event, x, y, flags, param):
    global lower_hue, upper_hue

    if event == cv2.EVENT_LBUTTONDOWN:
        hsv_pixel = hsv[y, x]
        print("Clicked HSV Values:", hsv_pixel)

        # Assign the Hue value to either lower or upper bound based on a toggle
        if toggle:
            lower_hue = hsv_pixel[0]
        else:
            upper_hue = hsv_pixel[0]


# Load an image
image = cv2.imread('keys.jpg')
# Convert the image to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window and set the callback function
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', get_hue_value)

# Toggle variable to switch between lower and upper bounds
toggle = True

while True:
    cv2.imshow('Image', image)

    # Press 't' to toggle between lower and upper bounds
    key = cv2.waitKey(1)
    if key == 27:  # Break the loop if 'ESC' key is pressed
        break
    elif key == ord('t'):
        toggle = not toggle

print("Lower Hue Bound:", lower_hue)
print("Upper Hue Bound:", upper_hue)

cv2.destroyAllWindows()
