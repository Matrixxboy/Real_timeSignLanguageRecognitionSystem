import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

from markdown_it.rules_inline import image

# import uuid

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 450
folder = "Data/hello"
counter = 0
previousTime = 0
currentTime = 0

def fit_hand_in_white_space(imgWhite, imgCrop, imgSize, x, w, frame_width):
    """Fits the cropped hand image into the white space while maintaining aspect ratio
       and positioning based on the hand's x-coordinate.

    Args:
      imgWhite: The blank white image.
      imgCrop: The cropped hand image.
      imgSize: The target size of the white image.
      x: The x-coordinate of the hand's bounding box.
      w: The width of the hand's bounding box.
      frame_width: The width of the original video frame.

    Returns:
      The updated imgWhite with the fitted hand image.
    """
    h, w_crop, _ = imgCrop.shape
    aspectRatio = h / w_crop

    if x < frame_width / 2:  # Position for left hand
        max_width = imgSize // 2
        wGap = 0
    else:  # Position for right hand
        max_width = imgSize // 2
        wGap = imgSize // 2

    if aspectRatio > 1:
        k = max_width / w_crop
        wCal = math.ceil(k * w_crop)
        hCal = math.ceil(k * h)
    else:
        k = max_width / w_crop
        wCal = math.ceil(k * w_crop)
        hCal = math.ceil(k * h)

    # Calculate necessary padding to center the hand within the half-space
    if wCal < max_width:
        w_padding = (max_width - wCal) // 2
        wGap += w_padding
    hGap = 0  # Initialize hGap here

    if hCal < imgSize:
        h_padding = (imgSize - hCal) // 2
        hGap += h_padding

    # Ensure proper bounds for wGap and hGap
    wGap = max(0, min(wGap, imgSize - wCal))
    hGap = max(0, min(hGap, imgSize - hCal))

    try:
        imgResize = cv2.resize(imgCrop, (wCal, hCal))
        imgWhite[hGap:hGap + hCal, wGap:wGap + wCal] = imgResize
    except ValueError:
        print(f"Error: Could not place hand image. Shape mismatch: {imgResize.shape} vs. {imgWhite[hGap:hGap + hCal, wGap:wGap + wCal].shape}")

    return imgWhite

while True:
    success, img = cap.read()
    if not success:
        break  # Exit the loop if video capture fails
    frame_width = img.shape[1]
    hands, img = detector.findHands(img)

    if hands:
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        for hand in hands:
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgWhite = fit_hand_in_white_space(imgWhite, imgCrop, imgSize, x, w, frame_width)
        cv2.imshow("ImageWhite", imgWhite)  # Display the combined image

    # calculating the FPS
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    # Displaying FPS on the image
    cv2.putText(img, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)  # Display the original image with hands

    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image for {len(hands)} hands")
    elif key & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()