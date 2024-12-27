from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

app = Flask(__name__)

def process_frame(img, detector, classifier, labels):
    """Processes a single frame from the video stream.

    Args:
        img: The input frame (OpenCV image).
        detector: The hand detector object.
        classifier: The gesture classifier object.
        labels: A list of gesture labels.

    Returns:
        The processed frame (OpenCV image) with detection and label information.
    """
    try:
        img_output = img.copy()  # Create a copy to draw on

        try:
            hands, img = detector.findHands(img)  # Detect hands in the frame
        except Exception as e:
            print(f"Error in hand detection: {e}")
            return img_output

        if hands:  # If at least one hand is detected
            hand = hands[0]  # Get the first detected hand
            x, y, w, h = hand['bbox']  # Get bounding box coordinates
            offset = 20  # Offset for cropping

            # Create a white image for consistent input to the classifier
            img_white = np.ones((300, 300, 3), np.uint8) * 255
            try:
                img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset] # Crop the hand image
            except Exception as e:
                print(f"Error in image cropping: {e}")
                return img_output

            if img_crop.size > 0: #check if image is not empty
                aspect_ratio = h / w  # Calculate aspect ratio of the cropped hand

                # Resize and center the cropped hand on the white image
                if aspect_ratio > 1:
                    k = 300 / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, 300))
                    w_gap = math.ceil((300 - w_cal) / 2)
                    img_white[:, w_gap:w_cal + w_gap] = img_resize
                else:
                    k = 300 / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (300, h_cal))
                    h_gap = math.ceil((300 - h_cal) / 2)
                    img_white[h_gap:h_cal + h_gap, :] = img_resize

                try:
                    prediction, index = classifier.getPrediction(img_white, draw=False)  # Classify the hand gesture
                    label = labels[index]  # Get the label corresponding to the predicted index
                    print(f"Detection: Label Index = {index}, Label Name = {label}")

                    # --- Label Box Adjustments ---
                    font_scale = 2  # Adjust font size
                    thickness = 3  # Adjust text thickness
                    font = cv2.FONT_HERSHEY_SIMPLEX #You can use other fonts also

                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]  # Get text size
                    text_bg_width = text_size[0] + 30  # Add padding
                    text_bg_height = text_size[1] + 20  # Add padding

                    text_bg_x = x - offset
                    text_bg_y = y - offset - text_bg_height - 10  # Position above the box

                    # Ensure box doesn't go off-screen
                    if text_bg_x < 0:
                        text_bg_x = 0
                    if text_bg_y < 0:
                        text_bg_y = 0

                    # Draw the label background and text
                    cv2.rectangle(img_output, (text_bg_x, text_bg_y),
                                  (text_bg_x + text_bg_width, text_bg_y + text_bg_height), (0, 0, 0), -1)
                    cv2.putText(img_output, label, (text_bg_x + 10, text_bg_y + text_size[1]), font, font_scale, (255, 255, 255), thickness)

                    # Detection Box with Gradient
                    overlay = img_output.copy()
                    cv2.rectangle(overlay, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), -1)  # Filled green
                    alpha = 0.3  # Adjust transparency here
                    cv2.addWeighted(overlay, alpha, img_output, 1 - alpha, 0, img_output)
                    cv2.rectangle(img_output, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 0, 0), 2)  # Black border



                except Exception as e:
                    print(f"Error during prediction: {e}")
        return img_output
    except Exception as e:
        print(f"Error in process_frame outer try: {e}")
        return img

def generate_frames():
    """Generates video frames for streaming."""
    cap = cv2.VideoCapture(0)  # Open the default camera (0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n' #return empty frame
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Set height

    try:
        detector = HandDetector(maxHands=1)  # Initialize hand detector
        classifier = Classifier("Final_Model/AtoZ.h5", "Final_Model/AtoZ.txt") # Initialize classifier
        with open("Final_Model/AtoZ.txt", 'r') as f: #open   file
            labels = [line.strip() for line in f] #read labels file
    except Exception as e:
        print(f"Error initializing detector or classifier or labels: {e}")
        cap.release()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n\r\n' #return empty frame
        return

    prev_frame_time = 0  # Initialize previous frame time

    while True:
        try:
            new_frame_time = time.time()  # Get current time
            success, img = cap.read()  # Read a frame from the camera
            if not success:
                print("Error reading frame. Reconnecting...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Camera reconnection failed.")
                    break
                continue

            img = process_frame(img, detector, classifier, labels)  # Process the frame

            fps = 1 / (new_frame_time - prev_frame_time)  # Calculate FPS
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps_text = "FPS: " + str(fps)  # Add "FPS: " prefix

            cv2.putText(img, fps_text, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)  # Display FPS

            ret, buffer = cv2.imencode('.jpg', img)  # Encode the frame to JPEG
            frame = buffer.tobytes()  # Convert to bytes
            yield (b'--frame\r\n'  # Yield the frame for streaming
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except Exception as e:
            print(f"Error in main loop: {e}")
            break
    cap.release()  # Release the camera

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/about_project")  # Route for the about project page
def about_project():
    return render_template("about_project.html")

@app.route("/about_us")  # Route for the about us page
def about_us():
    return render_template("about_us.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True,threaded=True)