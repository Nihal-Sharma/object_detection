from ultralytics import YOLO
import cv2
import keyboard
import time
import datetime

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)  # Replace "0" with the path to your video source or camera index

# Variables for FPS calculation
start_time = time.time()
frame_count = 0
 
while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame, show=True)

    if keyboard.is_pressed('q'):  # Press "q" to exit
        break

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Display FPS and live time on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, f'Live Time: {current_time}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Detection', frame)

cap.release()
cv2.destroyAllWindows()
