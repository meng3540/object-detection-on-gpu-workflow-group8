from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="cpu")

    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

