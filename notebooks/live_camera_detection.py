from ultralytics import YOLO
import cv2

# Load your custom-trained model
model = YOLO(r"C:\Users\chape\OneDrive\Bureau\best.pt")  # Make sure best.pt is in the same folder or give full path

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 prediction on the frame
    results = model.predict(source=frame, conf=0.3, verbose=False)

    # Draw annotations on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 - Webcam", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
