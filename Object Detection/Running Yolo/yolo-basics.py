from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
results = model(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Running Yolo/Images/2.png",
    show=True,  #! Show the output image
)
cv2.waitKey(0)  #! Keep image opened until the user presses a key
