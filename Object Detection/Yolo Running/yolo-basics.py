from ultralytics import YOLO
import cv2

model = YOLO(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Yolo Weights/yolov8n.pt"
)

results = model(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Images/cars.png",
    show=True,  #! Show the output image ==> Classification and confidence level of detected objects
)
cv2.waitKey(0)  #! Keep image opened until the user presses a key
