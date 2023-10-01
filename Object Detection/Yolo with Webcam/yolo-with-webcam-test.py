from ultralytics import YOLO
import cvzone
import cv2

cap = cv2.VideoCapture(0)
cap.set(3, 640)  # 1280
cap.set(4, 480)  # 720

model = YOLO(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Yolo Weights/yolov8n.pt"
)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #! FOR OPENCV
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            #! FOR CVZONE
            x1, y1, w, h = box.xywh[0]
            x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            bbox = x1, y1, w, h
            print(x1, y1, w, h)
            cvzone.cornerRect(img, bbox)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
