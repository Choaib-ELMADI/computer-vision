from ultralytics import YOLO
import cvzone
import math
import cv2

classNames = [
    "Excavator",
    "Gloves",
    "Hardhat",
    "Ladder",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "SUV",
    "Safety Cone",
    "Safety Vest",
    "bus",
    "dump truck",
    "fire hydrant",
    "machinery",
    "mini-van",
    "sedan",
    "semi",
    "trailer",
    "truck and trailer",
    "truck",
    "van",
    "vehicle",
    "wheel loader",
]

cap = cv2.VideoCapture(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Videos/ppe-1.mp4"
)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Projects/PPE Detection/ppe.pt"
)

while True:
    _, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.floor(box.conf[0] * 100) / 100

            cls = box.cls[0]
            clsIndex = int(cls)
            cvzone.putTextRect(
                img,
                f"{classNames[clsIndex]} {conf}",
                (max(5, x1), max(35, y1 - 20)),
                1,
                1,
            )

    cv2.imshow("PPE Detection", img)
    cv2.waitKey(1)
