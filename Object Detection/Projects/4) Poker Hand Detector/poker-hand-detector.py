from ultralytics import YOLO
import findPokerHand
import cvzone
import math
import cv2

classNames = [
    "10C",
    "10D",
    "10H",
    "10S",
    "2C",
    "2D",
    "2H",
    "2S",
    "3C",
    "3D",
    "3H",
    "3S",
    "4C",
    "4D",
    "4H",
    "4S",
    "5C",
    "5D",
    "5H",
    "5S",
    "6C",
    "6D",
    "6H",
    "6S",
    "7C",
    "7D",
    "7H",
    "7S",
    "8C",
    "8D",
    "8H",
    "8S",
    "9C",
    "9D",
    "9H",
    "9S",
    "AC",
    "AD",
    "AH",
    "AS",
    "JC",
    "JD",
    "JH",
    "JS",
    "KC",
    "KD",
    "KH",
    "KS",
    "QC",
    "QD",
    "QH",
    "QS",
]

cap = cv2.VideoCapture(
    1
    # "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Images/mypoker.jpg"
)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Projects/4) Poker Hand Detector/playing-cards.pt"
)

while True:
    _, img = cap.read()
    results = model(img, stream=True)
    hand = []

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
                img, f"{classNames[clsIndex]}", (x1 + 3, y1 - 5), 1.25, 1, offset=4
            )

            if conf > 0.4:
                hand.append(classNames[clsIndex])

    hand = list(set(hand))
    if len(hand) == 5:
        res = findPokerHand.findPokerHand(hand)
        cvzone.putTextRect(img, f"{res}", (40, 30), 1.5, 1, offset=6)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
