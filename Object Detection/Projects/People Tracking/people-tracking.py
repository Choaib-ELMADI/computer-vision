from ultralytics import YOLO
from sort import *
import cvzone
import math
import cv2

targetClassNames = ["person"]
classNames = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# FOR PEOPLE GOING DOWN
limitsDown = [505, 500, 705, 500]
totalCountDown = []
lineDownColor = (0, 0, 255)
# FOR PEOPLE GOING UP
limitsUp = [130, 240, 330, 240]
totalCountUp = []
lineUpColor = (0, 255, 0)

mask = cv2.imread(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Projects/People Tracking/mask.png"
)
cap = cv2.VideoCapture(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Videos/people.mp4"
)

model = YOLO(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Yolo Weights/yolov8n.pt"
)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    _, frame = cap.read()
    frameRegion = cv2.bitwise_and(frame, mask)
    imgGraphics = cv2.imread(
        "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Projects/People Tracking/graphics.png",
        cv2.IMREAD_UNCHANGED,
    )
    frame = cvzone.overlayPNG(frame, imgGraphics, (240, 0))
    results = model(frameRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in targetClassNames:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                conf = math.floor(box.conf[0] * 100) / 100

                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    trackerResults = tracker.update(detections)

    # FOR PEOPLE GOING DOWN
    cv2.line(
        frame,
        (limitsDown[0], limitsDown[1]),
        (limitsDown[2], limitsDown[3]),
        lineDownColor,
        3,
    )
    # FOR PEOPLE GOING UP
    cv2.line(
        frame,
        (limitsUp[0], limitsUp[1]),
        (limitsUp[2], limitsUp[3]),
        lineUpColor,
        3,
    )

    # FOR PEOPLE GOING DOWN
    cvzone.putTextRect(
        frame,
        f"{len(totalCountDown)}",
        (535, 50),
        2.5,
        2,
        (0, 0, 255),
        (255, 255, 255),
        cv2.FONT_HERSHEY_PLAIN,
        5,
    )
    # FOR PEOPLE GOING UP
    cvzone.putTextRect(
        frame,
        f"{len(totalCountUp)}",
        (375, 50),
        2.5,
        2,
        (0, 255, 0),
        (255, 255, 255),
        cv2.FONT_HERSHEY_PLAIN,
        5,
    )

    for res in trackerResults:
        x1, y1, x2, y2, id = res
        x1, y1, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        cvzone.cornerRect(frame, (x1, y1, w, h), 10, 2, 1)
        cvzone.putTextRect(frame, f"{int(id)}", (x1, y1 - 5))

        cx, cy = int(x1 + w / 2), int(y1 + h / 2)
        cv2.circle(frame, (cx, cy), 3, (247, 127, 0), cv2.FILLED)

        # FOR PEOPLE GOING DOWN
        if (
            limitsDown[0] <= cx <= limitsDown[2]
            and limitsDown[1] - 15 <= cy <= limitsDown[1] + 15
        ):
            if totalCountDown.count(id) == 0:
                lineDownColor = (255, 0, 0)
                totalCountDown.append(id)
            else:
                lineDownColor = (0, 0, 255)

        # FOR PEOPLE GOING UP
        if (
            limitsUp[0] <= cx <= limitsUp[2]
            and limitsUp[1] - 15 <= cy <= limitsUp[1] + 15
        ):
            if totalCountUp.count(id) == 0:
                lineUpColor = (255, 0, 0)
                totalCountUp.append(id)
            else:
                lineUpColor = (0, 255, 0)

    cv2.imshow("People Tracking", frame)
    cv2.waitKey(1)
