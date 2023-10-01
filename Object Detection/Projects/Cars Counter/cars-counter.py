from ultralytics import YOLO
from sort import *
import cvzone
import math
import cv2

targetClassNames = ["car", "motorbike", "bus", "truck"]
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
limits = [360, 297, 673, 297]

mask = cv2.imread(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Projects/Cars Counter/mask.png"
)
cap = cv2.VideoCapture(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Videos/cars.mp4"
)

model = YOLO(
    "C:/Users/Choaib ELMADI/Downloads/D.I.F.Y/Electronics/Computer Vision/Object Detection/Yolo Weights/yolov8n.pt"
)
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

while True:
    _, frame = cap.read()
    frameRegion = cv2.bitwise_and(frame, mask)
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
    cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

    for res in trackerResults:
        x1, y1, x2, y2, id = res
        x1, y1, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        cvzone.cornerRect(frame, (x1, y1, w, h), 10, 2, 1)
        cvzone.putTextRect(frame, f"{int(id)}", (x1, y1 - 5))

    cv2.imshow("Cars Counter", frame)
    cv2.waitKey(1)
