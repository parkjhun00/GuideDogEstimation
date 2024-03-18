import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results
from OneEuroFilter import OneEuroFilter

# Load the YOLOv8 model
#model = YOLO('yolov8n.pt')
model = YOLO('228v1.pt')

# Open the video file
video_path = "test.mp4"
#video_path = "test2.avi"
cap = cv2.VideoCapture(video_path)

# keypointの位置毎の名称定義
KEYPOINTS_NAMES_HANDLE = [
    "Head",  # 0
    "Grip",  # 1
]

KEYPOINTS_NAMES_DOG = [
    "head",  # 0
    "Leg(L)",  # 1
    "Leg(R)",  # 2
]

#  OneEuroFilterの設定
config = {
    'freq': 120,       # Hz
    'mincutoff': 1.0,  # Hz
    'beta': 0.1,       
    'dcutoff': 1.0    
    }

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        #Run YOLOv8 inference on the frame
        results = model(frame)
        annotatedFrame = results[0].plot()

        # 検出オブジェクトの名前、バウンディングボックス座標を取得
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes

        # 検出オブジェクトの名前を格納するリスト
        detected_object = []

        #print(classes)

        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            detected_object.append(name) # 検出オブジェクトの名前をリストに追加
            print(detected_object)
        
        detected_objects_count = len(detected_object)

        for i in range(detected_objects_count):
            if len(results[0].keypoints) == 0:
                continue

            keypoints = results[0].keypoints
            if results[0].keypoints.conf is not None:
                confs = keypoints.conf[i].tolist()  # 推論結果:1に近いほど信頼度が高い
            xys = keypoints.xy[i].tolist()  # 座標

            for index, keypoint in enumerate(zip(xys, confs)):
                score = keypoint[1]

                # スコアが0.6以下なら描画しない
                if score < 0.6:
                    continue

                x = int(keypoint[0][0])
                y = int(keypoint[0][1])

                # Create OneEuroFilter 
                x_filter = OneEuroFilter(**config)
                y_filter = OneEuroFilter(**config)

                # Apply the filter to x and y coordinates
                x_filtered = x_filter(x)
                y_filtered = y_filter(y)

                # Update the x and y values with the filtered values
                x = x_filtered
                y = y_filtered

                # Print the keypoint name, x and y coordinates, and score
                print(
                    f"Keypoint Name={KEYPOINTS_NAMES_HANDLE[index]}, X={x}, Y={y}, Score={score:.4}"
                )
                # 紫の四角を描画
                annotatedFrame = cv2.rectangle(
                    annotatedFrame,
                    (x, y),
                    (x + 3, y + 3),
                    (255, 0, 255),
                    cv2.FILLED,
                    cv2.LINE_AA,
                )
                # キーポイントの部位名称を描画
                annotatedFrame = cv2.putText(
                    annotatedFrame,
                    KEYPOINTS_NAMES_HANDLE[index],
                    (x + 5, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.3,
                    color=(255, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotatedFrame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()