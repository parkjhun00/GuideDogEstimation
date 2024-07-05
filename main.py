import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from ultralytics.engine.results import Results
from OneEuroFilter import OneEuroFilter
import numpy as np
import random
import pyqtgraph as pg
import csv
from pyqtgraph.Qt import QtCore
from collections import deque


# Load the YOLOv8 model
model = YOLO('628.pt')

# Open the video file
#video_path = "test2.avi"
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)
#Get the video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#Create a VideoWriter object
output_path = "test_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Create a CSV file to store the results
csv_filename = f"test.csv"
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Frame", "Angle"])

# Keypoint names definition
KEYPOINTS_NAMES_HANDLE = ["Handle_head", "Grip", ""]
KEYPOINTS_NAMES_DOG = ["Leg(L)", "Head", "Leg(R)", "Tail"]

# Get middle point of Leg(L) and Leg(R)
def get_middle_point(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2

def calculate_angle(vector_1, vector_2):
    #Calculate the angle between two vectors
    angle = np.arctan2(vector_2[1], vector_2[0]) - np.arctan2(vector_1[1], vector_1[0])
    # Convert the angle from radians to degrees
    angle = np.degrees(angle) + 180
    return angle

# Initialize variables
current_angle = 0
previous_angle = 0 
frame_number = 0


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Increment the frame number
        frame_number += 1
        
        # Run YOLOv8 inference on the frame
        results = model.predict(frame)
        annotatedFrame = results[0].plot()

        # Get object names, bounding box coordinates, and keypoints
        names = results[0].names
        classes = results[0].boxes.cls
        boxes = results[0].boxes

        # List to store detected object names
        detected_object = []

        for box, cls in zip(boxes, classes):
            name = names[int(cls)]
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            detected_object.append(name)

        #print(detected_object)
        detected_objects_count = len(detected_object)\
        
        dog_vector = np.zeros(shape=2)
        handle_vector = np.zeros(shape=2)
        angle_calculated = False  

        for i in range(detected_objects_count):

            # Initialize keypoint coordinates
            leg_l = leg_r = head = grip = head_2 = tail = None
            best_handle_score = 0
            best_dog_score = 0

            if len(results[0].keypoints) == 0:
                continue
            keypoints = results[0].keypoints

            if results[0].keypoints.conf is not None:
                confs = keypoints.conf[i].tolist()
                xys = keypoints.xy[i].tolist()

                # Assign keypoint names based on detected object
                if detected_object[i] == "Handle":
                    keypoint_names = KEYPOINTS_NAMES_HANDLE
                    if confs[0] > best_handle_score:
                        best_handle_score = confs[0]   
                    else:
                        continue

                elif detected_object[i] == "Dog":
                    keypoint_names = KEYPOINTS_NAMES_DOG
                    if confs[0] > best_dog_score:
                        best_dog_score = confs[0]
                    else:
                        continue              

                # Initialize keypoint coordinates
                leg_l = leg_r = head = grip = head_2 = None
                # Initialize vectors

                for index, keypoint in enumerate(zip(xys, confs)):
                    score = keypoint[1]
                    if score < 0.5:
                        continue

                    x = int(keypoint[0][0])
                    y = int(keypoint[0][1])

                    if keypoint_names[index] == "Leg(L)":
                        leg_l = keypoint[0]
                    elif keypoint_names[index] == "Leg(R)":
                        leg_r = keypoint[0]
                    elif keypoint_names[index] == "Head":
                        head = keypoint[0] 
                    elif keypoint_names[index] == "Handle_head":
                        grip = keypoint[0]
                    elif keypoint_names[index] == "Grip":
                        head_2 = keypoint[0]    
                    elif keypoint_names[index] == "Tail":
                        tail = keypoint[0]   

                    print(
                        f"Keypoint Name={keypoint_names[index]}, X={x}, Y={y}, Score={score:.3}"
                    )

                    # Draw lines if all keypoints are detected
                    if leg_l and leg_r and head and tail:
                        # Calculate midpoint between Leg(L) and Leg(R)
                        mid_x, mid_y = get_middle_point(int(leg_l[0]), int(leg_l[1]), int(leg_r[0]), int(leg_r[1]))
                        # Draw line from midpoint to tail
                        cv2.line(annotatedFrame, (mid_x, mid_y), (int(tail[0]), int(tail[1])), (255, 255, 0), 2)

                        # Calculate the vector from the midpoint to the Head
                        dog_vector = np.array([int(tail[0]) - mid_x, int(tail[1])- mid_y])
                        print(f"Vector u: {dog_vector}")

                    if grip and head_2:
                        cv2.line(annotatedFrame, (int(grip[0]), int(grip[1])), (int(head_2[0]), int(head_2[1])), (0, 255, 255), 2)

                        # Calculate the vector from the Grip to the Head_2
                        handle_vector = np.array([int(grip[0]) - int(head_2[0]), int(grip[1]) - int(head_2[1])])
                        print(f"Vector v: {handle_vector}")

                        if not np.array_equal(dog_vector, np.zeros(2)) and not np.array_equal(handle_vector, np.zeros(2)):
                            print("\nCalculating angle:")
                            current_angle = calculate_angle(dog_vector, handle_vector)
                            angle_calculated = True
                            print(f"\nFinal angle between vectors: {current_angle:.2f} degrees")


                    # Draw a purple square
                    annotatedFrame = cv2.rectangle(
                        annotatedFrame,
                        (x, y),
                        (x + 3, y + 3),
                        (255, 0, 255),
                        cv2.FILLED,
                        cv2.LINE_AA,
                    )

                    # Draw the keypoint name
                    annotatedFrame = cv2.putText(
                        annotatedFrame,
                        keypoint_names[index],
                        (x + 5, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=(255, 0, 255),
                        thickness=1,
                        lineType=cv2.LINE_AA,
                    )

        # Draw force bar
        #draw_force_bar(annotatedFrame, force)

        # Draw the angle on the annotated frame
        cv2.putText(annotatedFrame, f"Angle: {current_angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if abs(current_angle - previous_angle) > 0.01:  #If the angle has changed by more than 0.01 degrees
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([frame_number, f"{current_angle:.2f}"])
            previous_angle = current_angle  # Update the previous angle

        # Draw the force arrow on the annotated frame
        #draw_force_arrow(annotatedFrame, grip, head_2, force, scale=10)

        # Write the annotated frame to the output video
        out.write(annotatedFrame)

        # Display the annotated frame
        cv2.imshow("Guide Dog Estimation", annotatedFrame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
