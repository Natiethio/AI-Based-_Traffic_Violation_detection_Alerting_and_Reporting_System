import datetime
import ultralytics
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
import time
import os
import json
import serial
from tracker import *

# Define the path to the model file
model_path = Path('yolov8n.pt')

# Load the YOLOv8 model
model = YOLO(model_path)


class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

vehicle_classes = ['car', 'truck', 'motorbike', 'bus']

tracker = Tracker()
count = 0

down = {}  # store all cars touching the red line and the locations
up = {}
overspeed = []
violatered = []
violatelane = []
counter_down = []  # stores id of all vehicles touching the red line first then the blue line
counter_up = []  # stores id of all vehicles touching the blue line first then the red line
lane_violation_ids = set()
# Create folders to save frames
if not os.path.exists('Over_Speeding_cars'):
    os.makedirs('Over_Speeding_cars')

if not os.path.exists('Red_lineviolated_cars'):
    os.makedirs('Red_lineviolated_cars')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1020, 500))

cap = cv2.VideoCapture('highway.mp4')

count = 0

# Initialize serial communication
# ser = serial.Serial('COM11', 9600)  



arduino_port = "COM9"  
baud_rate = 9600 

ser = serial.Serial(arduino_port, baud_rate, timeout=1)

time.sleep(2)

print("Connected to Arduino port:", arduino_port)

# Load light configuration
with open('light_status.json', 'r') as filelight:
    configlight = json.load(filelight)
light_status = configlight["light_status"]

# Load line configuration
config_file_path = 'configtest.json'
last_modified_time = os.path.getmtime(config_file_path)
with open(config_file_path, 'r') as file:
    config = json.load(file)

red_color = (0, 0, 255)  # Red color in BGR
blue_color = (255, 0, 0)  # Blue color in BGR
text_color = (255, 255, 255)  # White color in BGR
green_color = (0, 255, 0)  # Green color in BGR
text_colorcounter = (255, 255, 255)  # White color for text
yellow_color = (0, 255, 255)  # Yellow color for background
counter_bg_color = (0, 0, 255)  # Red color for background
offset = 7

def update_lines(config):
    global green_line_start_x, green_line_start_y, green_line_end_x, green_line_end_y
    global red_line_start_x, red_line_start_y, red_line_end_x, red_line_end_y
    global blue_line_start_x, blue_line_start_y, blue_line_end_x, blue_line_end_y
    
    green_line_start_x = config["speed_test_line"]["green_line_start"]["x"]
    green_line_start_y = config["speed_test_line"]["green_line_start"]["y"]
    green_line_end_x = config["speed_test_line"]["green_line_end"]["x"]
    green_line_end_y = config["speed_test_line"]["green_line_end"]["y"]
    blue_line_start_x = config["speed_test_line"]["blue_line_start"]["x"]
    blue_line_start_y = config["speed_test_line"]["blue_line_start"]["y"]
    blue_line_end_x = config["speed_test_line"]["blue_line_end"]["x"]
    blue_line_end_y = config["speed_test_line"]["blue_line_end"]["y"]
    red_line_start_x = config["red_light_line"]["red_line_start"]["x"]
    red_line_start_y = config["red_light_line"]["red_line_start"]["y"]
    red_line_end_x = config["red_light_line"]["red_line_end"]["x"]
    red_line_end_y = config["red_light_line"]["red_line_end"]["y"]

update_lines(config)

# List of vehicle classes to detect

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_modified_time = os.path.getmtime(config_file_path)
    if current_modified_time != last_modified_time:
        with open(config_file_path, 'r') as file:
            config = json.load(file)
        update_lines(config)
        last_modified_time = current_modified_time

    count += 1

    frame = cv2.resize(frame, (1020, 500))


    # Read light status from Arduino
    if ser.in_waiting > 0:
        light_status = ser.readline().decode('utf-8').strip()
        configlight["light_status"] = light_status
        with open('light_status.json', 'w') as file:
            json.dump(configlight, file, indent=4)

    # Update the red line color based on light status
    # current_red_color = green_color if light_status == "green" else red_color
    current_red_color = green_color if light_status == "green" else (yellow_color if light_status == "yellow" else red_color)


    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c in vehicle_classes:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) // 2)
        cy = int((y3 + y4) // 2)

        if green_line_start_y < (cy + offset) and green_line_start_y > (cy - offset):
            down[id] = time.time()

        if id in down:
            if blue_line_start_y < (cy + offset) and blue_line_start_y > (cy - offset):
                elapsed_time = (time.time() - down[id]) - 2
                if counter_down.count(id) == 0:
                    counter_down.append(id)
                    distance = 25
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6

                    if a_speed_kh > 15:
                        cv2.circle(frame, (cx, cy), 4, red_color, -1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), red_color, 2)

                        (w, h), _ = cv2.getTextSize(str(int(a_speed_kh)) + 'Km/h', cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (x4, y4 - h - 10), (x4 + w, y4), red_color, -1)
                        cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

                        current_time = datetime.datetime.now()
                        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                        frame_filename = f'Over_Speeding_cars/{id}_{formatted_time}.jpg'
                        frame_filename2 = f'Violations/{id}_{formatted_time}_s.jpg'
                        cv2.imwrite(frame_filename, frame)
                        cv2.imwrite(frame_filename2, frame)

                        # frame_filename = f'Over_Speeding_cars/{id}.jpg'
                        # cv2.imwrite(frame_filename, frame)

        if blue_line_start_y < (cy + offset) and blue_line_start_y > (cy - offset):
            up[id] = time.time()

        if id in up:
            if green_line_start_y < (cy + offset) and green_line_start_y > (cy - offset):
                elapsed1_time = (time.time() - up[id]) - 2
                if counter_up.count(id) == 0:
                    counter_up.append(id)
                    distance1 = 25
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6

                    if a_speed_kh1 > 15:
                        cv2.circle(frame, (cx, cy), 4, red_color, -1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), red_color, 2)
                        (w, h), _ = cv2.getTextSize(str(int(a_speed_kh1)) + 'Km/h', cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
                        cv2.rectangle(frame, (x4, y4 - h - 10), (x4 + w, y4), red_color, -1)
                        cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2, cv2.LINE_AA)

                        current_time = datetime.datetime.now()
                        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                        frame_filename = f'Over_Speeding_cars/{id}_{formatted_time}.jpg'
                        frame_filename2 = f'Violations/{id}_{formatted_time}_s.jpg'
                        cv2.imwrite(frame_filename, frame)
                        cv2.imwrite(frame_filename2, frame)

                        # frame_filename = f'Over_Speeding_cars/{id}.jpg'
                        # cv2.imwrite(frame_filename, frame)
                        
        if light_status == "red":
            if red_line_start_y < (cy + offset) and red_line_start_y > (cy - offset):
                # Check if the car is coming from the left side of the red line
                if cx < red_line_end_x:  # Car is to the left of the red_line_end_x
                    cv2.circle(frame, (cx, cy), 4, red_color, -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), red_color, 2)

                    (w, h), _ = cv2.getTextSize('Violate Red', cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x4, y4 - h - 10), (x4 + w, y4), red_color, -1)
                    cv2.putText(frame, 'Violate Red', (x4, y4 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                if id not in violatered:
                    violatered.append(id)

                    current_time = datetime.datetime.now()
                    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                    frame_filename = f'Red_lineviolated_cars/{id}_{formatted_time}.jpg'
                    frame_filename2 = f'Violations/{id}_{formatted_time}_r.jpg'
                    cv2.imwrite(frame_filename, frame)
                    cv2.imwrite(frame_filename2, frame)
                    # frame_filename = f'Red_lineviolated_cars/{id}.jpg'
                    # cv2.imwrite(frame_filename, frame)

        for lane in config['lane']['lanes']:
            lane_start_x = int(lane['lane_start']['x'])
            lane_start_y = int(lane['lane_start']['y'])
            lane_end_x = int(lane['lane_end']['x'])
            lane_end_y = int(lane['lane_end']['y'])

            if cx < red_line_end_x: 
            #  if lane_start_x <= cx <= lane_end_x and lane_start_y  <= cy <= lane_end_y :
              if (lane_start_x - offset) <= cx <= (lane_end_x + offset):
                cv2.circle(frame, (cx, cy), 4, yellow_color, -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), yellow_color, 2)
                (w, h), _ = cv2.getTextSize('Lane Violation', cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x4, y4 - h - 10), (x4 + w, y4), yellow_color, -1)
                cv2.putText(frame, 'Lane Violation', (x4, y4 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                frame_filename = f'Violations/{id}_{formatted_time}_l.jpg'

                if id not in violatelane:
                  cv2.imwrite(frame_filename, frame)
                  lane_violation_ids.add(frame_filename)
                  cv2.imwrite(frame_filename, frame)
                  violatelane.append(id)

            # Check if the car's center crosses the lane line
            # if lane_start_x <= cx <= lane_end_x and lane_start_y  <= cy <= lane_end_y :
            #     cv2.circle(frame, (cx, cy), 4, yellow_color, -1)
            #     cv2.rectangle(frame, (x3, y3), (x4, y4), yellow_color, 2)
            #     (w, h), _ = cv2.getTextSize('Lane Violation', cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
            #     cv2.rectangle(frame, (x4, y4 - h - 10), (x4 + w, y4), yellow_color, -1)
            #     cv2.putText(frame, 'Lane Violation', (x4, y4 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            #     current_time = datetime.datetime.now()
            #     formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
            #     frame_filename = f'Violations/{id}_{formatted_time}_l.jpg'
            #     cv2.imwrite(frame_filename, frame)

        for lane in config["lane"]["lanes"]:
            lane_start_x = lane["lane_start"]["x"]
            lane_start_y = lane["lane_start"]["y"]
            lane_end_x = lane["lane_end"]["x"]
            lane_end_y = lane["lane_end"]["y"]
            cv2.line(frame, (lane_start_x, lane_start_y), (lane_end_x, lane_end_y), yellow_color, 1)
            cv2.putText(frame, 'Lane', (lane_start_x, lane_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (green_line_start_x, green_line_start_y), (green_line_end_x, green_line_start_y), green_color, 1)
    cv2.putText(frame, ('Green Line'), (green_line_start_x, green_line_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (blue_line_start_x, blue_line_start_y), (blue_line_end_x, blue_line_start_y), blue_color, 1)
    cv2.putText(frame, ('Blue Line'), (blue_line_start_x + 10, blue_line_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.line(frame, (red_line_start_x, red_line_start_y), (red_line_end_x, red_line_start_y), current_red_color, 1)
    cv2.putText(frame, ('Light Status'), (10, red_line_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (10, 10), (260, 40), counter_bg_color, -1)
    cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colorcounter, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (760, 10), (1010, 40), counter_bg_color, -1)
    cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (765, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colorcounter, 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("frames", frame)

    if cv2.waitKey(0) & 0xFF == 27:
        break

ser.close()
cap.release()
out.release()
cv2.destroyAllWindows()

print("Lane Violations:")
for id in violatelane:
    print(f"Car ID {id} violated the lane")

# Display overspeed violations
print("\nOverspeed Violations:")
for id in overspeed:
    print(f"Car ID {id} was overspeeding")

# Display red light violations
print("\nRed Light Violations:")
for id in violatered:
    print(f"Car ID {id} violated the red light")

# Optionally save logs or images here

# Clear arrays
violatered.clear()
overspeed.clear()
violatelane.clear()
lane_violation_ids.clear()