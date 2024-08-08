import datetime
import ultralytics
import numpy as np
import pandas as pd
import cv2
import time
import os
import json
import requests
import jwt 
import time
import threading
import json
import base64
import uvicorn
import mysql.connector
import re


from pathlib import Path
from ultralytics import YOLO
from tracker import*
from Anprtest import*
from fastapi import FastAPI
from typing import Optional
from mysql.connector import Error
from queue import Queue, Empty
from PIL import Image

#JWT configuration parameters 
header = {  
"alg": "HS256",  
"typ": "JWT"  
}  
secret = "1YQN8khetY+tURx6va71D4DZGJJ+tsUWis8F/fgkB3rR9ovUoZ6XyBkTdx/ca01v"   


unsent_buffer = Queue()              # Unset packate buffer
global response
TIMEOUT = 2                     # Database wait time
REQUEST_TIME = 15
config = {}
CMD_RCV_ENDPOINT = ''
SERVER_ENDPOINT = ''
HOST = ''

#API SERVICE 
# app = FastAPI()
# @app.post("/cmd")
# def read_message(message: Optional[str] = None):
#     if message:
#         try:  
#             decoded_jwt = jwt.decode(message, secret, algorithms=['HS256'],verify=True)  
#             print("This is the data\n",decoded_jwt)  
#         except jwt.exceptions.InvalidSignatureError:  
#             return {"message": f"Invaild Signature Used"}
#         return {"message": f"Command received"}
#     else:
#         return {"message": "Command not received"}

# def start_api():
#     # Start Uvicorn server programmatically
#     print("Starting API service ....")
#     uvicorn.run(app, host=HOST, port=8000)


#Thread element to send unsent pck to the central server
def send_data_in_buffer():
    print("Packet Handler is running ...")
    while True:
        try:
            pck = unsent_buffer.get(timeout=1)  # Wait for up to 1 second for a packet
            try:
                response = requests.get(SERVER_ENDPOINT, data=pck, timeout=TIMEOUT)
                if response.status_code == 200:
                    print(f"Packet sent successfully: {pck}")
                else:
                    print(f"Failed to send packet: {pck}, status code: {response.status_code}")
                    unsent_buffer.put(pck)  # Requeue the packet for retry
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                unsent_buffer.put(pck)  # Requeue the packet for retry
        except Empty:
            pass  # No packets to send, continue looping

def add_pck_to_queue(packet):
    unsent_buffer.put(packet)

def get_iso_timestamp():
    try:
        now = datetime.now()
        iso_timestamp = now.isoformat()
        return iso_timestamp
    except AttributeError as e:
        print(f"AttributeError: {e}")
        return None
    except TypeError as e:
        print(f"TypeError: {e}")
        return None
    except Exception as e:
        # Handle any other unforeseen exceptions
        print(f"Unexpected error: {e}")
        return None


def merge_photos(img1, img2, output_path):
    # Open the images
    image1 = Image.open(img1)
    image2 = Image.open(img2)

    # Get the sizes of the images
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Calculate the new size for the second image
    new_width2 = int(width1)
    new_height2 = int((new_width2 / width2) * height2)

    # Resize the second image
    image2 = image2.resize((new_width2, new_height2), Image.LANCZOS)

    # Calculate the size of the new image
    result_width = max(width1, new_width2)
    result_height = height1 + new_height2

    # Create a new blank image with the appropriate size
    result = Image.new('RGB', (result_width, result_height))

    # Paste the images into the result image
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))
    result.save(output_path)

#Violation Packate Cooking 
def violation_report(vtype,licensePlate,timeStamp,imageFile,croppedImage):
  try:
    payload = {  
    "sub": config['deviceid'], 
    "type" : vtype,
    "lp" : licensePlate,
    "time": timeStamp,
    "location": config['location']
    }  

    merge_photos(imageFile,croppedImage,imageFile)
    
    file = {
        'image_file': 
        open(imageFile, 'rb'),
    }

    encoded_jwt = jwt.encode(payload, secret, algorithm='HS256', headers=header) 
    data = {
      "pck": encoded_jwt
    }

    response = requests.post(SERVER_ENDPOINT, data = data , files = file )
    print( 'Response from server', response.status_code , response)
  except Error as e :
    print("Waiting for api service to run in the central ...")
    time.sleep(2)
    violation_report(vtype,licensePlate,timeStamp,imageFile,croppedImage)

def violation_report_lp_not_present(vtype,licensePlate,timeStamp,imageFile):
  try:
    payload = {  
    "sub": config['deviceid'], 
    "type" : vtype,
    "lp" : "NULL",
    "time": timeStamp,
    "location": config['location']
    }  


    
    file = {
        'image_file': 
        open(imageFile, 'rb'),
    }

    encoded_jwt = jwt.encode(payload, secret, algorithm='HS256', headers=header) 
    data = {
      "pck": encoded_jwt
    }

    response = requests.post(SERVER_ENDPOINT, data = data , files = file )
    print( 'Response from server', response.status_code , response)
  except Error as e :
    print("Waiting for api service to run in the central ...")
    time.sleep(2)
    violation_report(vtype,licensePlate,timeStamp,imageFile)


#Flagged Packate Cooking 

def flagged_report(licensePlate,imageFile,croppedImage):
    try:

        connection = mysql.connector.connect(
        host='localhost',
        port=3307,
        database='violation',
        user='root',
        password=''
        )

        if connection.is_connected():
            print("Connected to MySQL database")
            cursor = connection.cursor()
            cursor.execute("SELECT plate_number FROM flagged_car")
            result = cursor.fetchall()
            for row in result:
                # print(row[0])
                if(str(row[0]) == licensePlate):
                    licensePlate = row[0]
                    query = "SELECT flagged_id FROM flagged_car WHERE plate_number = %s"
                    cursor.execute(query, (licensePlate,))
                    result2 = cursor.fetchall()
                    for row in result2:
                        print(str(row[0]))
                        print("=========Sending Flagged======")
                        payload = {  
                            "type": "flagged",
                            "id" : row[0],     
                            "location": config['location']
                        }
                        merge_photos(imageFile,croppedImage,imageFile)
                        file = {
                            'image_file': 
                            open(imageFile, 'rb'),
                        }
                        encoded_jwt = jwt.encode(payload, secret, algorithm='HS256', headers=header) 
                        data = {
                            "pck": encoded_jwt
                        }
                        try:
                            response = requests.post(FLAGGED, data=data , files = file )
                            print("system response " ,response.status_code, response)
                            return
                        except:
                            add_pck_to_queue(data)


             
    except Error as e:
        print(e)

def is_flagged(licensePlate,imageFile,croppedImage):
 try: 
   connection = mysql.connector.connect(
            host='localhost',
            port=3307,
            database='violation',
            user='root',
            password=''
    )

   if connection.is_connected():
      print("Connected to MySQL database")
      cursor = connection.cursor()
      cursor.execute('SELECT plate_number FROM flagged_car')
      result = cursor.fetchall()

      for row in result:
        print(row[0])
        if(str(row[0]) == licensePlate):
            cursor.execute('SELECT flagged_id FROM flagged_car WHERE plate_number = %s', (licensePlate))
            
            payload = {  
                "type": "flagged",
                "id" : row[0],     
                "location": config['location']
                }
            merge_photos(imageFile,croppedImage,imageFile)
            file = {
                'image_file': 
                open(imageFile, 'rb'),
            }

            encoded_jwt = jwt.encode(payload, secret, algorithm='HS256', headers=header) 
            data = {
                "pck": encoded_jwt
            }
            try:
                response = requests.post(SERVER_ENDPOINT, data=data)
            except:
                add_pck_to_queue(data)
          
 except Error as e :
    print(e)
    # time.sleep(2)
    # flagged_report(licensePlate,imageFile,croppedImage)


#Recieve new flagged license plates 
def push_to_db(entries):
    try: 
        connection = mysql.connector.connect(
                host='localhost',
                port=3307,
                database='violation',
                user='root',
                password=''
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            cursor = connection.cursor()
            insert_query = "INSERT INTO flagged_car (flagged_id, plate_number) VALUES (%s, %s)"
            for item in entries:
                cursor.execute(insert_query, item)
            connection.commit()
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")

def extract_pairs(text):
    if not text:
        return []

    pairs = text.split(',')
    extracted_elements = []
    pattern = re.compile(r'^(\d+):([^,]+)$')

    for pair in pairs:
        match = pattern.match(pair)
        if match:
            extracted_elements.append((match.group(1), match.group(2)))
        else:
            print(f"Invalid pair: {pair}")
            continue

    return extracted_elements

def receive_flagged():
    while True:
        print("== Recieving flagged plates from central == ")
        try : 
            response = requests.get(LIST)
            print(response.status_code)
            data = response.json()
            print(data['content']) 
            flagged = extract_pairs(data['content'])
            if len(flagged) != 0:
                push_to_db(flagged)
            time.sleep(REQUEST_TIME)
        except Error as e :
            print(f"Error while Recieving flagged update trying again .....")
    
def traffic_violation_detection():
    print("VIOLATION detection running ..... ")
    model_path = Path('yolov8n.pt')
    if not model_path.exists():
        raise FileNotFoundError(f'Model file {model_path} does not exist')

    model = YOLO(model_path)

    class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
                  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    vehicle_classes = ['car', 'truck', 'motorbike', 'bus']

    tracker=Tracker()
    anpr = Anprtest()

    count=0

    down = {}  # store all cars touching the red line and the locations
    up = {}
    overspeed = []
    violatered = []
    violatelane = []
    previous_positions = {}
    violation = []
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

    cap = cv2.VideoCapture('lane_red.mp4')
    # if not cap.exists():
    #     raise FileNotFoundError(f'Model file {cap} does not exist')
    count = 0

    # Load light configuration
    with open('light_status.json', 'r') as filelight:
        configlight = json.load(filelight)
    light_status = configlight["light_status"]

    # Load line configuration
    config_file_path = 'configtest.json'
    # config_file_path = 'config2.json'
    last_modified_time = os.path.getmtime(config_file_path)
    with open(config_file_path, 'r') as file:
        config = json.load(file)
    debug = False
    red_color = (0, 0, 255)  # Red color in BGR
    blue_color = (255, 0, 0)  # Blue color in BGR
    text_color = (255, 255, 255)  # White color in BGR
    green_color = (0, 255, 0)  # Green color in BGR
    black_color = (0, 0, 0)  # Green color in BGR
    text_colorcounter = (255, 255, 255)  # White color for text
    yellow_color = (0, 255, 255)  # Yellow color for background
    counter_bg_color = (0, 0, 255)  # Red color for background
    offset = 7

    light_change_time = time.time()

    def update_lines(config):
        global green_line_start_x, green_line_start_y, green_line_end_x, green_line_end_y
        global red_line_start_x, red_line_start_y, red_line_end_x, red_line_end_y
        global licenseplate_line_start_x, licenseplate_line_start_y, licenseplate_line_end_x, licenseplate_line_end_y
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

        licenseplate_line_start_x = config["licenseplate_line"]["licenseplate_line_start"]["x"]
        licenseplate_line_start_y = config["licenseplate_line"]["licenseplate_line_start"]["y"]
        licenseplate_line_end_x = config["licenseplate_line"]["licenseplate_line_end"]["x"]
        licenseplate_line_end_y = config["licenseplate_line"]["licenseplate_line_end"]["y"]

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
        
        # Change light status every 15 seconds for testing
        if time.time() - light_change_time > 15:
            light_status = "green" if light_status == "red" else "red"
            configlight["light_status"] = light_status
            light_change_time = time.time()
            with open('light_status.json', 'w') as file:
                json.dump(configlight, file, indent=4)

        # Update the red line color based on light status
        current_red_color = green_color if light_status == "green" else red_color

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
            cars = {}

            elapsed_time = (time.time() - cars[id])-10

            if green_line_start_y < (cy + offset) and green_line_start_y > (cy - offset):
                down[id] = time.time()
           
            if id in down:
                if blue_line_start_y < (cy + offset) and blue_line_start_y > (cy - offset):
                    elapsed_time = (time.time() - down[id])-2
                    if counter_down.count(id) == 0:
                        counter_down.append(id)
                        distance = 50 #test with 10m
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 3.6

                        if a_speed_kh > 15:
                            print("=============overspeeding detected=============")
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
                            # if y4 > y3 and x4 > x3:
                            frame_copy = frame[y3:y4, x3:x4]
                            # if frame is not None or frame.size != 0:
                            licensePlate,croppedImage = anpr.test(frame_copy)
                            if licensePlate is None or croppedImage is None:
                                print("No license plate detected.")
                            else : 
                                violation_report("overspeeding",licensePlate,formatted_time,frame_filename2,croppedImage)
  
            if light_status == "red":
              if red_line_start_y < (cy + offset) and red_line_start_y > (cy - offset):
                # Check if the car is coming from the left side of the red line
                if cx < red_line_end_x:  # Car is to the left of the red_line_end_x
                    print("=============Redlight violation detected=============")
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
                    # if y4 > y3 and x4 > x3:
                    frame_copy = frame[y3:y4, x3:x4]
                    # if frame is not None or frame.size != 0:
                    
                    licensePlate,croppedImage = anpr.test(frame_copy)
                    if licensePlate is None or croppedImage is None:
                        print("No license plate detected.")
                    else : 
                        violation_report("redlight",licensePlate,formatted_time,frame_filename2,croppedImage)
                    print("READ LICENCEPLATE : ", licensePlate)
                    # test(licensePlate,frame_filename)
                     

            # Licenseplate flagged check 
            if licenseplate_line_start_y < (cy + offset) and licenseplate_line_start_y > (cy - offset):
                print("licensePlate checking ")
                if id not in violatered or id not in violatelane or id not in overspeed:
                    current_time = datetime.datetime.now()
                    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                    frame_copy = frame[y3:y4, x3:x4]
                    frame_filename2 = f'Violations/{id}_{formatted_time}_f.jpg'
                    cv2.imwrite(frame_filename2, frame)
                    licensePlate,croppedImage = anpr.test(frame_copy)
                    # cv2.imshow("licence check " , frame_copy)
                    if licensePlate is None or croppedImage is None:
                        print("No license plate detected.")
                    else:
                        print("===== Read License Plate =====" , licensePlate)
                        flagged_report(licensePlate,frame_filename2,croppedImage)
                        

            # Check for lane violations
            center_lane = config['lane']['lanes'][0]
            center_start_x = center_lane['lane_start']['x']
            center_end_x = center_lane['lane_end']['x']
            if id in previous_positions:
                prev_cx, prev_cy = previous_positions[id]
                if (prev_cx < center_start_x and cx >= center_start_x) or (prev_cx > center_start_x and cx <= center_start_x):
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), yellow_color, 2)
                    (w, h), _ = cv2.getTextSize('Lane Violation', cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
                    cv2.rectangle(frame, (x4, y4 - h - 10), (x4 + w, y4), yellow_color, -1)
                    cv2.putText(frame, 'Lane Violation', (x4, y4 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
                    current_time = datetime.datetime.now()
                    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
                    frame_filename = f'Violations/{id}_{formatted_time}_l.jpg'
                    frame_filename2 = f'Violations/{id}_{formatted_time}_l.jpg'
                    frame_copy = frame[y3:y4, x3:x4]
                    cv2.imwrite(frame_filename, frame)
                    cv2.imwrite(frame_filename, frame_copy)
                    cv2.imshow("image",frame_copy)
                    licensePlate,croppedImage = anpr.test(frame_copy)
                    if licensePlate is None or croppedImage is None:
                        violation_report_lp_not_present("lane",licensePlate,formatted_time,frame_filename2)
                        print("No license plate detected.")
                    else : 
                        violation_report("lane violation",licensePlate,formatted_time,frame_filename,croppedImage)
                    if id not in violatelane:
                        violatelane.append(id)
            previous_positions[id] = (cx, cy)


        cv2.line(frame, (green_line_start_x, green_line_start_y), (green_line_end_x, green_line_start_y), green_color, 1)
        cv2.putText(frame, ('Green Line'), (green_line_start_x, green_line_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.line(frame, (blue_line_start_x, blue_line_start_y), (blue_line_end_x, blue_line_start_y), blue_color, 1)
        cv2.putText(frame, ('Blue Line'), (blue_line_start_x + 10, blue_line_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


        cv2.line(frame, (red_line_start_x, red_line_start_y), (red_line_end_x, red_line_start_y), current_red_color, 1)
        cv2.putText(frame, ('Traffic Light'), (10, red_line_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.line(frame, (licenseplate_line_start_x, licenseplate_line_start_y), (licenseplate_line_end_x, licenseplate_line_start_y), green_color, 3)
        cv2.putText(frame, ('License plate detection'), (10, licenseplate_line_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        for lane in config["lane"]["lanes"]:
            lane_start_x = lane["lane_start"]["x"]
            lane_start_y = lane["lane_start"]["y"]
            lane_end_x = lane["lane_end"]["x"]
            lane_end_y = lane["lane_end"]["y"]
            cv2.line(frame, (lane_start_x, lane_start_y), (lane_end_x, lane_end_y), yellow_color, 3)
            cv2.putText(frame, 'Lane', (lane_start_x, lane_start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


        cv2.rectangle(frame, (10, 10), (260, 40), counter_bg_color, -1)
        cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colorcounter, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (760, 10), (1010, 40), counter_bg_color, -1)
        cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (765, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_colorcounter, 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("frames", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # After the video processing loop

    # Display lane violations
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

    # Clear arrays
    violatered.clear()
    overspeed.clear()
    violatelane.clear()
    lane_violation_ids.clear()
                      
with open("device.json", "r") as f: # Initial configurations
    config = json.load(f)

#MAIN SERVICE 
CMD_RCV_ENDPOINT = config['cmd_api']
SERVER_ENDPOINT = config['server_gate']
HOST  = config['host']
FLAGGED =  config['flagged_gate']
LIST =  config['flagged_list']

# flagged_report('AP40AR0658','violations/2_2024_07_24_13_20_56_f.jpg','violations/2_2024_07_24_13_17_08_f.jpg')

    # receive_flagged()
flagged_handler_thread = threading.Thread(target=receive_flagged, daemon=True)
flagged_handler_thread.start()

violation_handler_thread = threading.Thread(target=traffic_violation_detection, daemon=True)
violation_handler_thread.start()

# packet_handler_thread = threading.Thread(target=send_data_in_buffer, daemon=True)
# packet_handler_thread.start()
# is_flagged('23970')
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Main thread interrupted and exiting")