import cv2
import pytesseract
import re
import string
import numpy as np
from ultralytics import YOLO
import requests
import matplotlib.pyplot as plt

class Anprtest:
    debug = False 
    # Specify Tesseract executable location
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    model = YOLO(r'runs\detect\train\weights\best.pt')  # Corrected the path

    # cap = cv2.VideoCapture("demo.mp4")  # Change to your video file path

    detected_plates = {}
    
    # Function to sanitize text for filenames
    def sanitize_text(self,text):
        text = re.sub(r'[-_\s]', '', text)
        text = re.sub(r'[^a-zA-Z0-9]', '', text)
        return text[:255]  # Limit the filename length to 255 characters
    def test(self,frame):
        output = False;
        if self.debug:
            cv2.imshow("Captured Frame", frame)
            cv2.waitKey(0)
        results = self.model(frame)
        print("READING LICENCE PLATe ...........................")
        if output:
            cv2.imwrite('doc/vehiclecropped.jpg', frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                width = x2 - x1
                height = y2 - y1
                area = width * height
                print(area)
                if area > 3000:
                    print(f"Skipping large detected area: {area} (threshold: {3000})")
                    continue
                cropped_image = frame[y1:y2, x1:x2]
                if output:
                    cv2.imwrite('doc/platecropped.jpg', cropped_image)
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                if output:
                    cv2.imwrite('doc/gray.jpg', gray)
                gray = cv2.resize( gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
                if output:
                    cv2.imwrite('doc/resized.jpg', gray)
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                gray = cv2.medianBlur(gray, 3)
                if output:
                    cv2.imwrite('doc/blur.jpg', gray)
                # perform otsu thresh (using binary inverse since opencv contours work better with white text)
                ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
                if output:
                    cv2.imwrite('doc/threshold.jpg', ret)
                # cv2.imshow("Otsu", thresh)
                # cv2.waitKey(0)
                rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                if output:
                    cv2.imwrite('doc/morph_rect.jpg', rect_kern)
                # apply dilation 
                dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
                if output:
                    cv2.imwrite('doc/morph_dil.jpg', dilation)
                #cv2.imshow("dilation", dilation)
                #cv2.waitKey(0)
                # find contours
                try:
                    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if output:
                        cv2.imwrite('doc/contours.jpg', contours)
                except:
                    ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
                print("Number of contours found:", len(sorted_contours))
                # create copy of image
                im2 = gray.copy()
                if self.debug:
                    cv2.imshow("Image 2", im2)

                plate_num = ""
                # loop through contours and find letters in license plate
                for cnt in sorted_contours:
                    x,y,w,h = cv2.boundingRect(cnt)
                    height, width = im2.shape
                    # print(w,h)
                    # if height of box is not a quarter of total height then skip
                    # if height / float(h) > 6: continue
                    # ratio = h / float(w)
                    # # if height to width ratio is less than 1.5 skip
                    # if ratio < 1.5: continue
                    area = h * w
                    # if width is not more than 25 pixels skip
                    # if width / float(w) > 15: continue
                    # if area is less than 100 pixels skip
                    if area < 100: continue
                    # draw the rectangle
                    rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
                    if self.debug:
                        cv2.imshow("ROI", rect)
                    roi = thresh[y-5:y+h+5, x-5:x+w+5]
                    if roi is None or roi.size == 0:
                        print(f"Error: ROI is empty for contour at ({x}, {y}, {w}, {h})")
                        break
                    roi = cv2.bitwise_not(roi)
                    if output:
                        cv2.imwrite('doc/roi_bitwise.jpg', roi)
                    roi = cv2.medianBlur(roi, 5)
                    if output:
                        cv2.imwrite('doc/median.jpg', roi)
                    if self.debug:
                        cv2.imshow("ROI after bitwise", roi)
                        cv2.waitKey(0)
                    text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3',lang="eng").strip()
                    if self.debug:
                        print('Tesseract text read : ',text)
                    plate_num += text
               
                cropped_image_path = f'detected_license/{plate_num}.jpg'
                print("cropped image path : ", cropped_image_path)
                cv2.imwrite(cropped_image_path, cropped_image)
                print("plate num :" , plate_num , " Cropped image :" , cropped_image_path)
                if plate_num is None or plate_num == '' or cropped_image_path is None or cropped_image_path == '':
                    continue  
                return plate_num,cropped_image_path
            return None, None
                # return plate_num
                # cv2.imshow("Character's Segmented", im2)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

    def detected_license(self,frame):
        cv2.imshow(" detected_licenseplate_frame ", frame)
        results = self.model(frame)
       
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cropped_image = frame[y1:y2, x1:x2]

                # Convert to grayscale
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
               
        
                # Apply preprocessing
                bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
                edged = cv2.Canny(bfilter, 30, 200)  # Edge detection

                custom_config = r'--oem 3 --psm 8'
                text = pytesseract.image_to_string(gray, config=custom_config, lang="eng").strip()
                print(" text after gray: " , text)
                # Apply thresholding for better OCR results
                _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imshow(" detected_licenseplate_frame_filtered", binary_image)
                custom_config = r'--oem 3 --psm 8'
                text = pytesseract.image_to_string(binary_image, config=custom_config, lang="eng").strip()
                print(" text after gray: " , text)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                sanitized_text = self.sanitize_text(text)
                # print(self.sanitize_text)

                # Check OCR confidence and save only if it meets a threshold
                data = pytesseract.image_to_data(binary_image, config=custom_config, output_type=pytesseract.Output.DICT)
                if 'conf' in data and data['conf']:
                    confidence = data['conf']
        
                    confidences = [int(conf) for conf in confidence if str(conf).isdigit()]
        
                    if confidences:
                        average_confidence = np.mean(confidences)
                    else:
                        average_confidence = 0
                else:
                    average_confidence = 0

                if len(sanitized_text) == 6 and average_confidence >= 10 and sanitized_text not in self.detected_plates:  # Adjust confidence threshold as needed
                    self.detected_plates[sanitized_text] = True

                    # Specify output file paths
                    output_file_text = "output2.txt"
                    confidence_str = f'{average_confidence:.2f}'
                    output_file_image = f'detected_license/{sanitized_text}_{confidence_str}.jpg'

                    # Save the cropped image
                    cv2.imwrite(output_file_image, cropped_image)
                    # print("Image saved to:", output_file_image)

                    # Write the recognized text to the output file
                    with open(output_file_text, "a", encoding="utf-8") as file:
                        file.write(f'{sanitized_text} {confidence_str}\n')  # Add a newline after each text

                    # print("Text saved to:", output_file_text)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, sanitized_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    print(sanitized_text)
                    return sanitized_text
       
