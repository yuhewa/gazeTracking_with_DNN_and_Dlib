import numpy as np
from cv2 import cv2
from dlib import dlib
import os.path


def calibration():

    ##################### threshold的校正
    bg = np.zeros((300, 1200, 3), dtype = np.uint8)
    cv2.putText(bg, 'look here...', (400,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)                    
    cv2.imshow('bg', bg)

    dir_path = os.path.dirname(__file__)
    model = os.path.join(dir_path, "res10_300x300_ssd_iter_140000.caffemodel")
    prototxt = os.path.join(dir_path, "deploy.prototxt.txt")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    cap = cv2.VideoCapture(0)
    predictor_path = os.path.join(dir_path, "shape_predictor_5_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)
    tholdValue = 20
    # print('look forward')
    while True:
        tholdValue += 2 ## 提前加校正數值
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                landmarks = predictor(gray, dlib.rectangle(int(startX), int(startY),int(endX), int(endY)))
                min_x = landmarks.part(1).x
                max_x = landmarks.part(0).x
                min_y = landmarks.part(0).y + (max_x-min_x)//3
                max_y = landmarks.part(0).y - (max_x-min_x)//3
                eye = frame[max_y: min_y, min_x: max_x]
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  #灰階處理    
                gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0) #模糊化, 去除一些雜訊
                _, threshold = cv2.threshold(gray_eye, tholdValue, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow("threshold",threshold)  
            
            
        if cv2.countNonZero(threshold) > 115:
            cv2.putText(bg, 'OK!', (500,210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
            cv2.imshow('bg', bg)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            break    

        key = cv2.waitKey(1)
        if key == 27: #27代表ESC
            break

    ##################### 左右的校正
    bg = np.zeros((300, 1200, 3), dtype = np.uint8)
    cv2.putText(bg, 'look here again', (300,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 3)
    cv2.imshow('bg', bg)
    cv2.waitKey(500)
    avg_hr_ratio = 0
    time = 0
    while time < 10:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                landmarks = predictor(gray, dlib.rectangle(int(startX), int(startY),int(endX), int(endY)))
                min_x = landmarks.part(1).x
                max_x = landmarks.part(0).x
                min_y = landmarks.part(0).y + (max_x-min_x)//3
                max_y = landmarks.part(0).y - (max_x-min_x)//3
                eye = frame[max_y: min_y, min_x: max_x]
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  #灰階處理    
                gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0) #模糊化, 去除一些雜訊
                _, threshold = cv2.threshold(gray_eye, tholdValue, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #找出其輪廓
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) #排序輪廓由大到小
                for cnt in contours:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    pupil_center = (x + w//2, y + h//2) #瞳孔中心座標
                    hr_ratio = int(pupil_center[0]/(max_x - min_x) * 100)
                    if hr_ratio < 70 and hr_ratio > 40:
                        avg_hr_ratio += hr_ratio
                        time += 1
                        print("瞳孔水平比例", hr_ratio, "%")

    cv2.waitKey(1100)
    cv2.putText(bg, 'OK!', (500,210), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    cv2.imshow('bg', bg)
    cv2.waitKey(1000)
        
    cap.release()
    cv2.destroyAllWindows()
    return tholdValue, avg_hr_ratio//10
