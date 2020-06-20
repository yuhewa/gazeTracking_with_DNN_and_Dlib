import numpy as np
from cv2 import cv2
from dlib import dlib
import os.path


def calibration():
    dir_path = os.path.dirname(__file__)
    model = os.path.join(dir_path, "res10_300x300_ssd_iter_140000.caffemodel")
    prototxt = os.path.join(dir_path, "deploy.prototxt.txt")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    cap = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    tholdValue = 20
    print('look forward')
    while True:
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
            print('ok')
            break    
        tholdValue += 1

        key = cv2.waitKey(1)
        if key == 27: #27代表ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    return tholdValue


