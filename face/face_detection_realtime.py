import numpy as np
from cv2 import cv2
from dlib import dlib
import os.path

def detectFace():
    	
    #創net必要的兩個檔案 1.model(訓練好的模型) 2.prototxt(模型架構)
    dir_path = os.path.dirname(__file__)
    model = os.path.join(dir_path, "res10_300x300_ssd_iter_140000.caffemodel")
    prototxt = os.path.join(dir_path, "deploy.prototxt.txt")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    #抓影片檔案
    cap = cv2.VideoCapture(0)
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = frame.shape[:2]
        #blob 直接翻譯為二進位大型物件 binary large object
        #先將圖片resize為300x300在傳進去,最後一項為normalize因數
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        #將blob作為net的輸入
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            #confidence可以理解為分數
            confidence = detections[0, 0, i, 2]
            #confidence要大於多少是可以調整的,以後可以用外部參數傳入
            if confidence > 0.5:
                #box為臉部範圍
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                #臉部特徵
                landmark = predictor(gray, dlib.rectangle(int(startX), int(startY),int(endX), int(endY)))
                #眼部座標
                left_eye = np.array( [ [landmark.part(36).x,landmark.part(36).y], [landmark.part(37).x,landmark.part(37).y], [landmark.part(38).x,landmark.part(38).y], [landmark.part(39).x,landmark.part(39).y], [landmark.part(40).x,landmark.part(40).y], [landmark.part(41).x,landmark.part(41).y] ], np.int32)
                
                # 眼部多邊形輪廓
                cv2.polylines(frame, [left_eye], True, (0, 0, 225), 1)
                
                min_x = np.min(left_eye[:,0])
                max_x = np.max(left_eye[:,0])
                min_y = np.min(left_eye[:,1])
                max_y = np.max(left_eye[:,1])
                eye = frame[min_y: max_y, min_x: max_x]
                
                # 放大眼部
                # eye = cv2.resize(eye, None, fx=5, fy=5)

                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  #灰階處理               
                gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0) #模糊化 去除一些雜訊
                _, threshold = cv2.threshold(gray_eye, 60, 255, cv2.THRESH_BINARY_INV)  #二值化處理(處理目標, threshold, 最大值, 二值化)
                print(cv2.countNonZero(threshold))
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #找出其輪廓
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) #排序輪廓由大到小


                for cnt in contours:
                    cv2.drawContours(eye, [cnt], -1, (0, 0, 255), 3)  #arg為(原圖, 輪廓座標, -1為顯示全部輪廓, 顏色, 線寬pixel)
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    cv2.rectangle(eye, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.line(eye,(x + int(w/2), 0),(x + int(w/2), width),(0,255,0),2)
                    cv2.line(eye,(0, y + int(h/2)),(height, y + int(h/2)),(0,255,0),2)
                    
                    break   

                cv2.imshow("gray_eye", gray_eye)
                cv2.imshow("threshold",threshold)

        cv2.imshow("Output", frame)

        key = cv2.waitKey(1)
        if key == 27: #27代表ESC
            break
    

    cap.release()
    cv2.desttroyAllWindows()


detectFace()