import numpy as np
from cv2 import cv2
from dlib import dlib
import os.path
from eye import eye
from calibration import calibration


def detectFace():
    #創net必要的兩個檔案 1.model(訓練好的模型) 2.prototxt(模型架構)
    #取得檔案路徑後, 在其後加上欲讀取檔案名稱

    tholdValue = calibration()

    dir_path = os.path.dirname(__file__)
    model = os.path.join(dir_path, "res10_300x300_ssd_iter_140000.caffemodel")
    prototxt = os.path.join(dir_path, "deploy.prototxt.txt")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    cap = cv2.VideoCapture(0) #設0的話就是用攝像頭畫面
    predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
    while True:
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (height, width) = frame.shape[:2]
        #blob 直接翻譯為二進位大型物件(binary large object)
        #先將圖片resize為300x300在傳進去,最後一項為normalize因數, 參考別人的
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        #將blob作為net的輸入
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            #confidence可以理解為分數
            confidence = detections[0, 0, i, 2]
            #confidence要大於多少是可以調整的,以後可以用外部參數傳入
            if confidence > 0.6:
                #box為臉部範圍
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")
                #臉部特徵, 之所以加int轉型是因為沒加會報錯, 但是上行的astype已經指定回傳int型態, 不知為什麼報錯
                landmarks = predictor(gray, dlib.rectangle(int(startX), int(startY),int(endX), int(endY)))
                
                ####               之後拉出來寫成眼部處理函數                    ####

                # detectPupil(原圖, 點的編號, 點的編號, thold_value, 左右區間值or中心位置)
                left_eye = eye(frame, landmarks, 1, 0 ,40, tholdValue)
                left_eye.detectPupil()
                text = ''
                if left_eye.isBlink():
                    text = 'blink'
                elif left_eye.isLeft():
                    text = 'left' 
                elif left_eye.isRight():
                    text = 'right'
                else:
                    text = 'center'
                
                cv2.putText(frame, text, (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                ####               /之後拉出來寫成眼部瞳孔處理函數                    ####

        cv2.imshow("Output", frame)

        key = cv2.waitKey(1)
        if key == 27: #27代表ESC
            break

    cap.release()
    cv2.destroyAllWindows()

detectFace()