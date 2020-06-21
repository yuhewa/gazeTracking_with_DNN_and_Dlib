import numpy as np
from cv2 import cv2
from dlib import dlib
import os.path
from calibration import calibration 


def detectFace():

    #取得檔案路徑後, 在其後加上欲讀取檔案名稱
    dir_path = os.path.dirname(__file__)
    
    ## threshold校正
    tholdValue, pupilCenter = calibration()
    print('校正數值')
    print('threshold  : ', tholdValue)
    print('pupilCenter: ', pupilCenter)

    #創net必要的兩個檔案 1.model(訓練好的模型) 2.prototxt(模型架構)
    model = os.path.join(dir_path, "res10_300x300_ssd_iter_140000.caffemodel")
    prototxt = os.path.join(dir_path, "deploy.prototxt.txt")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    cap = cv2.VideoCapture(0) #設0的話就是用攝像頭畫面

    predictor_path = os.path.join(dir_path, "shape_predictor_5_face_landmarks.dat")
    predictor = dlib.shape_predictor(predictor_path)
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
                #### detectPupil(原圖, 點的編號, 點的編號, thold_value, 左右區間值or中心位置)
                # 眼部座標, 原點在左上角, 因此min_y要用加數值才會往下
                min_x = landmarks.part(1).x
                max_x = landmarks.part(0).x
                min_y = landmarks.part(0).y + (max_x-min_x)//3
                max_y = landmarks.part(0).y - (max_x-min_x)//3
                eye = frame[max_y: min_y, min_x: max_x]
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2) #矩形框出眼部位置
                # 灰階處理輸入的矩形區塊要注意其座標, 錯誤擺放會無法執行
                # 執行到一半時常常突然中止, 顯示灰階處理的部分會錯誤, 
                # 猜測是landmarks擷取失敗, 導致參數eye無法正確輸入灰階處理的函數
                #### 可能的sol: 想辦法加個landmarks是否正確取得的判斷
                # print(min_x, max_x, min_y, max_y)
                #若太近, 則不繼續, 並顯示太近
                if min_x > 500:
                    cv2.putText(frame, 'Too close', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
                    continue
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  #灰階處理    
                gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0) #模糊化, 去除一些雜訊
                _, threshold = cv2.threshold(gray_eye, tholdValue, 255, cv2.THRESH_BINARY_INV)  
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #找出其輪廓
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) #排序輪廓由大到小
                # 偵測眨眼, 若偵測到則不繼續, 並顯示blink
                if cv2.countNonZero(threshold) < 20:
                    cv2.putText(frame, 'Blink', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 3)
                    continue
                for cnt in contours:
                    # cv2.drawContours(eye, [cnt], -1, (0, 0, 255), 3)  #arg為(原圖, 輪廓座標, -1為顯示全部輪廓, 顏色, 線寬)
                    
                    # 取出能包圍瞳孔輪廓的最小矩形
                    # xy為左上角原點, wh為寬高 (須注意xy為相對於眼部區域的矩形左上角原點的距離, 並非原圖)
                    (x, y, w, h) = cv2.boundingRect(cnt)

                    # 用矩形框出瞳孔位置
                    # cv2.rectangle(eye, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # 以瞳孔矩形定出中心線, 線延伸到眼部矩形
                    cv2.line(eye,(x + int(w/2), 0),(x + int(w/2), width),(0,255,0),2)
                    cv2.line(eye,(0, y + int(h/2)),(height, y + int(h/2)),(0,255,0),2)
                    pupil_center = (x + w//2, y + h//2) #瞳孔中心座標
                    # print("瞳孔中心:       " , pupil_center, )
                    # print("眼部區域左右座標: 0", max_x - min_x)
                    hr_ratio = int(pupil_center[0]/(max_x - min_x) * 100)
                    # print("瞳孔水平比例", hr_ratio, "%")
                    # cv2.putText(影像, 文字, 座標, 字型, 大小, 顏色, 線條寬度, 線條種類)
                    # 水平偵測
                    
                    if   hr_ratio < pupilCenter - 15: 
                        cv2.putText(frame, 'Right', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3) 
                    elif hr_ratio > pupilCenter + 15:
                        cv2.putText(frame, 'Left', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
                    else:
                        cv2.putText(frame, 'Center', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)

                    break

                cv2.imshow("gray_eye", gray_eye)
                cv2.imshow("threshold",threshold)
        

        cv2.imshow("Output", frame)
        # 如果沒有waitKey, imshow會無法顯示
        key = cv2.waitKey(1)
        if key == 27: #27代表ESC
            break

    cap.release()
    cv2.destroyAllWindows()

detectFace()