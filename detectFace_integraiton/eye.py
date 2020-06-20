from cv2 import cv2
import numpy as np

class eye():
    def __init__(self, frame, landmarks, x1, x2, thold_value = 40, leftRange = 40, rightRange = 70):
        self.frame = frame
        self.landmarks = landmarks
        self.x1 = x1
        self.x2 = x2
        self.thold_value = thold_value

        self.hr_ratio = 0
        self.leftRange = leftRange
        self.rightRange = rightRange
        self.pupilThreshold = 0
    # 左眼 x1 = 1, x2 = 0
    # 右眼 x1 = 2, x2 = 3

    def detectPupil(self):
        ####               之後拉出來寫成眼部處理函數                    ####
                #### detectPupil(原圖, 點的編號, 點的編號, thold_value, 左右區間值or中心位置)
                # 眼部座標, 原點在左上角, 因此min_y要用加數值才會往下
                min_x = self.landmarks.part(self.x1).x
                max_x = self.landmarks.part(self.x2).x
                min_y = self.landmarks.part(self.x2).y + (max_x-min_x)//3
                max_y = self.landmarks.part(self.x2).y - (max_x-min_x)//3
                eye = self.frame[max_y: min_y, min_x: max_x]
                cv2.rectangle(self.frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2) #矩形框出眼部位置
                # 灰階處理輸入的矩形區塊要注意其座標, 錯誤擺放會無法執行
                # 執行到一半時常常突然中止, 顯示灰階處理的部分會錯誤, 
                # 猜測是landmarks擷取失敗, 導致參數eye無法正確輸入灰階處理的函數
                #### 可能的sol: 想辦法加個landmarks是否正確取得的判斷
                # print(min_x, max_x, min_y, max_y)
                
                gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)  #灰階處理    
                gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0) #模糊化, 去除一些雜訊
                #### 二值化處理(處理目標, threshold, 最大值, 二值化), 之後要寫一個函數能找到最佳threshold
                _, threshold = cv2.threshold(gray_eye, self.thold_value, 255, cv2.THRESH_BINARY_INV)  
                self.pupilThreshold = threshold
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #找出其輪廓
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) #排序輪廓由大到小

                
                for cnt in contours:
                    cv2.drawContours(eye, [cnt], -1, (0, 0, 255), 3)  #arg為(原圖, 輪廓座標, -1為顯示全部輪廓, 顏色, 線寬)
                    # 取出能包圍瞳孔輪廓的最小矩形
                    # xy為左上角原點, wh為寬高 (須注意xy為相對於眼部區域的矩形左上角原點的距離, 並非原圖)
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    pupil_center = (x + w//2, y + h//2) #瞳孔中心座標
                    self.hr_ratio = int(pupil_center[0]/(max_x - min_x) * 100)
                    print("瞳孔水平比例", self.hr_ratio, "%")
                    #### 之後打算做一個校正程式, 可以決定其threshold和左右區間 ####
                    break
                ####               /之後拉出來寫成眼部瞳孔處理函數                    ####

    def isLeft(self):
        if self.hr_ratio < 40:
            return True
        return False

    def isRight(self):
        if self.hr_ratio > 70:
            return True
        return False
    
    #偵測Blink, 為了放上文字. threshold是瞳孔黑白圖, 用來算面積
    def isBlink(self):
        #cv2.countNonZero(threshold)可以找出瞳孔面積, 若為0則沒有瞳孔, 判斷為眨眼, 用10是為了準確率
        if cv2.countNonZero(self.pupilThreshold) < 10:
            return True
        return False