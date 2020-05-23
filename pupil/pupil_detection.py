import cv2

#符號"_"在這用意為沒使用到的參數，平常代表最後執行結果

cap = cv2.VideoCapture("pupil_target.mp4") #這裡要輸入影片的檔案名稱

while True:
    ret, frame = cap.read()
    roi = frame[333:625,95:506] #只取有興趣的區域, 手動框出眼睛部位
    rows,cols,_ =roi.shape
    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  #灰階處理               
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0) #模糊化 去除一些雜訊
    _, threshold = cv2.threshold(gray_roi, 17, 255, cv2.THRESH_BINARY_INV)  #二值化處理(處理目標, threshold, 最大值, 二值化)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #找出其輪廓
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) #排序輪廓由大到小


    #取出第一個輪廓後即break, 所以是最大輪廓
    for cnt in contours:
        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)  #arg為(原圖, 輪廓座標, -1為顯示全部輪廓, 顏色, 線寬pixel)
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi,(x + int(w/2), 0),(x + int(w/2), rows),(0,255,0),2)
        cv2.line(roi,(0, y + int(h/2)),(cols, y + int(h/2)),(0,255,0),2)
        break

    cv2.imshow("threshold", threshold)
    cv2.imshow("roi", roi)
    key = cv2.waitKey(30)
    if key == 27: #27代表ESC
        break
    
cap.release()
cv2.desttroyAllWindows()

