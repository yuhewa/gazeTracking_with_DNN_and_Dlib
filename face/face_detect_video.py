import numpy as np
import cv2

def detectFace(video):
    	
    #創net必要的兩個檔案 1.model 2.prototxt
    model = "./res10_300x300_ssd_iter_140000.caffemodel"
    prototxt = "./deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(prototxt, model)

    #抓影片檔案
    cap = cv2.VideoCapture(video)

    while True:
        _, frame = cap.read()
        (h, w) = frame.shape[:2]
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
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                #顯示出框出的位置
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


        cv2.imshow("Output", frame)

        key = cv2.waitKey(1)
        if key == 27: #27代表ESC
            break
    

    cap.release()
    cv2.desttroyAllWindows()


detectFace("./face_target.mp4")