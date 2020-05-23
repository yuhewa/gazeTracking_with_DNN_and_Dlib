# gazeTracking_with_DNN_and_Dlib
poject
預計使用到:
1.OpenCV DNN偵測臉部
2.dlib抓取臉部特徵(facial landmarks)
3.針對眼部進行瞳孔追蹤

階段目標
1.將faceDetection改為處理即時影像且可以從外部導入參數(使用argparse)
2.導入dlib抓取特徵
3.利用眼部特徵抓取瞳孔位置並判斷其位移方向