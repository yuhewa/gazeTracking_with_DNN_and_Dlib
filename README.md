# gazeTracking_with_DNN_and_Dlib

## 相關技術:
> OpenCV DNN偵測臉部
> dlib抓取臉部特徵(facial landmarks)

## 步驟:
> DNN找出臉部
> dlib找出眼部區域
> 模糊並門檻化眼部區域, 找出瞳孔輪廓
> 設一值為瞳孔中心, 判斷其x軸位移並輸出訊號
