# 地脚螺栓主柱中心偏移-小圆部分
## 版本
本文件当前版本为1.0。
- 1.0： 首次发布
## 拍照
需要两组照片，一组用于相机标定（标定用照片），另一组用于识别四圆柱中心（识别用照片）。

拍摄识别用照片时，应该近距离、无角度拍摄。

拍照示例详见本仓库的照片（photos）文件夹，该文件夹收录了目前可用的测试照片。
## 代码介绍
### 依赖包
- OpenCV
- NumPy
- Glob
- Math
### 文件架构
- `perspective_transform.py`：读取标定用照片，计算相机参数矩阵，在读取识别用照片，对识别用照片进行去畸变和透视变换。
- `recognition.py`：读取识别用照片，进行放缩、裁剪、整体嵌套边缘识别（HED）、霍夫圆识别、筛选、计算误差。
- `hed_pretrained_bsds.caffemodel`：HED的caffe神经网络模型。
- `deploy.prototxt`：HED模型的配置文件。
### 测试用机制
目前代码仍在测试和开发阶段，为了便于调试，存在以下的测试用机制：
- 程序读取路径内的所有文件。文件路径需要手动更改。
- 为了验证霍夫圆识别效果，将霍夫圆识别的结果绘制于透视变换后的照片上、存储该照片。
- 为了计算误差，通过识别棋盘格函数`cv2.findChessboardCorners`，得到棋盘格的中心，将其作为“实际中心”。

以上机制在正式发布的代码中应该删除或更改。