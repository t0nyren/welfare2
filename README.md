welfare2
========

Web-level Face Recognition Version 2

本项目实现了人脸检测，标定，建模以及识别四个功能。可用于linux或windows Visual Studio环境，所有依赖已包含在此项目中，无需安装其他第三方库。程序具体使用方式如下：

### faceDetect

faceDetect实现人脸检测，将一张图片中的人脸以及五官位置标出。

### buildModel

buildModel实现人脸特征的提取以及建模。首先将不同人的照片放置于不同文件夹中，
修改src/buildModel.cpp中照片数据集以及输出人脸和模型的路径，编译后执行程序即可。

### loadModel

loadModel会对人脸模型的准确率进行测试，输出识别正确率，修改src/loadModel.cpp中的路径，编译后执行程序即可。

### faceRec

faceRec会监控摄像头或视频文件，逐帧进行人脸检测以及识别，并在图形界面上表示出检测结果，修改src/faceRec.cpp中的路径，编译后执行程序即可。

## 主要工具以及类

### Detector

Detector实现了人脸检测以及标定功能，在原图中标出人脸，亦可输出自定大小的标准化人脸截图。

### Encoder

Encoder对标准化的人脸进行特征提取以及编码，将图片转换为高维特征向量，为后期的索引，查询以及识别工作做准备。

### Recognizer 

Recognizer对人脸特征向量构建索引，并提供查询以及识别功能。
