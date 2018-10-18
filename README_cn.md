# Keras Faster-RCNN

>这是一个非常有用的基于tensorflow和keras的fast-rcnn实现，模型非常清晰，只保存在.h5文件中，开箱即可使用，并且易于在其他数据集上进行全面支持。如果您有任何疑问，请随时通过微信询问我：jintianiloveu, 或者关注奇异AI的公众号加入社群与奇异AI的客服支持询问。

## 要求

基本上，这段代码支持python2.7和python3.5，应该安装以下包：

* tensorflow
* keras
* scipy
* cv2

## 开箱即用模型进行预测

我训练了一个模型来预测kitti。我稍后会更新Dropbox链接。让我们看看预测的结果：

![](http://opbocoyb4.bkt.clouddn.com/000010.png)

## 训练新数据集

训练新数据集也非常简单直接。只需将您的检测标签文件转换为以下格式：

```
/path/training/image_2/000000.png,712.40,143.00,810.73,307.92,Pedestrian
/path/training/image_2/000001.png,599.41,156.40,629.75,189.25,Truck
```
这是`/path/to/img.png，x1，y1，x2，y2，class_name`，有了这个简单的文件，我们不需要类映射文件，我们的训练程序会自动统计这个。

## For Predict

如果你想看看训练有素的模型有多好，只需运行：
```
python test_frcnn_kitti.py
```
你也可以使用`-p`来预测特定的单个图像，或者发送一个包含很多图像的路径，我们的程序会自动识别出来。

**这就是全部，帮助你享受！**