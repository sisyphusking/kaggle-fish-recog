## nature-conservancy-fisheries-monitoring

kaggle竞赛

#### inception_v3
- 使用inception v3模型的网络结构，在最后一层接入神经网络分类器
- 使用梯度下降优化，learning rate设置成7e-4，准确率可以达到80左右（训练速度很慢）
- 使用数据增强、增大训练次数可以提高准确率



#### transfer_learning_svm
- 使用数据增强，扩大数据集
- 使用迁移学习，利用google训练好的模型，提取图片特征
- 使用svm进行分类，使用pca降维，最终模型的准确率在91%以上

#### transfer_learning_rf
- 同上，使用迁移学习提取图片特征，后面使用随机森林来进行分类
- 模型准确率不是很高，在60%左右，参数还有调优空间


#### 可优化点
- 使用目标检测技术: [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)、[YOLO](https://pjreddie.com/darknet/yolo/)、[SSD](https://github.com/rykov8/ssd_keras)
- 可以借鉴`VGGnet`，效果比`Inception V3`好，[代码](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069)


#### 参考资料
- [What I’ve learned from Kaggle’s fisheries competition](https://medium.com/@gidishperber/what-ive-learned-from-kaggle-s-fisheries-competition-92342f9ca779)
- [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)
- [Detect and Classify Species of Fish from Fishing Vessels with Modern Object Detectors and Deep Convolutional Networks](https://flyyufelix.github.io/2017/04/16/kaggle-nature-conservancy.html)
- [Transfer Learning: retraining Inception V3 for custom image classification](https://becominghuman.ai/transfer-learning-retraining-inception-v3-for-custom-image-classification-2820f653c557)
- [kaggle data](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data)
- [inveption-v3-2016-03-01 model](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz)
