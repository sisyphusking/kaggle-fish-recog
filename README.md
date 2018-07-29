## nature-conservancy-fisheries-monitoring

#### inception_v3
- 使用inception v3模型的网络结构，在最后一层接入神经网络分类器
- 使用梯度下降优化，learning rate设置成7e-4，准确率可以达到80左右（训练速度很慢）
- 使用数据增强、增大训练次数可以提高准确率


#### transfer_learning_svm
- 使用数据增强，扩大数据集
- 使用迁移学习，利用google训练好的模型，提取图片特征
- 使用svm进行分类，最终模型的准确率在85%以上


#### 参考资料
- [Transfer Learning: retraining Inception V3 for custom image classification](https://becominghuman.ai/transfer-learning-retraining-inception-v3-for-custom-image-classification-2820f653c557)
- [data](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data)
- [inveption-v3-2016-03-01 model](http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz)
