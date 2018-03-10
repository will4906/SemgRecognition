# SemgRcognition

表面肌信号手势识别

## 背景信息

以现在的技术成果，通过表面肌电信号进行手势识别具有较高的准确率。但是仍然存在一些问题，比如对肌电电极的摆放位置要求较严格，传统机器学习方法显示，如果电极发生错位会对结果有较大影响。因此我们尝试使用卷积神经网络的方法来规避电极位置错位带来的问题。

## 基础信息

### 数据集

* 来源 [CapgMyo](http://zju-capg.org/myo/data/index.html)

    一个利用高密度电极采集的肌电信号数据，数据格式为16r*8c

* 数据预处理(采用已经预处理过的数据)

    带阻滤波器、二阶巴特沃斯滤波器、低通滤波器等

* 模拟错位

    由于经费、时间等成本问题，我们无法实际进行错位的数据收集。因而采用将capgmyo数据进行扩充和旋转的方式进行模拟。

    具体方法：

    * 原始数据为16行8列，我们将16行标号为0……15。
    * 扩充数据，形如[14, 15, 0……15, 0, 1],[13, 14, 15, 0……15, 0]
    
    TODO:具体的公式以后再列

### 模型定义

卷积神经网络

* 8层结构，基于[Gesture recognition by instantaneous surface EMG images](http://www.nature.com/articles/srep36571)

    * 两层卷积层、两层局部连接层、两层512单元的全连接层、一层128单元的全连接层、8单元输出层。
    * 每层激活函数都为ReLU non-linearity。
    * 在去线性激活函数之前和输入之后，每层都进行归一化处理。
    * 在第4、5、6层有0.5的dropout。
    * 使用随机梯度下降的方法，学习率设置为0.1，权重衰减0.0001
    * batch size 1000
    * epoch 10(此处论文中为28，但是耗时太久，遂减为10)
    
    [模型结构图](images/8layer_model.png)

* 3层结构，基于[Self-Recalibrating Surface EMG Pattern Recognition for Neuroprosthesis Control Based on Convolutional Neural Network](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5504564/)

    * 一个卷积层、两个全连接层

    TODO: 论文作者认为分类器的参数设定影响不是很大。实验后感觉确实只有微小差距，数据仍在训练等待
    
## 结果

* 3层结构的平均准确率为：76.1208289931%
* 8层结构仍的平均准确率为：86.25458331778646%