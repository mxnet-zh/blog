---
title: 神经网络推理加速之模型量化
author: 赵鹏, 陈新宇, 秦臻南, 叶军  翻译： 包怡欣 （INTEL MLT TEAM）
---

## 1. 引言

在深度学习中，推理是指将一个预先训练好的神经网络模型部署到实际业务场景中，如图像分类、物体检测、在线翻译等。由于推理结果会直接呈现给终端用户，因此在实际中推理至关重要，尤其对于企业级的产品而言更是如此。

衡量推理性能的重要指标包括延迟（latency）和吞吐量（throughput）。延迟是指完成一次预测所需的时间；吞吐量（throughput）是指单位时间内处理数据的数量。低延迟和高吞吐量能够保证良好的用户体验和工业生产要求。

实际生产环境中，为了提供高性能和低成本的服务，许多云服务提供商和硬件供应商会针对推理专门优化他们的服务和架构，如亚马逊的[SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/deploy-model.html)、[Deep Learning AMIs](https://aws.amazon.com/blogs/machine-learning/aws-deep-learning-amis-now-come-with-tensorflow-1-13-mxnet-1-4-and-support-amazon-linux-2/)、英特尔®的[Deep Learning Boost](https://www.intel.ai/intel-deep-learning-boost/#gs.0ngn54) (Intel® DL Boost)，包括第二代英特尔®至强®可扩展处理器中的矢量神经网络指令 (VNNI)。

在软件层，Apache MXNet\*社区提供了丰富的量化工具来提升推理性能并降低部署成本。量化后的模型可以通过像VNNI这样的低精度（INT8）指令进行加速。除此之外，低精度的数据类型可以节省存储带宽，提高缓存命中率，并减少能耗。

通过MXNet的模型量化，ResNet50 v1可以在[AWS* EC2 CPU](https://amazonaws-china.com/ec2/instance-types/c5/) 实例中达到6.42倍的性能加速。其中运算符融合获得了1.75倍的加速，VNNI的INT8指令则带来了3.66倍的加速。量化后的精度损失仅为0.38%。因此，我们可以看到借助MXNet进行模型量化，并配合VNNI指令进行加速，能够带来前所未有的推理体验。

本文先描述模型量化原理以及在MXNET上的实现，然后讲述如何从用户角度使用量化工具，最后介绍在英特尔至强服务器上VNNI带来的性能提升。

## 2. 模型量化

MXNet支持从单精度（FP32）到有符号的8比特整型（s8）以及无符号的8比特整型（u8）的模型量化。u8可以用于CNN网络的推理。对于大多数CNN网络而言，Relu是常用的激活函数，其输出是非负的。因此，在CNN网络的推理中使用u8的优势显而易见——我们可以多使用1比特来表示数据从而达到更高的精度。s8则可以用于更为通用的模型量化。

在进行模型量化时，用户不需要重新训练模型，其只需通过量化校准工具就可以对训练好的FP32模型进行量化加速。因此部署过程非常方便快捷。

模型的量化推理包含两个阶段：

![图1. MXNet INT8推理流程示意图](img/2019-07-02-fig1.png)

- 校准过程（预处理阶段），我们使用验证集中的一小部分图片，通常为整个数据集的1-5%，来收集数据分布的信息，其中包括最小值/最大值或者基于熵理论的最佳阈值，以及基于对称量化和各层执行配置文件定义的量化因子。最终，这些量化参数会被记录在新生成的量化模型中。
- INT8推理（运行阶段）， 量化模型可以像原始模型一样被加载并用于推理。

## **3. 模型优化**

MXNet提供了很多高级性能优化方法来加速量化模型的部署，包括支持INT8的数据加载器，离线校准，图优化等。在深度学习领域，MXNet是最早提供完整量化方案的学习框架之一。为了减少在量化过程中，反复的数据类型转化，MXNet对一些常见的操作进行功能合并，如Conv+Relu, Conv+BN, Conv+Sum等，从而使得整个量化后的网络相比于原始模型更加简洁高效。例如，下图中的ResNet50 V1显示了网络在运算符融合和量化前后的变化。

![图2. ResNet50 V1网络结构（左图：FP32，右图：INT8）](img/2019-07-02-fig2.png)

另外值得一提的是，MXNet的量化方案在功能上具有很好的向后及向前兼容性。当用户在不同的硬件上部署量化模型时, 无论是否具有VNNI指令的支持，所有的这些软件功能还都是有效的。比如从C5.18x.large （英特尔第一代可扩展处理器，不支持VNNI指令) 切换到C5.24x.large (英特尔第二代可扩展处理器，支持VNNI指令），用户无需修改代码就可以获得大幅度的性能提升。 

![图3. 英特尔® Deep Learning Boost](img/2019-07-02-fig3.png)

## 4. 模型部署

用户可以使用MXNet的量化校准工具和API轻松地将他们的FP32模型量化成INT8模型。MXNet官方也提供了两种类型的量化示例：[图像分类的模型量化](https://github.com/apache/incubator-mxnet/tree/master/example/quantization)和物体检测的模型量化 ([SSD-VGG16](https://github.com/apache/incubator-mxnet/tree/master/example/ssd))。用户也可以参考[量化APIs](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py)来将这些工具集成到实际的推理任务中。下面，以SSD-VGG16为例介绍MXNet的模型量化过程。

### 4.1 准备阶段

使用以下命令可以安装具有CPU性能优化的MXNet

```
pip install --pre mxnet-mkl
```

首先，下载已经训练好的[SSD-VGG16模型](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/models/ssd_vgg16_reduced_300-dd479559.zip) 和[打包的二进制数据](http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/ssd-val-fc19a535.zip)。创建**model** 和**data**的目录，解压zip文件并重命名，如下所示。

```
data/ 
	|--val.rec
	|--val.lxt 
	|--val.idx
model/
	|--ssd_vgg16_reduced_300–0000.params
	|--ssd_vgg16_reduced_300-symbol.json
```

然后，你可以使用如下命令来验证float32的预训练模型：

```python
# USE MKLDNN AS SUBGRAPH BACKEND
export MXNET_SUBGRAPH_BACKEND=MKLDNN
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/ssd_
```

### 4.2 量化阶段

MXNet为SSD-VGG16提供了一个[量化脚本](https://github.com/apache/incubator-mxnet/blob/master/example/ssd/quantization.py)。用户可以通过设置不同的配置项将模型从FP32量化成INT8，包括batch size、量化用的batch数目、校准模式、输入数据的量化目标数据类型、不做量化的层，以及数据加载器的其他配置。我们可以使用如下指令进行量化，默认情况下脚本使用5个batch（每个batch包含32个样本）进行量化。

```
python quantization.py
```

量化后的INT8模型会以如下形式存储在**model**目录下。

```
data/    
    |--val.rec    
    |--val.lxt    
    |--val.idx
model/    
    |--ssd_vgg16_reduced_300–0000.params    
    |--ssd_vgg16_reduced_300-symbol.json    
    |--cqssd_vgg16_reduced_300–0000.params   
    |--cqssd_vgg16_reduced_300-symbol.json	
```

### 4.3 部署INT8推理

使用如下指令执行INT8模型的推理。

```
python evaluate.py --cpu --num-batch 10 --batch-size 224 --deploy --prefix=./model/cqssd_
```

### 4.4 结果可视化

从Pascal VOC2007验证集中取一张图片，其检测结果如下图所示。图4.1显示的是FP32模型的推理结果，图4.2显示的是INT8模型的推理结果。

结果可视化的指令如下。

```python
# Download demo image
python data/demo/download_demo_images.py
# visualize float32 detection
python demo.py --cpu --network vgg16_reduced --data-shape 300 --deploy --prefix=./model/ssd_
# visualize int8 detection
python demo.py --cpu --network vgg16_reduced --data-shape 300 --deploy --prefix=./model/cqssd_
```

![图4.1. SSD-VGG检测结果, FP32](img/2019-07-02-fig4.1.jpg)

![图4.2. SSD-VGG检测结果, INT8](img/2019-07-02-fig4.2.jpg)


## 5. 性能

在本节中，我们将展示在使用Intel DL Boost进行推理时端到端的性能提升。更多的模型和性能数据，请参考[Apache / MXNet C ++接口](https://github.com/apache/incubator-mxnet/tree/master/cpp-package/example/inference)和[GluonCV模型](https://gluon-cv.mxnet.io/build/examples_deployment/int8_inference.html)中的示例。 

本小节的数据均来自AWS EC2 C5.24xlarge实例上的英特尔第二代可扩展处理器。完整的硬件和软件配置请参阅通知和免责声明。 

通过运算符融合和模型量化，图5中的总吞吐量得到了从6.42X到4.06X不等的显著提高。 运算符融合带来的加速随着模型中可以融合数量的多少而变化。

![图5. MXNet\* 融合和量化加速对比](img/2019-07-02-fig5.png)


模型量化则可以为大部分模型提供更稳定的加速，例如ResNet 50 v1为3.66X，ResNet 101 v1为3.82X，SSD-VGG16为3.77X，其值已经非常接近INT8的4倍理论加速比。

![图6. MXNet\* VNNI加速比](img/2019-07-02-fig6.png)


在图7中的延迟测试中，越短的运行时间越好。 除了SSD-VGG16之外，我们测试的模型都可以在7毫秒内完成。 特别是对于需要在移动端部署的MobileNet v1，其整个网络计算时间只需要1.01毫秒。 实际生产环境通常并不需要使用CPU中的所有核心进行推理任务。所以建议可以选取适当的计算核心（比如4个），以达到推理时间和成本的最优配比。

![图7. MXNet\* 延迟比较](img/2019-07-02-fig7.png)


除了极佳的加速，Apache / MXNet量化解决方案的准确性非常接近FP32模型的精度，如图8所示，量化后的精度损失低于0.5％。

![图8. MXNet\* 融合和量化精度对比](img/2019-07-02-fig8.png)


### 6. 总结

- 对于图像分类和物体检测的CNN网络，经过运算符融合和INT8量化后进行推理能够带来显著的加速。

- 量化后的INT8模型的精度与FP32模型的精度很接近，差异小于0.5%。

- 第二代英特尔至强可扩展处理器，使用英特尔DL Boost和新的VNNI指令集，能够让用户无需修改代码即可进一步提升模型的计算性能（~4X）。

### 7. 感谢

感谢Apache社区和亚马逊MXNet团队的支持。感谢 [Mu Li](https://github.com/mli), [Jun Wu](https://github.com/reminisce), [Da Zheng](https://github.com/zheng-da/), [Ziheng Jiang](https://github.com/zihengjiang/), [Sheng Zha](https://github.com/szha), [Anirudh Subramanian](https://github.com/anirudh2290), Kim Sukwon, [Haibin Lin](https://github.com/eric-haibin-lin), [Yixin Bao](https://github.com/ElaineBao), [Emily Hutson](https://www.linkedin.com/in/emilyhutson/) , [Emily Backus](https://www.linkedin.com/in/backusemily/)提供的帮助。另外也感谢Apache MXNet的用户提出的很多中肯的意见和建议。

### 8. 附录

| Models        | Data Shape | Batch Size | Metric              | C5.24xlarge      Base | C5.24xlarge      Fusion | C5.24xlarge      Fusion      +Quantization | Speedup | FP32 Acc  | INT8 Acc  |
| ------------- | ---------- | ---------- | ------------------- | --------------------- | ----------------------- | ------------------------------------------ | ------- | --------- | --------- |
| ResNet50 V1   | 3x224x224  | 1          | Latency(ms)         | 9.18                  | 6.05                    | 2.41                                       | 3.81    | 76.48     | 76.10     |
|               |            | 64         | Throughput(img/sec) | 347.77                | 610.04                  | 2,232.67                                   | 6.42    |           |           |
| ResNet101 V1  | 3x224x224  | 1          | Latency(ms)         | 17.28                 | 12.08                   | 4.92                                       | 3.51    | 77.30     | 77.02     |
|               |            | 64         | Throughput(img/sec) | 201.88                | 316.52                  | 1,210.37                                   | 6.00    |           |           |
| MobileNet 1.0 | 3x224x224  | 1          | Latency(ms)         | 2.96                  | 2.02                    | 1.01                                       | 2.92    | 72.14     | 71.97     |
|               |            | 64         | Throughput(img/sec) | 1,070.08              | 2,222.70                | 5,778.97                                   | 5.40    |           |           |
| Inception V3  | 3x299x299  | 1          | Latency(ms)         | 13.24                 | 10.05                   | 6.07                                       | 2.18    | 77.86     | 77.95     |
|               |            | 64         | Throughput(img/sec) | 304.72                | 423.26                  | 1,344.95                                   | 4.41    |           |           |
| SSD-VGG16     | 3x300x300  | 1          | Latency(ms)         | 25.89                 | 25.25                   | 8.24                                       | 3.14    | 83.58 mAP | 83.33 mAP |
|               |            | 224        | Throughput(img/sec) | 78.01                 | 84.09                   | 317.04                                     | 4.06    |           |           |

### 9. 通知和免责申明

*性能测试中使用的软件和工作集群可能只针对Intel微处理器的性能进行了优化。*

*如SYSmark和MobileMark等的性能测试是通过特定的计算机系统、组件、软件、操作符和函数进行测量的，上述任何因素的改变都可能引起结果的改变。您需要综合其他信息和性能测试，包括与其他产品组合时该产品的性能，以帮助您全面评估您的预期购买。更多完整信息请访问[www.intel.com/benchmarks](http://www.intel.com/benchmarks)。*

*性能结果基于AWS截至2019年7月1日的测试，并不能反映所有公开可用的安全更新。没有任何产品或组件是绝对安全的。*

*测试配置：*

*重现脚本: [https://github.com/intel/optimized-models/tree/v1.0.6/mxnet/blog/medium_vnni](https://github.com/intel/optimized-models/tree/v1.0.6/mxnet/blog/medium_vnni)*

*软件: Apache MXNet 1.5.0b20190623*

*硬件: AWS EC2 c5.24xlarge实例， 基于英特尔第二代可扩展至强处理器（Cascade Lake），全核频率为3.6GHz，单核最高频率可达3.9GHz*

*英特尔、英特尔标识和英特尔至强是英特尔公司或其在美国和/或其他国家的子公司的商标。\*其他名称和品牌可能被视为他人的财产。©英特尔公司*