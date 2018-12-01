---
title: 深度学习在文本分类问题上的应用
author: 何孝霆 中国科学院 硕士在读
---


最近经常做一些文本分类任务，就来聊聊深度学习在文本分类问题上的应用。

文本分类任务是自然语言处理领域的基础任务之一，根据数据领域的不同又可以分为内容分类和情感分类等。卷积神经网络（CNN）和循环神经网络（RNN）都在文本分类中有很多应用。
CNN的如Kim[1]在EMNLP2014上和Kalchbrenner等人[2]在ACL2014提出的文本分类模型。RNN的如Tang等人[3]在EMNLP2015提出的文档级情感分类模型

RNN模型擅长对整体句子结构进行建模，理论上可以捕捉长距离依赖信息（可能仅仅是“理论上”）。而事实上，大多数情况下文本的局部短语信息非常关键，RNN反而无法关注这一重要信息。CNN模型由于擅长抽取空间上的局部特征，故更善于捕捉这种局部短语信息。

2014年的《Convolutional Neural Networks for Sentence Classification》Yoon Kim在EMNLP2014上提出 ，简称textCNN。虽然模型比较老，但现在很多工业应用或者比赛还是会使用。模型简单，好吃不贵，效果也很不错。我在现在的实际应用中，也会经常拿预训练的textCNN做其他任务的encoder。

我在《动手学深度学习》一书中贡献的“文本情感分类：使用卷积神经网络（textCNN）”章节，对textCNN模型进行了具体的讲解：
> TextCNN 主要使用了一维卷积层和时序最大池化层。假设输入的文本序列由  n  个词组成，每个词用  d  维的词向量表示。那么输入样本的宽为  n ，高为 1，输入通道数为  d 。textCNN 的计算主要分为以下几步：
> 1. 定义多个一维卷积核，并使用这些卷积核对输入分别做卷积计算。宽度不同的卷积核可能会捕捉到不同个数的相邻词的相关性。
> 2. 对输出的所有通道分别做时序最大池化，再将这些通道的池化输出值连结为向量。
> 3. 通过全连接层将连结后的向量变换为有关各类别的输出。这一步可以使用丢弃层应对过拟合。

## 预处理：
* 英文可以考虑使用spaCy进行分词，比用空格分词要好一些。同时使用pyenchant进行拼写检查。
* vocab可以基于词频或者计算TFIDF去掉一些高频词和出现次数极少的词。
* 对于补定长，textCNN由于max-over-time pooling的存在，对填充并不敏感，所以可以取文本最大长度为定长。但实际中，由于长尾效应的存在，可以考虑把定长设为能覆盖90%~95%文本的长度，避免训练过慢。

## 嵌入层：
对于嵌入层，textCNN模型提出了四种变体。可以直接在textCNN过程中训练出词向量(**CNN-rand**)，也可以使用word2vec或glove预训练好的词向量。使用预训练词向量，可以分为**CNN-static**和**CNN-non-static**，这两者的区别是要不要在训练过程中更新嵌入层的参数。
按照原文的结果，non-static一般要比static好一点点。但每个batch都更新嵌入层的话会比较慢，故实际中还有一种介于两者中间的玩法，即间隔若干batch更新一次嵌入层的参数。
还有一种变体是**CNN-multichannel**，就是使用两个channel的词向量矩阵拼接作为嵌入层，其中一个channel的词向量在训练过程中不更新，另一个channel的词向量可以在训练过程中被更新。文章指出，也可以在CNN-static的基础上，增加几维，后面加的维度可以被训练。这样也可以做到CNN-multichannel的思想。
对于未登录词(\<UNK>)，可以用全0或者随机初始化，论文作者认为可以用预训练词向量矩阵的方差做正态分布来进行初始化。

* 对于词向量的选择，可以使用开源预训练好的词向量，实际体验上使用GloVe或word2vec对结果有影响，在实际应用中感觉GloVe一般表现的比word2vec要好一些。在Gluon-NLP中提供了这些预训练好的词向量，可以根据任务来选择。
* 如果训练集比较大，建议使用**CNN-non-static**，即允许词向量在训练过程中被更新，可能取得更好的效果。如果训练集比较小，建议还是用**CNN-static**。

## 卷积层&池化层：

定义多组不同宽度的卷积核，就类似于提取n-gram语义特征。
关于超参数设定，可以看下Ye Zhang等人[4]的文章《A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification》
* 不同的数据集都有自己最佳的卷积核宽度，建议采用网格搜索来找到最佳的卷积核宽度，大多数文本的最佳范围都是1-10之间。少部分的长文本数据，这个最佳值可能要更大。如CR数据集，最大句子长度105，最佳的卷积核宽度35-36.
* 找到这个“最佳”的大小之后，尝试这个“最佳”值附近值的组合，即结合最佳值及最佳值的附近值。
* 卷积核数目对结果有影响，对于大多数文本，这个数目在100-600之间，较大的卷积核数目除了更容易产生过拟合，还会导致模型训练需要更长时间。
* 激活函数可以选择tanh、ReLU。在部分数据集上Iden(不使用激活函数)可以取得最佳的效果，这表明在部分数据集上，线性变换足以捕获词嵌入和类别标签的相关性。
* 对于池化层，1-max pooling基本上总是优于其他方式，如k-max pooling和avg pooling。


## 参考文献：
1. Kim, Y. (2014). Convolutional neural networks for sentence classification. arXiv preprint arXiv:1408.5882.
2. Kalchbrenner, N., Grefenstette, E., & Blunsom, P. (2014). A convolutional neural network for modelling sentences. arXiv preprint arXiv:1404.2188.
3. Tang, D., Qin, B., & Liu, T. (2015). Document modeling with gated recurrent neural network for sentiment classification. In Proceedings of the 2015 conference on empirical methods in natural language processing (pp. 1422-1432).
4. Zhang, Y., & Wallace, B. (2015). A sensitivity analysis of (and practitioners' guide to) convolutional neural networks for sentence classification. arXiv preprint arXiv:1510.03820.
