#在 Gluon 里hybridize 动态模型

之前我们一直跟 Gluon 用户说，如果想 hybridize Gluon 的模型，请不要在模型里使用条件判断或循环。原因是 MXNet 的计算图只支持静态的计算，而不支持各种动态的跳转。可是现在的模型变得越来越复杂，越来越动态开，很多时候，模型需要运行的计算取决于用户的输入。比较典型的例子是 NLP 模型：我们需要在序列数据上运行 RNN cell，但是这些序列数据的长度是未知的或不定长的。这个时候我们希望我们的计算图有循环跳转来处理序列数据里的每个单元。

我们在最近发布的 MXNet v1.3 里加入了三个 control flow operators：cond， while_loop 和 foreach，来解决这个动态计算的问题。有了这些 operators 之后，我们的计算图可以记录更复杂的计算。现在我们可以方便地在 Gluon 模型里使用条件判断和循环，并且可以导出整个 Gluon 模型来实现多语言的部署。

![](img/control_flow.png)

上面的示意图展示了这三个 operator 的功能：

* [cond](https://mxnet.incubator.apache.org/api/python/symbol/contrib.html?highlight=while_loop#mxnet.symbol.contrib.cond) 是通过一个条件判断来决定运行 then_func 函数还是运行 else_func，并且返回 then_func 或者 else_func 运行的结果。
* [while_loop](https://mxnet.incubator.apache.org/api/python/symbol/contrib.html?highlight=while_loop#mxnet.symbol.contrib.while_loop) 是运行一个循环，每次迭代都会用 cond 来判断循环是否结束。loop_func 的返回值分成两部分：第一部分的返回值会作为整个循环的输出被串起来，第二部分的返回值会传给下一次的迭代，成为下一次迭代的输入。
* [foreach](https://mxnet.incubator.apache.org/api/python/symbol/contrib.html?highlight=while_loop#mxnet.symbol.contrib.foreach) 也是运行一个循环。它其实是一种特殊的循环：每次迭代会从一个 NDArray 里取出一个切片（slice）作为这次迭代的一个输入，迭代的次数和这个 NDArray 的长度相关。loop_func 的返回值和 while_loop 里的 loop_func 是一样的。所以 foreach 其实可以看成是 while_loop 的一个特殊形式。

具体的使用方法，请参考这些 operator 的文档。

下面我们用一个简单的例子来演示如何在 Gluon 里使用 control flow operators。举一个简单的例子，如果我们想在序列数据上运行 RNN，并且想得到每一步的状态，我们可以用 foreach 很简单的实现这个功能。

```python
class ForeachRNN(gluon.HybridBlock):
    def __init__(self, cell, prefix=None, params=None):
        super(ForeachRNN, self).__init__(prefix=prefix, params=params)
        self.cell = cell

    def hybrid_forward(self, F, inputs, states):
        def body(data, states):
            out, states = self.cell(data, states)
            return [out, states], states
        outs, _ = F.contrib.foreach(body, inputs, states)
        return outs

data = mx.nd.random.normal(shape=(100, 32, 256))
rnn = ForeachRNN(gluon.rnn.RNNCell(hidden_size=512))
rnn.initialize()
outs = rnn(data, [mx.nd.normal(shape=(32, 512))])
```

当然我们这里实现的功能是非常简单的。如何使用这些 operator 来实现更复杂的计算，请参考这个详细的[教程](https://mxnet.incubator.apache.org/tutorials/control_flow/ControlFlowTutorial.html)。这个教程提供了不少的使用范例。如果想参考更复杂的例子，有兴趣的朋友可以参考 Gluon-NLP 里的代码。比如 Gluon-NLP 中的 [beam search](https://github.com/dmlc/gluon-nlp/blob/5895627d1134fc35a28f95d69c402ac781f99ce4/gluonnlp/model/sequence_sampler.py#L551) 就是用这些 control flow operator 实现的。

虽说我们加入这些 operator 是为了能更方便的在 Gluon 里 hybridize 模型，使用这些 operator 还是能提升计算的速度。至于能提升多少，很大程度上取决于计算本身。比如说一个循环本身不需要太多的计算（例如每次只处理一个 sample），使用 control flow operator 在 hybridize 之后就能带来不少的速度提升；反之，速度的提升就会比较小。如下图所示，当 batch size 是1的时候，control flow operator 能让 RNN inference 变快 80%。

![](img/cf_speedup.png)

现在我们的 control flow operator 还是处于实验阶段，所以它们在 contrib 里。希望大家能给我们提供更多的回馈来帮助我们提高用户的使用体验和性能。