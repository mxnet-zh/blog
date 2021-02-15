---
title: Hybridizing dynamic models in Gluon
author: Da Zheng Amazon AI Applied Scientist
---

As deep learning models are becoming increasingly complex and dynamic these days, the computation of a model would become dependent on user inputs. For example, we want to run RNN models on sequential data, whose length are either unknown or not fixed. We need to use loops or jumps to perform the dynamic computation. In Gluon, we use Python conditions and loops to implement the dynamic model.

Gluon has a hybridization mode that automatically turns Python code into a computation graph. The benefit of hybridization is to improve performance and simplify deployment of a Gluon model in multiple languages and different platforms. Please see [this tutorial](https://mxnet.incubator.apache.org/tutorials/gluon/hybrid.html) for more details of Gluon hybridization. One of the limitations in Gluon hybridization is that Gluon's computation graph previously only supported static computation. For example, when running an RNN sequence in Gluon hybrdization, the RNN has to be explicitly unrolled for a predefined length (the figure below shows an RNN computation graph for a sequence of length 3). Thus, we usually suggested not using conditional jumps or loops in Gluon if you want to use hybridization.

![](https://raw.githubusercontent.com/zheng-da/mxnet-zh-blog/control_flow/img/static_RNN.png)

Since MXNet v1.3, we introduced 3 control flow operators: cond, while_loop and foreach to enable dynamic computation in Gluon computation graph. With those operators, we can now use conditional jumps and loops in Gluon models to simplify the dynamic model deployment.

![](https://raw.githubusercontent.com/zheng-da/mxnet-zh-blog/control_flow/img/control_flow.png)

The diagram above shows the functionality of the 3 operators:

* [cond](https://mxnet.incubator.apache.org/api/python/symbol/contrib.html?highlight=while_loop#mxnet.symbol.contrib.cond) decides whether to run `then_func` or `else_func` function based on a condition, and returns the result of the corresponding choice.

* [while_loop](https://mxnet.incubator.apache.org/api/python/symbol/contrib.html?highlight=while_loop#mxnet.symbol.contrib.while_loop) runs a loop, and detects whether the loop is finished based on `cond`. The return value of `loop_func` comes in 2 parts: the first part will become part of the whole loop, the second part will be passed to the next iteration as the input.

* [foreach](https://mxnet.incubator.apache.org/api/python/symbol/contrib.html?highlight=while_loop#mxnet.symbol.contrib.foreach) is also a loop, but it's in fact a special one: each iterations will take a slice of an NDArray as the input, thus the length of the loop will depend on the size of input NDArray. The return value of `loop_func` behaves in the same way as the `loop_func` for `while_loop`, so `foreach` could also be considered as a special form of `while_loop`.


For more details, please refer to the documentations of the operators.

Now we use a simple example to show how to use control flow operators in Gluon. In this example, we show how to run RNN models on sequential data using `foreach` operator.

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

The computation graph after Gluon hybridization is shown as below.

![](https://raw.githubusercontent.com/zheng-da/mxnet-zh-blog/control_flow/img/dynamic_RNN.png)

Obviously this example is a very simple one, for more on how to use control flow operators to implement complex computation, please refer to this detailed [tutorial](https://mxnet.incubator.apache.org/tutorials/control_flow/ControlFlowTutorial.html) which covers quite a few example use cases. For even more complex models, one can also refer to Gluon-NLP. For instance, [beam search] in Gluon-NLP was implemented using those control flow operators.

Although the main reason to introduce control flow operators is to make hybridizing Gluon models more convenient, using them could also speed up computation. The actual speed gain shall depend on the model itself. If a loop does not require too much computation (for example, 1 sample at a time), it could benefit a lot from control flow operators; vice versa, the speed gain would be small. We performed a [benchmark](https://github.com/apache/incubator-mxnet/blob/master/benchmark/python/control_flow/rnn.py) to get the performance gain of the example model above on a c5.18xlarge EC2 instance. As the image below shows, when batch size is set to 1, using control flow operators could speed up RNN inference by 80%.

![](https://raw.githubusercontent.com/zheng-da/mxnet-zh-blog/control_flow/img/cf_speedup.png){:width="700px"}

Control flow operators are still experimental at this moment, so they're put under contrib package. We appreciate more feedbacks from users so that we could keep improving MXNet's user experience and performance.
