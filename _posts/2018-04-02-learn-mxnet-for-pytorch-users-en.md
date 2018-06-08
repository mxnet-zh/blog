---
title: Switching from PyTorch to MXNet in 10 minutes
author: Mu Li Amazon Principal Scientist
translator: Hao Jin Amazon Software Development Engineer

---

[PyTorch](pytorch.org) is a purely imperative deep learning framework. It’s quickly gaining popularity due to its easy-to-understand programming interface.

Maybe this is news to you, but our MXNet is also providing a programming interface that resembles PyTorch’s through `NDArray` and `Gluon`. We’ll compare how the two frameworks implement the same algorithm.

![](img/pytorch-to-mxnet.png){:width="500px"}

## Installation

PyTorch recommends installation through conda:

```bash
conda install pytorch-cpu -c pytorch
```

MXNet recommends installation through pip, here we are using --pre to install the nightly build version:

```bash
pip install --pre mxnet
```

## Multi-dimensional Matrices

For Multi-dimensional matrices, PyTorch inherits the naming from Torch and calls them tensor, while MXNet follows NumPy’s naming and calls them ndarray. The following code snippets show how to create a 2-dimensional matrix with all entries initialized to 1, then add 1 to each entry, and finally print it out.

- PyTorch:

  ```python
  import torch
  x = torch.ones(5,3)
  y = x + 1
  print(y)
  ```
  ```
   2  2  2
   2  2  2
   2  2  2
   2  2  2
   2  2  2
  [torch.FloatTensor of size 5x3]
  ```

- MXNet:

  ```python
  from mxnet import nd
  x = nd.ones((5,3))
  y = x + 1
  print(y)
  ```
  ```
  [[2. 2. 2.]
   [2. 2. 2.]
   [2. 2. 2.]
   [2. 2. 2.]
   [2. 2. 2.]]
  <NDArray 5x3 @cpu(0)>
  ```




We can see that the only difference here is that the shape arguments in MXNet need to be wrapped by parenthesis, which is the same as NumPy.

## Model Training

Now let’s move on to a more complicated example: Training an Multi-Level Perceptron(MLP) with MNIST dataset. For readability, we split the whole process into 4 sub-parts.

### Loading the Dataset

Here we show how to download the MNIST dataset and load into memory, so that we can easily read data in batches later.

- PyTorch:

  ```python
  import torch
  from torchvision import datasets, transforms

  train_data = torch.utils.data.DataLoader(
      datasets.MNIST(train=True, transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.13,), (0.31,))])),
      batch_size=128, shuffle=True, num_workers=4)
  ```

- MXNet:

  ```python
  from mxnet import gluon
  from mxnet.gluon.data.vision import datasets, transforms

  train_data = gluon.data.DataLoader(
      datasets.MNIST(train=True).transform_first(transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize(0.13, 0.31)])),
  	batch_size=128, shuffle=True, num_workers=4)
  ```


Note that the difference here is that MXNet is using transform_first to denote that the transformation of the data happens to the first element of the example that’s being read while the other elements are not affected. For MNIST dataset, this means the transformation happens to the pictures of handwritten digits, not to the labels.

### Defining the Model

Here we define an MLP with a single hidden layer:

- PyTorch:

  ```python
  from torch import nn

  net = nn.Sequential(
      nn.Linear(28*28, 256),
      nn.ReLU(),
      nn.Linear(256, 10)
  )
  ```

- MXNet:

  ```python
  from mxnet.gluon import nn

  net = nn.Sequential()
  with net.name_scope():
      net.add(
          nn.Dense(256, activation='relu'),
          nn.Dense(10)
      )
  net.initialize()
  ```



We used `Sequential` container to build the network through combination of layers. Here the main differences between PyTorch and MXNet are:

- There’s no need to specify input size for MXNet, as it will be inferred by the system later
- MXNet allows specification of activation functions for fully connected layers and convolutional layers
- For MXNet a name scope `name_scope` is needed to give a unique name to every layer, which will be needed when we access the model.
- An explicit call to initialize the network is necessary for MXNet

We all know that `Sequential` containers can only execute the computation in a layer-by-layer fashion. To customize how the `forward` function is executed, we can build models that inherit from `nn.Module` in PyTorch, similarly, we can build models that inherit from `nn.Block` to do the same thing.

### Loss function and Optimizer

- PyTorch:

  ```python
  loss_fn = nn.CrossEntropyLoss()
  trainer = torch.optim.SGD(net.parameters(), lr=0.1)
  ```

- MXNet:

  ```python
  loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
  trainer = gluon.Trainer(net.collect_params(),
                          'sgd', {'learning_rate': 0.1})
  ```

Here we show how to specify cross-entropy loss function and Stochastic Gradient Descent optimizer with learning rate of 0.1.

### Training the Model

Finally we implement the training process with the results attached. Note that every time we may get different initial weights and read the data in different order, so the final results may vary.

- PyTorch

  ```python
  from time import time
  for epoch in range(5):
      total_loss = .0
      tic = time()
      for X, y in train_data:
          X, y = torch.autograd.Variable(X), torch.autograd.Variable(y)
          trainer.zero_grad()
          loss = loss_fn(net(X.view(-1, 28*28)), y)
          loss.backward()
          trainer.step()
          total_loss += loss.mean()
      print('epoch %d, avg loss %.4f, time %.2f' % (
          epoch, total_loss/len(train_data), time()-tic))
  ```

  ```
  epoch 0, avg loss 0.3251, time 3.71
  epoch 1, avg loss 0.1509, time 4.05
  epoch 2, avg loss 0.1057, time 4.07
  epoch 3, avg loss 0.0820, time 3.70
  epoch 4, avg loss 0.0666, time 3.63
  ```

- MXNet

  ```python
  from time import time
  for epoch in range(5):
      total_loss = .0
      tic = time()
      for X, y in train_data:
          with mx.autograd.record():
  	        loss = loss_fn(net(X.flatten()), y)
          loss.backward()
          trainer.step(batch_size=128)
          total_loss += loss.mean().asscalar()
      print('epoch %d, avg loss %.4f, time %.2f' % (
          epoch, total_loss/len(train_data), time()-tic))
  ```

  ```
  epoch 0, avg loss 0.3162, time 1.59
  epoch 1, avg loss 0.1503, time 1.49
  epoch 2, avg loss 0.1073, time 1.46
  epoch 3, avg loss 0.0830, time 1.48
  epoch 4, avg loss 0.0674, time 1.75
  ```



We notice the main differences between PyTorch and MXNet are:

- There’s no need to feed the inputs into `Variable`, but we need to wrap the forward computation in `mx.autograd.record()` for derivative calculation later.
- No need to reset the gradients to 0 for each batch, as the gradients are overwritten, not accumulated in MXNet.
- Batch size is needed for each `step` in MXNet.
- Need to call `asscalar()` to cast multidimensional matrix to scalar when using MXNet
- For this example we observe a speedup of about 2 compared to PyTorch, but we also need to be cautious about this number.
