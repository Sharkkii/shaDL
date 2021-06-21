# Test

import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import Recorder1d
from src.variables import *
from src.functions import *
from src.optimizers import *
from src.networks import *
from src.data import *


def test_utils():
    print("#### Utils ####")
    recorder = Recorder1d(n_capacity=5, alpha=0.5)
    a = np.arange(20)[::-1]
    for i in range(20):
        recorder.record(a[i])
        assert(np.isclose(recorder.average, np.mean(a[:i+1])))
        assert(np.isclose(recorder.simple_moving_average, np.mean(a[max(i-4,0):i+1])))

def test_add():
    print("#### Add ####")
    np.random.seed(0)
    a = np.random.randn(3)
    b = np.random.randn(3)
    a_shadl = Variable(a.copy())
    b_shadl = Variable(b.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)

    c_shadl = a_shadl + b_shadl; c_shadl.backward()
    c_torch = a_torch + b_torch; d_torch = torch.sum(c_torch); d_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_sub():
    print("#### Sub ####")
    np.random.seed(0)
    a = np.random.randn(3)
    b = np.random.randn(3)
    a_shadl = Variable(a.copy())
    b_shadl = Variable(b.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)

    c_shadl = a_shadl - b_shadl; c_shadl.backward()
    c_torch = a_torch - b_torch; d_torch = torch.sum(c_torch); d_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_mul():
    print("#### Mul ####")
    np.random.seed(0)
    a = np.random.randn(3)
    b = np.random.randn(3)
    a_shadl = Variable(a.copy())
    b_shadl = Variable(b.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)

    c_shadl = a_shadl * b_shadl; c_shadl.backward()
    c_torch = a_torch * b_torch; d_torch = torch.sum(c_torch); d_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_div():
    print("#### Div ####")
    np.random.seed(0)
    a = np.random.randn(3)
    b = np.random.randn(3)
    a_shadl = Variable(a.copy())
    b_shadl = Variable(b.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)

    c_shadl = a_shadl / b_shadl; c_shadl.backward()
    c_torch = a_torch / b_torch; d_torch = torch.sum(c_torch); d_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_pow():
    print("#### Pow ####")
    np.random.seed(0)
    a = np.random.randn(3)
    b = np.random.randn(3)
    a_shadl = Variable(a.copy())
    b_shadl = Variable(b.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)

    c_shadl = a_shadl ** b_shadl; c_shadl.backward()
    c_torch = a_torch ** b_torch; d_torch = torch.sum(c_torch); d_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_exp():
    print("#### Exp ####")
    np.random.seed(0)
    a = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = exp(a_shadl); b_shadl.backward()
    b_torch = torch.exp(a_torch); c_torch = torch.sum(b_torch); c_torch.backward()
    # print(c_shadl.data)
    # print(c_torch.data)
    # print(a_shadl.grad.data)
    print(np.all(np.isclose(b_shadl.data, b_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))

def test_log():
    print("#### Log ####")
    np.random.seed(0)
    a = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = log(a_shadl); b_shadl.backward()
    b_torch = torch.log(a_torch); c_torch = torch.sum(b_torch); c_torch.backward()
    print(np.all(np.isclose(b_shadl.data, b_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))

def test_sigmoid():
    print("#### Sigmoid ####")
    np.random.seed(0)
    a = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = sigmoid(a_shadl); b_shadl.backward()
    b_torch = torch.sigmoid(a_torch); c_torch = torch.sum(b_torch); c_torch.backward()
    print(np.all(np.isclose(b_shadl.data, b_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))

def test_tanh():
    print("#### Tanh ####")
    np.random.seed(0)
    a = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = tanh(a_shadl); b_shadl.backward()
    b_torch = torch.tanh(a_torch); c_torch = torch.sum(b_torch); c_torch.backward()
    print(np.all(np.isclose(b_shadl.data, b_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))

def test_relu():
    print("#### ReLU ####")
    np.random.seed(0)
    a = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = relu(a_shadl); b_shadl.backward()
    b_torch = F.relu(a_torch); c_torch = torch.sum(b_torch); c_torch.backward()
    print(np.all(np.isclose(b_shadl.data, b_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))

def test_reshape():
    print("#### Reshape ####")
    np.random.seed(0)
    a = np.random.randn(2,3,2)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    shape = (3,4)

    b_shadl = reshape(a_shadl, shape); c_shadl = exp(b_shadl); c_shadl.backward()
    b_torch = torch.reshape(a_torch, shape); b_torch.retain_grad(); c_torch = torch.exp(b_torch)
    d_torch = torch.sum(c_torch); d_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_getitem():
    print("#### Getitem ####")
    np.random.seed(0)
    a = np.random.randn(3,4)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    idx = (slice(0,2), slice(1,3))
    b_shadl = a_shadl[idx]; c_shadl = sigmoid(b_shadl); c_shadl.backward()
    # b_shadl = getitem(a_shadl, idx); c_shadl = sigmoid(b_shadl); c_shadl.backward()
    b_torch = a_torch[idx]; b_torch.retain_grad(); c_torch = torch.sigmoid(b_torch)
    d_torch = torch.sum(c_torch); d_torch.backward()

    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

    # e_shadl = Variable(a.copy())
    # f_shadl = sigmoid(e_shadl)
    # f_shadl.backward()
    # print(c_shadl)
    # print(f_shadl)
    # print(b_shadl.grad)
    # print(e_shadl.grad)
    

def test_sum():
    print("#### Sum ####")
    np.random.seed(0)
    a = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = sum(a_shadl); c_shadl = exp(b_shadl); c_shadl.backward()
    b_torch = torch.sum(a_torch); b_torch.retain_grad(); c_torch = torch.exp(b_torch); c_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_mean():
    print("#### Mean ####")
    np.random.seed(0)
    a = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = mean(a_shadl); c_shadl = exp(b_shadl); c_shadl.backward()
    b_torch = torch.mean(a_torch); b_torch.retain_grad(); c_torch = torch.exp(b_torch); c_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_softmax():
    print("#### Softmax ####")
    np.random.seed(0)
    a = np.random.randn(3,4)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)

    b_shadl = softmax(a_shadl); b_shadl.backward()
    b_torch = torch.softmax(a_torch, dim=1); c_torch = torch.sum(b_torch); c_torch.backward()
    # print(a_shadl.grad.data)
    # print(a_shadl.grad.data)
    print(np.all(np.isclose(b_shadl.data, b_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))

def test_dot():
    print("#### Dot ####")
    np.random.seed(0)
    a = np.random.randn(3)
    b = np.random.randn(3)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_shadl = Variable(b.copy())
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)

    c_shadl = dot(a_shadl, b_shadl); c_shadl.backward()
    c_torch = torch.dot(a_torch, b_torch); d_torch = torch.sum(c_torch); d_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_linear():
    print("#### Linear ####")
    np.random.seed(0)
    a = np.random.randn(3,4)
    b = np.random.randn(5,4)
    c = np.random.randn(1,5)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_shadl = Variable(b.copy())
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)
    c_shadl = Variable(c.copy())
    c_torch = torch.autograd.Variable(torch.tensor(c.copy()), requires_grad=True)

    d_shadl = linear(a_shadl, b_shadl, c_shadl); d_shadl.backward()
    d_torch = F.linear(a_torch, b_torch, c_torch); e_torch = torch.sum(d_torch); e_torch.backward()
    print(np.all(np.isclose(d_shadl.data, d_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))
    print(np.all(np.isclose(c_shadl.grad.data, c_torch.grad.data)))

def test_mse():
    print("#### MSE ####")
    np.random.seed(0)
    a = np.random.randn(3,4)
    b = np.random.randn(3,4)
    a_shadl = Variable(a.copy())
    a_torch = torch.autograd.Variable(torch.tensor(a.copy()), requires_grad=True)
    b_shadl = Variable(b.copy())
    b_torch = torch.autograd.Variable(torch.tensor(b.copy()), requires_grad=True)

    c_shadl = mean_squared_error(a_shadl, b_shadl); c_shadl.backward()
    c_torch = F.mse_loss(a_torch, b_torch, reduction="mean"); c_torch.backward()
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))

def test_ffnn():
    print("#### FFNN ####")
    np.random.seed(0)

    # print("FFNN setup")

    x = Variable(np.random.uniform(low=-1, high=1, size=(100,1)))
    y = 2 * x - 1

    l1 = Linear(d_in=1, d_out=5, name="Linear1")
    a1 = Sigmoid(name="Sigmoid1")
    l2 = Linear(d_in=5, d_out=5, name="Linear2")
    a2 = ReLU(name="ReLU2")
    l3 = Linear(d_in=5, d_out=1)
    ll = LayerList(l1, a1, l2, a2, l3)
    n = FeedForwardNeuralNetwork(ll, name="FFNN")
    opt = SGD(n.parameters(), lr=1e-2)

    N_epoch = 2000
    for epoch in range(N_epoch):

        # print("FFNN.forward")

        y_pred = n(x)
        loss = mean_squared_error(y_pred, y)
        # print(y_pred, y)
        if ((epoch+1) % 100 == 0):
            print("%d %f" % (epoch+1, loss.data))

        # print("FFNN.backward")

        opt.reset()
        loss.backward()
        opt.step()

def test_dataloader():

    print("#### DataLoader ####")
    np.random.seed(0)
    x = np.random.uniform(low=-1, high=1, size=(100,1))
    y = 2 * x - 1
    dataset = Dataset(x, y)
    dataloader = DataLoader(dataset, n_batch=10, do_shuffle=True)
    # for x, y in dataloader:
    #     print(x, y)

    l1 = Linear(d_in=1, d_out=5, name="Linear1")
    a1 = Sigmoid(name="Sigmoid1")
    l2 = Linear(d_in=5, d_out=5, name="Linear2")
    a2 = ReLU(name="ReLU2")
    l3 = Linear(d_in=5, d_out=1)
    ll = LayerList(l1, a1, l2, a2, l3)
    n = FeedForwardNeuralNetwork(ll, name="FFNN")
    opt = SGD(n.parameters(), lr=1e-2)

    N_epoch = 100
    for epoch in range(N_epoch):

        for i, (x, y) in enumerate(dataloader):

            x, y = Variable(x), Variable(y)
            y_pred = n(x)
            loss = mean_squared_error(y_pred, y)

            opt.reset()
            loss.backward()
            opt.step()

        if ((epoch+1) % 10 == 0):
            print("%d %f" % (epoch+1, loss.data))

def test_sequential_dataloader():

    print("#### Sequential DataLoader ####")
    np.random.seed(0)
    x = np.linspace(start=-1, stop=1, num=60).reshape(-1,1)
    y = 2 * x - 1
    dataset = SequentialDataset(x, y)
    dataloader = SequentialDataLoader(dataset, n_batch=10, l_bptt=3, do_shuffle=False)
    for x, y in dataloader:
        print(x, y)

def test_rnn():

    print("#### RNN ####")
    np.random.seed(0)
    l_seq = 1000
    l_bptt = 5
    n_batch = 2
    d_hidden = 8
    n_epoch = 100
    n_iter_per_epoch = int(l_seq / (n_batch * l_bptt))

    x = np.linspace(0, 1, l_seq+1, endpoint=True).reshape(-1,1)
    noise = np.random.randn(*x.shape)
    x = np.sin(2 * np.pi * x) + noise * 0.05
    x, y = x[:-1], x[1:]
    dataset = SequentialDataset(x, y)
    dataloader = SequentialDataLoader(dataset, n_batch=n_batch, l_bptt=l_bptt, do_shuffle=True)

    class Model:
        def __init__(self, l_seq):
            self.rnn = RNN(d_in=1, d_hidden=d_hidden, name="RNN")
            self.linear = Linear(d_in=d_hidden, d_out=1, name="Linear")
            # self.linear = Linear(d_in=l_bptt*d_hidden, d_out=1, name="Linear")
        def __call__(self, x):
            x, _ = self.rnn(x)
            x = x[-1, :, :]
            # x = reshape(x, (-1, l_bptt*d_hidden))
            x = self.linear(x)
            return x
        def parameters(self):
            return LayerList(self.rnn, self.linear).parameters()

    model = Model(l_seq)
    opt = SGD(model.parameters(), lr=5e-3)
    recorder = Recorder1d(n_capacity=1, alpha=1)
    for epoch in range(n_epoch):

        for idx, (x, y) in enumerate(dataloader):

            x, y = Variable(x), Variable(y)
            y_pred = model(x)
            loss = mean_squared_error(y_pred, y.data[-1, :, :])
            recorder.record(loss.data)
            
            opt.reset()
            loss.backward()
            opt.step()

            if ((idx+1) % (int(n_iter_per_epoch / 1)) == 0):
                print("epoch (%d,%d) | loss %4.4f" % (epoch+1, idx+1, recorder.average))
                recorder.reset()


def main():

    # test_utils()
    # test_add()
    # test_sub()
    # test_mul()
    # test_div()
    # test_pow()
    # test_exp()
    # test_log()
    # test_sigmoid()
    # test_relu()
    # test_tanh()
    # test_reshape()
    # test_getitem()
    # test_sum()
    # test_mean()
    # test_softmax()
    # test_dot()
    # test_linear()
    # test_mse()
    # test_ffnn()
    # test_dataloader()
    # test_sequential_dataloader()
    test_rnn()



if __name__ == "__main__":
    main()