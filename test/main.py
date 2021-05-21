# Test

import numpy as np
import torch
import torch.autograd
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.variables import Variable
from src.functions import *


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


def main():

    # test_add()
    # test_sub()
    # test_mul()
    # test_div()
    # test_pow()
    # test_exp()
    # test_log()
    # test_sigmoid()
    # test_relu()
    # test_reshape()
    # test_sum()
    # test_mean()
    test_softmax()



if __name__ == "__main__":
    main()