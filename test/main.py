# Test

import numpy as np
import torch
import torch.autograd

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
    # print(c_shadl.data)
    # print(c_torch.data)
    # print(a_shadl.grad.data, b_shadl.grad.data)
    # print(a_torch.grad.data, b_torch.grad.data)
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
    # print(c_shadl.data)
    # print(c_torch.data)
    # print(a_shadl.grad.data, b_shadl.grad.data)
    # print(a_torch.grad.data, b_torch.grad.data)
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
    # print(c_shadl.data)
    # print(c_torch.data)
    # print(a_shadl.grad.data, b_shadl.grad.data)
    # print(a_torch.grad.data, b_torch.grad.data)
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
    # print(c_shadl.data)
    # print(c_torch.data)
    # print(a_shadl.grad.data, b_shadl.grad.data)
    # print(a_torch.grad.data, b_torch.grad.data)
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
    # print(c_shadl.data)
    # print(c_torch.data)
    # print(a_shadl.grad.data, b_shadl.grad.data)
    # print(a_torch.grad.data, b_torch.grad.data)
    print(np.all(np.isclose(c_shadl.data, c_torch.data)))
    print(np.all(np.isclose(a_shadl.grad.data, a_torch.grad.data)))
    print(np.all(np.isclose(b_shadl.grad.data, b_torch.grad.data)))


def main():

    test_add()
    test_sub()
    test_mul()
    test_div()
    test_pow()


if __name__ == "__main__":
    main()