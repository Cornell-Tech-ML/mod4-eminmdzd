"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """Multiply x by y"""
    return x * y


# - id
def id(x: float) -> float:
    """Return input"""
    return x


# - add
def add(x: float, y: float) -> float:
    """Add x to y"""
    return x + y


# - neg
def neg(x: float) -> float:
    """Negate x"""
    return -x


# - lt
def lt(x: float, y: float) -> float:
    """Check if x is less than y"""
    return 1.0 if x < y else 0.0


# - eq
def eq(x: float, y: float) -> float:
    """Check if x is equal to y"""
    return 1.0 if x == y else 0.0


# - max
def max(x: float, y: float) -> float:
    """Return the maximum of x and y"""
    if x > y:
        return x
    return y


# - is_close
def is_close(x: float, y: float) -> bool:
    """Check if x and y are separated by at most 1e-2"""
    return (x - y < 1e-2) and (y - x < 1e-2)


# - sigmoid
def sigmoid(x: float) -> float:
    """Implements sigmoid function on x"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    """Implements relu (rectified linear unit) function on x"""
    return x if x > 0 else 0.0


EPS = 1e-6


# - log
def log(x: float) -> float:
    """Implements logarithm of x"""
    return math.log(x + EPS)


# - exp
def exp(x: float) -> float:
    """Implements exponential function of x"""
    return math.exp(x)


# - log_back
def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return y / (x + EPS)


# - inv
def inv(x: float) -> float:
    """Returns inverse of a number x as 1/x"""
    return 1.0 / x


# - inv_back
def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return -y / (pow(x, 2))


# - relu_back
def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU times a second arg"""
    if x <= 0:
        return 0
    return y


def sigmoid_back(x: float, y: float) -> float:
    """Computes the derivative of sigmoid times a second arg"""
    return y * sigmoid(x) * (1 - sigmoid(x))


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map.

    See https://en.wikipedia.org/wiki/Map_(higher-order_function)

    Args:
    ----
        fn: Function from one value to one value.

    Returns:
    -------
        A function that takes a list, applies 'fn' to each element, and returns a new list

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


# - zipWith
def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith."""

    def _zipWith(l1: Iterable[float], l2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(l1, l2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


# - reduce
def reduce(
    fn: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce."""

    def _reduce(ls: Iterable[float]) -> float:
        ret = init
        for x in ls:
            ret = fn(ret, x)
        return ret

    return _reduce


# Use these to implement
# - negList : negate a list
def negList(x: Iterable[float]) -> Iterable[float]:
    """Negates all elements in a list"""
    return map(neg)(x)


# - addLists : add two lists together
def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    """Adds the elements of 2 lists"""
    return zipWith(add)(x, y)


# - sum: sum lists
def sum(x: Iterable[float]) -> float:
    """Sums all the elements of a list"""
    return reduce(add, 0)(x)


# - prod: take the product of lists
def prod(x: Iterable[float]) -> float:
    """Multiply all the elements of a list"""
    return reduce(mul, 1)(x)
