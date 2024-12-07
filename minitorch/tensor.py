"""Implementation of the core Tensor object for autodifferentiation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import operators
from .autodiff import Context, Variable, backpropagate
from .tensor_data import TensorData

# Comment these out if not yet implemented
from .tensor_functions import (
    EQ,
    LT,
    Add,
    All,
    Copy,
    Exp,
    Inv,
    IsClose,
    Log,
    MatMul,
    Mul,
    Neg,
    Permute,
    ReLU,
    Sigmoid,
    Sum,
    View,
    tensor,
)

if TYPE_CHECKING:
    from typing import Any, Iterable, List, Optional, Sequence, Tuple, Type, Union

    import numpy.typing as npt

    from .tensor_data import Shape, Storage, Strides, UserIndex, UserShape, UserStrides
    from .tensor_functions import Function
    from .tensor_ops import TensorBackend

    TensorLike = Union[float, int, "Tensor"]


@dataclass
class History:
    """`History` stores the history of `Function` operations that was
    used to construct the current Variable.
    """

    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()


_tensor_count = 0


class Tensor:
    """Tensor is a generalization of Scalar in that it is a Variable that
    handles multidimensional arrays.
    """

    backend: TensorBackend
    history: Optional[History]
    grad: Optional[Tensor]
    _tensor: TensorData
    unique_id: int
    name: str

    def __init__(
        self,
        v: TensorData,
        back: Optional[History] = None,
        name: Optional[str] = None,
        backend: Optional[TensorBackend] = None,
    ):
        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        assert isinstance(v, TensorData)
        assert backend is not None
        self._tensor = v
        self.history = back
        self.backend = backend
        self.grad = None
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)

        self.f = backend

    def requires_grad_(self, x: bool) -> None:
        """Set the tensor to require gradients."""
        self.history = History()

    def requires_grad(self) -> bool:
        """Check if the tensor requires gradients."""
        return self.history is not None

    def to_numpy(self) -> npt.NDArray[np.float64]:
        """Return the tensor as a numpy array."""
        return self.contiguous()._tensor._storage.reshape(self.shape)

    def _ensure_tensor(self, b: TensorLike) -> Tensor:
        """Convert a Python number into a tensor."""
        if isinstance(b, (int, float)):
            c = Tensor.make([b], (1,), backend=self.backend)
        else:
            b._type_(self.backend)
            c = b
        return c

    def item(self) -> float:
        """Convert a 1-element tensor to a float."""
        assert self.size == 1
        x: float = self._tensor._storage[0]
        return x

    def contiguous(self) -> Tensor:
        """Return a contiguous tensor with the same data."""
        return Copy.apply(self)

    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return self._tensor.to_string()

    def __getitem__(self, key: Union[int, UserIndex]) -> float:
        """Get an item from the tensor by index."""
        key2 = (key,) if isinstance(key, int) else key
        return self._tensor.get(key2)

    def __setitem__(self, key: Union[int, UserIndex], val: float) -> None:
        """Set an item in the tensor by index."""
        key2 = (key,) if isinstance(key, int) else key
        self._tensor.set(key2, val)

    # Internal methods used for autodiff.
    def _type_(self, backend: TensorBackend) -> None:
        """Set the backend of the tensor."""
        self.backend = backend
        if backend.cuda:  # pragma: no cover
            self._tensor.to_cuda_()

    def _new(self, tensor_data: TensorData) -> Tensor:
        """Create a new tensor from existing data."""
        return Tensor(tensor_data, backend=self.backend)

    @staticmethod
    def make(
        storage: Union[Storage, List[float]],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
        backend: Optional[TensorBackend] = None,
    ) -> Tensor:
        """Create a new tensor from storage, shape, and strides."""
        return Tensor(TensorData(storage, shape, strides), backend=backend)

    def expand(self, other: Tensor) -> Tensor:
        """Expand the size of another tensor to match this tensor for backprop."""
        if self.shape == other.shape:
            return other
        true_shape = TensorData.shape_broadcast(self.shape, other.shape)
        buf = self.zeros(true_shape)
        self.backend.id_map(other, buf)
        if self.shape == true_shape:
            return buf
        out = buf
        orig_shape = [1] * (len(out.shape) - len(self.shape)) + list(self.shape)
        for dim, shape in enumerate(out.shape):
            if orig_shape[dim] == 1 and shape != 1:
                out = self.backend.add_reduce(out, dim)
        assert out.size == self.size, f"{out.shape} {self.shape}"
        return Tensor.make(out._tensor._storage, self.shape, backend=self.backend)

    def zeros(self, shape: Optional[UserShape] = None) -> Tensor:
        """Return a new tensor filled with zeros."""

        def zero(shape: UserShape) -> Tensor:
            return Tensor.make(
                [0.0] * int(operators.prod(shape)), shape, backend=self.backend
            )

        if shape is None:
            out = zero(self.shape)
        else:
            out = zero(shape)
        out._type_(self.backend)
        return out

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        """Return the tensor's data as a tuple (storage, shape, strides)."""
        return self._tensor.tuple()

    def detach(self) -> Tensor:
        """Return a new tensor detached from the computation graph."""
        return Tensor(self._tensor, backend=self.backend)

    # Variable elements for backprop

    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate derivative (for leaf nodes in backprop)."""
        assert self.is_leaf(), "Only leaf variables can have derivatives."
        if self.grad is None:
            self.grad = Tensor.make(
                [0.0] * int(operators.prod(self.shape)),
                self.shape,
                backend=self.backend,
            )
        self.grad += x

    def is_leaf(self) -> bool:
        """Check if the tensor is a leaf node in the computation graph."""
        return self.history is not None and self.history.last_fn is None

    def is_constant(self) -> bool:
        """Check if the tensor is a constant (no history)."""
        return self.history is None

    @property
    def parents(self) -> Iterable[Variable]:
        """Return the parents of the tensor in the computation graph."""
        assert self.history is not None
        return self.history.inputs

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to compute gradients."""
        h = self.history
        assert h is not None
        assert h.last_fn is not None
        assert h.ctx is not None

        x = h.last_fn._backward(h.ctx, d_output)
        assert len(x) == len(h.inputs), f"Bug in function {h.last_fn}"
        return [
            (inp, inp.expand(self._ensure_tensor(d_in)))
            for inp, d_in in zip(h.inputs, x)
        ]

    def backward(self, grad_output: Optional[Tensor] = None) -> None:
        """Compute the gradients of the tensor with respect to its inputs."""
        if grad_output is None:
            assert self.shape == (1,), "Must provide grad_output if non-scalar"
            grad_output = Tensor.make([1.0], (1,), backend=self.backend)
        backpropagate(self, grad_output)

    def __truediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self, Inv.apply(self._ensure_tensor(b)))

    def __rtruediv__(self, b: TensorLike) -> Tensor:
        return Mul.apply(self._ensure_tensor(b), Inv.apply(self))

    def __matmul__(self, b: Tensor) -> Tensor:
        """Not used until Module 3"""
        return MatMul.apply(self, b)

    @property
    def shape(self) -> UserShape:
        """Returns
        shape of the tensor

        """
        return self._tensor.shape

    @property
    def size(self) -> int:
        """Returns the total number of elements in the tensor."""
        return self._tensor.size

    @property
    def dims(self) -> int:
        """Returns the number of dimensions of the tensor."""
        return self._tensor.dims

    def __add__(self, other: TensorLike) -> Tensor:
        """Element-wise addition with broadcasting support."""
        other_tensor = self._ensure_tensor(other)
        return Add.apply(self, other_tensor)

    def __sub__(self, other: TensorLike) -> Tensor:
        """Element-wise subtraction with broadcasting support."""
        other_tensor = self._ensure_tensor(other)
        return Add.apply(self, -other_tensor)

    def __mul__(self, other: TensorLike) -> Tensor:
        """Element-wise multiplication with broadcasting support."""
        other_tensor = self._ensure_tensor(other)
        return Mul.apply(self, other_tensor)

    def __lt__(self, other: TensorLike) -> Tensor:
        """Element-wise less-than comparison."""
        other_tensor = self._ensure_tensor(other)
        return LT.apply(self, other_tensor)

    def __eq__(self, other: TensorLike) -> Tensor:
        """Element-wise equality comparison."""
        other_tensor = self._ensure_tensor(other)
        return EQ.apply(self, other_tensor)

    def __gt__(self, other: TensorLike) -> Tensor:
        """Element-wise greater-than comparison."""
        other_tensor = self._ensure_tensor(other)
        return LT.apply(other_tensor, self)

    def __neg__(self) -> Tensor:
        """Element-wise negation."""
        return Neg.apply(self)

    def __radd__(self, other: TensorLike) -> Tensor:
        """Right-hand side addition to support scalar + tensor."""
        return self + other

    def __rmul__(self, other: TensorLike) -> Tensor:
        """Right-hand side multiplication to support scalar * tensor."""
        return self * other

    def is_close(self, other: Tensor) -> Tensor:
        """Element-wise closeness comparison within a tolerance."""
        return IsClose.apply(self, other)

    def sigmoid(self) -> Tensor:
        """Applies the sigmoid function element-wise."""
        return Sigmoid.apply(self)

    def relu(self) -> Tensor:
        """Applies the ReLU function element-wise."""
        return ReLU.apply(self)

    def log(self) -> Tensor:
        """Computes the natural logarithm element-wise."""
        return Log.apply(self)

    def exp(self) -> Tensor:
        """Computes the exponential function element-wise."""
        return Exp.apply(self)

    def sum(self, dim: Optional[int] = None) -> Tensor:
        """Computes the sum over the specified dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, sums over all elements.

        Returns:
        -------
            Tensor: Summed tensor.

        """
        if dim is None:
            # Sum over all elements
            return Sum.apply(self.contiguous().view(self.size), self._ensure_tensor(0))
        else:
            # Sum over the specified dimension
            return Sum.apply(self, self._ensure_tensor(dim))

    def mean(self, dim: Optional[int] = None) -> Tensor:
        """Computes the mean over the specified dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, computes mean over all elements.

        Returns:
        -------
            Tensor: Tensor with mean values.

        """
        if dim is not None:
            return self.sum(dim) / self.shape[dim]
        else:
            return self.sum() / self.size

    def permute(self, *order: int) -> Tensor:
        """Permutes the dimensions of the tensor according to the specified order.

        Args:
        ----
            *order (int): The desired ordering of dimensions.

        Returns:
        -------
            Tensor: Permuted tensor.

        """
        order_tensor = tensor(list(order))
        return Permute.apply(self, order_tensor)

    def view(self, *shape: int) -> Tensor:
        """Reshapes the tensor to the specified shape.

        Args:
        ----
            *shape (int): The desired shape.

        Returns:
        -------
            Tensor: Reshaped tensor.

        """
        shape_tensor = tensor(list(shape))
        return View.apply(self, shape_tensor)

    def all(self, dim: Optional[int] = None) -> Tensor:
        """Checks if all elements are true (non-zero) over the specified dimension.

        Args:
        ----
            dim (int, optional): Dimension to reduce. If None, checks all elements.

        Returns:
        -------
            Tensor: Tensor with boolean values indicating if all elements are true.

        """
        if dim is None:
            # Reduce over all elements
            return All.apply(self.view(self.size), self._ensure_tensor(0))
        else:
            # Reduce over the specified dimension
            return All.apply(self, self._ensure_tensor(dim))

    def zero_grad_(self) -> None:
        """Sets the gradient of the tensor to None."""
        self.grad = None
