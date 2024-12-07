from typing import Tuple, TypeVar, Any

from numba import njit as _njit
from numba import prange

from .autodiff import Context
from .tensor import Tensor
from .tensor_data import (
    Shape,
    Strides,
    Storage,
    broadcast_index,
    index_to_position,
    to_index,
)
from .tensor_functions import Function

Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function using NUMBA, enforcing inline execution."""
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


# JIT compile fast versions of tensor_data functions.
# In case of errors, consult NUMBA documentation about allowed operations.
to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


def _tensor_conv1d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Perform a 1D convolution given an input tensor and a set of weights.

    Input shape: batch, in_channels, width
    Weight shape: out_channels, in_channels, k_width
    Output shape: batch, out_channels, width

    The `reverse` flag decides if the convolution kernel is anchored on the left (False)
    or on the right (True).
    """
    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    # Basic dimension checks
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )
    s1 = input_strides
    s2 = weight_strides
    s3 = out_strides

    for b in prange(batch):
        for oc in prange(out_channels):
            for ow in prange(out_width):
                acc = 0.0
                for ic in prange(in_channels):
                    for kw_ in prange(kw):
                        # Compute input index based on 'reverse' flag.
                        iw = ow - kw_ if reverse else ow + kw_
                        if 0 <= iw < width:
                            in_val = input[b * s1[0] + ic * s1[1] + iw * s1[2]]
                            w_val = weight[oc * s2[0] + ic * s2[1] + kw_ * s2[2]]
                            acc += in_val * w_val
                out[b * s3[0] + oc * s3[1] + ow * s3[2]] = acc


tensor_conv1d = njit(_tensor_conv1d, parallel=True)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 1D convolution.

        Args:
        ----
            ctx (Context): Autodiff context.
            input (Tensor): Input of shape [batch, in_channel, width].
            weight (Tensor): Weights of shape [out_channel, in_channel, k_width].

        Returns:
        -------
            Tensor: Output of shape [batch, out_channel, width].

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, w = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        # Initialize output tensor and perform convolution
        output = input.zeros((batch, out_channels, w))
        tensor_conv1d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backpropagation for 1D convolution.

        Computes gradients of the input and the weight given the upstream gradient.
        """
        input, weight = ctx.saved_values
        batch, in_channels, w = input.shape
        out_channels, in_channels, kw = weight.shape

        # Compute gradient w.r.t. weight
        grad_weight = grad_output.zeros((in_channels, out_channels, kw))
        new_input = input.permute(1, 0, 2)
        new_grad_output = grad_output.permute(1, 0, 2)
        tensor_conv1d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2)

        # Compute gradient w.r.t. input
        grad_input = input.zeros((batch, in_channels, w))
        new_weight = weight.permute(1, 0, 2)
        tensor_conv1d(
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv1d = Conv1dFun.apply


def _tensor_conv2d(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    input: Storage,
    input_shape: Shape,
    input_strides: Strides,
    weight: Storage,
    weight_shape: Shape,
    weight_strides: Strides,
    reverse: bool,
) -> None:
    """Perform a 2D convolution given an input tensor and a set of weights.

    Input shape: batch, in_channels, height, width
    Weight shape: out_channels, in_channels, k_height, k_width
    Output shape: batch, out_channels, height, width

    The `reverse` flag determines if the kernel is anchored top-left (False)
    or bottom-right (True).
    """
    batch_, out_channels, out_height, out_width = out_shape
    batch, in_channels, height, width = input_shape
    out_channels_, in_channels_, kh, kw = weight_shape

    # Check dimension agreement
    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = input_strides
    s2 = weight_strides
    s3 = out_strides

    # Unpack strides for clarity
    s10, s11, s12, s13 = s1[0], s1[1], s1[2], s1[3]
    s20, s21, s22, s23 = s2[0], s2[1], s2[2], s2[3]
    s30, s31, s32, s33 = s3[0], s3[1], s3[2], s3[3]

    for b in prange(batch):
        for oc in prange(out_channels):
            for oh in prange(out_height):
                for ow in prange(out_width):
                    # Accumulate over input channels and kernel spatial dimensions
                    acc = 0.0
                    for ic in prange(in_channels):
                        for kh_ in prange(kh):
                            for kw_ in prange(kw):
                                if reverse:
                                    ih = oh + kh_ - kh + 1
                                    iw = ow + kw_ - kw + 1
                                else:
                                    ih = oh + kh_
                                    iw = ow + kw_

                                if 0 <= ih < height and 0 <= iw < width:
                                    in_idx = b * s10 + ic * s11 + ih * s12 + iw * s13
                                    w_idx = oc * s20 + ic * s21 + kh_ * s22 + kw_ * s23
                                    acc += input[in_idx] * weight[w_idx]

                    out_idx = b * s30 + oc * s31 + oh * s32 + ow * s33
                    out[out_idx] = acc


tensor_conv2d = njit(_tensor_conv2d, parallel=True, fastmath=True)


class Conv2dFun(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """Compute a 2D convolution.

        Args:
        ----
            ctx (Context): Autodiff context.
            input (Tensor): Input tensor of shape [batch, in_channel, height, width].
            weight (Tensor): Weight tensor of shape [out_channel, in_channel, k_height, k_width].

        Returns:
        -------
            Tensor: Output tensor of shape [batch, out_channel, height, width].

        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, h, w = input.shape
        out_channels, in_channels2, kh, kw = weight.shape
        assert in_channels == in_channels2

        output = input.zeros((batch, out_channels, h, w))
        tensor_conv2d(
            *output.tuple(), output.size, *input.tuple(), *weight.tuple(), False
        )
        return output

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backpropagation for 2D convolution.

        Calculates gradients for both inputs and weights given the upstream gradient.
        """
        input, weight = ctx.saved_values
        batch, in_channels, h, w = input.shape
        out_channels, in_channels, kh, kw = weight.shape

        # Compute gradient w.r.t. weight
        grad_weight = grad_output.zeros((in_channels, out_channels, kh, kw))
        new_input = input.permute(1, 0, 2, 3)
        new_grad_output = grad_output.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_weight.tuple(),
            grad_weight.size,
            *new_input.tuple(),
            *new_grad_output.tuple(),
            False,  # type: ignore
        )
        grad_weight = grad_weight.permute(1, 0, 2, 3)

        # Compute gradient w.r.t. input
        grad_input = input.zeros((batch, in_channels, h, w))
        new_weight = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            *grad_input.tuple(),
            grad_input.size,  # type: ignore
            *grad_output.tuple(),
            *new_weight.tuple(),
            True,  # type: ignore
        )
        return grad_input, grad_weight


conv2d = Conv2dFun.apply
