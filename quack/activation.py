# Copyright (c) 2025, Tri Dao.

import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def sigmoid(x: Float32, *, loc=None, ip=None) -> Float32:
    return 0.5 + 0.5 * cute.math.tanh(0.5 * x, fastmath=True)


@dsl_user_op
def relu(x: Float32, *, loc=None, ip=None) -> Float32:
    return cute.arch.fmax(x, Float32(0.0))


@cute.jit
@dsl_user_op
def drelu(x: Float32, dout: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    x_pos = cutlass.Boolean(x > 0)
    return dout if x_pos else Float32(0.0), cute.arch.fmax(x, Float32(0.0))


@dsl_user_op
def relu_sq(x: Float32, *, loc=None, ip=None) -> Float32:
    return cute.arch.fmax(x, Float32(0.0)) * x


@cute.jit
@dsl_user_op
def drelu_sq(x: Float32, dout: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """
    ReLU squared backward pass: computes gradient w.r.t. x and recomputes forward
    Given: relu_sq_out = max(x, 0) * x, and dout = grad w.r.t. relu_sq_out
    Returns: (dx, relu_sq_out) where:
    - dx = dout * 2 * x if x > 0, else 0
    - relu_sq_out = max(x, 0) * x
    """
    x_pos = cutlass.Boolean(x > 0)
    relu_sq_out = cute.arch.fmax(x, Float32(0.0)) * x
    # Derivative: d/dx[max(x,0) * x] = 2*x if x > 0, else 0
    dx = (2.0 * dout * x) if x_pos else Float32(0.0)
    return dx, relu_sq_out


@dsl_user_op
def gelu_tanh_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            = 0.5 * x * (1 + tanh(x * (0.797885 + 0.0356774 * x * x)))
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # ~0.0356774
    return 0.5 * (
        x
        * (1 + cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x)), fastmath=True))
    )


@dsl_user_op
def dgelu_tanh_approx(x: Float32, dout: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """
    GELU tanh approximation backward pass: computes gradient w.r.t. x and recomputes forward
    Given: gelu_out = 0.5 * x * (1 + tanh(x * (c1 + c2 * x^2))), and dout = grad w.r.t. gelu_out
    Returns: (dx, gelu_out)

    Derivative uses the chain rule:
    d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
    where z = x * (c1 + c2 * x^2), dz/dx = c1 + 3 * c2 * x^2
    and sech^2(z) = 1 - tanh^2(z)
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # c1 ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # c2 ~0.0356774
    sqrt_2_over_pi_coeff_3 = 3.0 * sqrt_2_over_pi_coeff  # c3 ~0.01070322

    # Compute z = x * (c1 + c2 * x^2)
    x_sq = x * x
    tanh_z = cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq), fastmath=True)
    half_tanh_z_plus_one = 0.5 + 0.5 * tanh_z
    gelu_out = x * half_tanh_z_plus_one

    # Compute gradient
    # sech^2(z) = 1 - tanh^2(z)
    sech2_z = 1 - tanh_z * tanh_z
    # dz/dx = c1 + 3 * c2 * x^2
    dz_dx = sqrt_2_over_pi + sqrt_2_over_pi_coeff_3 * x_sq
    # d/dx[gelu(x)] = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
    dgelu = half_tanh_z_plus_one + x * (0.5 * (sech2_z * dz_dx))

    dx = dout * dgelu
    return dx, gelu_out


@dsl_user_op
def silu(x: Float32, *, loc=None, ip=None) -> Float32:
    """
    silu(x) = x * sigmoid(x) = x * (1 + tanh(x / 2)) / 2 = (0.5 * x) * tanh(0.5 * x) + (0.5 * x)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * x, MUFU.TANH, and FFMA.
    """
    x_half = 0.5 * x
    return x_half * cute.math.tanh(x_half, fastmath=True) + x_half


@dsl_user_op
def swiglu(x: Float32, y: Float32, *, loc=None, ip=None) -> Float32:
    return silu(x) * y


@dsl_user_op
def dswiglu(
    x: Float32, y: Float32, dout: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32]:
    """
    SwiGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: swiglu_out = silu(x) * y, and dout = grad w.r.t. swiglu_out
    Returns: (dx, dy, swiglu_out) where dx = dout * y * d_silu(x), dy = dout * silu(x)

    d_silu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))

    This has been optimized to use fewer instructions (i.e. we expand things out
    to use FFMA instead of FADD and FMUL).
    """
    # Compute sigmoid(x) using tanh: sigmoid(x) = 0.5 * (1 + tanh(0.5 * x))
    # FMUL, MUFU.TANH, then FFMA
    sigmoid_x = sigmoid(x)
    silu_x = x * sigmoid_x  # FMUL
    silu_x_dout = silu_x * dout  # FMUL
    #   d_silu(x) * dout
    # = sigmoid_x * (1 + x * (1 - sigmoid_x)) * dout
    # = (sigmoid_x + sigmoid_x * x * (1 - sigmoid_x)) * dout
    # = (sigmoid_x + silu_x * (1 - sigmoid_x)) * dout
    # = (sigmoid_x + silu_x - silu_x * sigmoid_x) * dout
    # = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x * dout
    d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x_dout  # FFMA, FFMA
    dx = d_silu_x_dout * y  # FMUL
    dy = silu_x_dout
    swiglu_out = silu_x * y  # FMUL
    # Overall it's 1 MUFU.TANH, 5 FMUL, 3 FFMA
    return dx, dy, swiglu_out


@dsl_user_op
def swiglu_oai(x: Float32, y: Float32, alpha: float = 1.702, *, loc=None, ip=None) -> Float32:
    """The swiglu variant used in gpt-oss, which has a scaling factor on x and bias of 1 to y.
    https://github.com/openai/gpt-oss/blob/7be9334950053a888e24887a57dac797a17d6e00/gpt_oss/torch/model.py#L249
    x * sigmoid(alpha * x) * (y + 1)
    Compile down to FMUL, FMUL, TANH, FFMA, FFMA
    """
    # Compute sigmoid(alpha * x) using tanh: sigmoid(z) = 0.5 * (1 + tanh(z/2))
    x_half = 0.5 * x
    silu_x = x_half * cute.math.tanh(alpha * x_half, fastmath=True) + x_half
    return silu_x * y + silu_x


@dsl_user_op
def dswiglu_oai(
    x: Float32, y: Float32, dout: Float32, alpha: float = 1.702, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32]:
    """
    Swiglu OAI backward pass: computes gradients w.r.t. x and y
    Given: swiglu_oai_out = x * sigmoid(alpha * x) * (y + 1), and dout = grad w.r.t. swiglu_oai_out
    Returns: (dx, dy, swiglu_oai_out)

    Derivative of x * sigmoid(alpha * x) w.r.t. x:
    d/dx[x * sigmoid(alpha * x)] = sigmoid(alpha * x) + alpha * x * sigmoid(alpha * x) * (1 - sigmoid(alpha * x))
    """
    # Compute sigmoid(alpha * x) using tanh: sigmoid(z) = 0.5 * (1 + tanh(z/2))
    alpha_x_half = (0.5 * alpha) * x  # FMUL
    # MUFU.TANH, then FFMA
    sigmoid_alpha_x = 0.5 + 0.5 * cute.math.tanh(alpha_x_half, fastmath=True)
    silu_x = x * sigmoid_alpha_x  # FMUL
    silu_x_dout = silu_x * dout  # FMUL
    # FFMA, FFMA, FMUL
    d_silu_x_dout = (sigmoid_alpha_x + alpha * (silu_x - silu_x * sigmoid_alpha_x)) * dout
    dx = d_silu_x_dout * y + d_silu_x_dout  # FFMA, instead of multiply by y + 1
    dy = silu_x_dout
    swiglu_out = silu_x * y + silu_x  # FFMA, instead of multiply by y + 1
    # Overall it's 1 MUFU.TANH, 4 FMUL, 5 FFMA
    return dx, dy, swiglu_out


@dsl_user_op
def glu(x: Float32, y: Float32, *, loc=None, ip=None) -> Float32:
    """GLU: Gated Linear Unit
    glu(x, y) = sigmoid(x) * y
    Using tanh to compute sigmoid: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    """
    sigmoid_x = sigmoid(x)  # FMUL, MUFU.TANH, then FFMA
    return sigmoid_x * y  # FMUL


@dsl_user_op
def dglu(
    x: Float32, y: Float32, dout: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32]:
    """
    GLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: glu_out = sigmoid(x) * y, and dout = grad w.r.t. glu_out
    Returns: (dx, dy, glu_out) where:
    - dx = dout * y * sigmoid(x) * (1 - sigmoid(x))
    - dy = dout * sigmoid(x)
    - glu_out = sigmoid(x) * y
    """
    # Compute sigmoid(x) using tanh: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    sigmoid_x = sigmoid(x)  # FMUL, MUFU.TANH, then FFMA
    sigmoid_x_dout = sigmoid_x * dout  # FMUL
    glu_out = sigmoid_x * y  # FMUL
    # dx = y * sigmoid(x) * (1 - sigmoid(x)) * dout
    #    = y * (1 - sigmoid(x)) * sigmoid_x_dout
    #    = (y - y * sigmoid(x)) * sigmoid_x_dout
    #    = (y - glu_out) * sigmoid_x_dout
    dx = (y - glu_out) * sigmoid_x_dout  # FADD, FMUL
    dy = sigmoid_x_dout
    # Total: 1 MUFU.TANH, 4 FMUL, 1 FADD, 1 FFMA
    return dx, dy, glu_out


@dsl_user_op
def reglu(x: Float32, y: Float32, *, loc=None, ip=None) -> Float32:
    """ReGLU: ReLU Gated Linear Unit
    reglu(x, y) = relu(x) * y = max(x, 0) * y
    """
    return cute.arch.fmax(x, Float32(0.0)) * y


@cute.jit
@dsl_user_op
def dreglu(
    x: Float32, y: Float32, dout: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32]:
    """
    ReGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: reglu_out = relu(x) * y, and dout = grad w.r.t. reglu_out
    Returns: (dx, dy, reglu_out) where:
    - dx = dout * y if x > 0, else 0
    - dy = dout * relu(x)
    - reglu_out = relu(x) * y
    """
    x_pos = cutlass.Boolean(x > 0)
    relu_x = cute.arch.fmax(x, Float32(0.0))
    dx = (dout * y) if x_pos else Float32(0.0)
    dy = dout * relu_x
    reglu_out = relu_x * y
    return dx, dy, reglu_out


@dsl_user_op
def geglu(x: Float32, y: Float32, *, loc=None, ip=None) -> Float32:
    """GeGLU: GELU Gated Linear Unit
    geglu(x, y) = gelu(x) * y
    Uses the tanh approximation of GELU
    """
    return gelu_tanh_approx(x) * y


@dsl_user_op
def dgeglu(
    x: Float32, y: Float32, dout: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32]:
    """
    GeGLU backward pass: computes gradients w.r.t. x (gate) and y (up projection)
    Given: geglu_out = gelu(x) * y, and dout = grad w.r.t. geglu_out
    Returns: (dx, dy, geglu_out) where:
    - dx = dout * y * d_gelu(x)
    - dy = dout * gelu(x)
    - geglu_out = gelu(x) * y
    """
    # Reuse dgelu_tanh_approx to compute d_gelu(x) * dout and gelu(x)
    dgelu_x_dout, gelu_x = dgelu_tanh_approx(x, dout)
    # Compute gradients for geglu
    dx = dgelu_x_dout * y
    dy = gelu_x * dout
    geglu_out = gelu_x * y
    return dx, dy, geglu_out
