# Copyright (c) 2025, Tri Dao.

import math
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, const_expr
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

import quack.utils as utils


F32_or_F32x2 = Float32 | Tuple[Float32, Float32]


@dsl_user_op
def tanh(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "tanh.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def sigmoid(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        # return 0.5 + 0.5 * cute.math.tanh(0.5 * x, fastmath=True)
        return 0.5 + 0.5 * tanh(0.5 * x)
    else:
        x_half = utils.mul_packed_f32x2((0.5, 0.5), x)
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return utils.fma_packed_f32x2(tanh_x_half, (0.5, 0.5), (0.5, 0.5))


@dsl_user_op
def relu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0))
    else:
        return cute.arch.fmax(x[0], Float32(0.0)), cute.arch.fmax(x[1], Float32(0.0))


@cute.jit
@dsl_user_op
def drelu(x: Float32, dout: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    x_pos = cutlass.Boolean(x > 0)
    return dout if x_pos else Float32(0.0), cute.arch.fmax(x, Float32(0.0))


@dsl_user_op
def relu_sq(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * x
    else:
        relu_x = (cute.arch.fmax(x[0], Float32(0.0)), cute.arch.fmax(x[1], Float32(0.0)))
        return utils.mul_packed_f32x2(relu_x, x)


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
def gelu_tanh_approx(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            = 0.5 * x * (1 + tanh(x * (0.797885 + 0.0356774 * x * x)))
    """
    sqrt_2_over_pi = math.sqrt(2 / math.pi)  # ~0.797885
    sqrt_2_over_pi_coeff = 0.044715 * sqrt_2_over_pi  # ~0.0356774
    if const_expr(not isinstance(x, tuple)):
        return 0.5 * (
            x
            # Currently cute.math.tanh(x, fastmath=True) generates very slow code
            # * (1 + cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x)), fastmath=True))
            * (1.0 + tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * (x * x))))
        )
    else:
        x_sq = utils.mul_packed_f32x2(x, x)
        x_sq_scaled = utils.fma_packed_f32x2(
            x_sq, (sqrt_2_over_pi_coeff, sqrt_2_over_pi_coeff), (sqrt_2_over_pi, sqrt_2_over_pi)
        )
        z = utils.mul_packed_f32x2(x, x_sq_scaled)
        tanh_z = (tanh(z[0]), tanh(z[1]))
        x_tanh_z = utils.fma_packed_f32x2(tanh_z, x, x)
        return utils.mul_packed_f32x2((0.5, 0.5), x_tanh_z)


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
    # tanh_z = cute.math.tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq), fastmath=True)
    tanh_z = tanh(x * (sqrt_2_over_pi + sqrt_2_over_pi_coeff * x_sq))
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
def silu(x: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """
    silu(x) = x * sigmoid(x) = x * (1 + tanh(x / 2)) / 2 = (0.5 * x) * tanh(0.5 * x) + (0.5 * x)
    This compiles down to 3 SASS instructions: FMUL to get 0.5 * x, MUFU.TANH, and FFMA.
    """
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x
        # return x_half * cute.math.tanh(x_half, fastmath=True) + x_half
        return x_half * tanh(x_half) + x_half
    else:
        x_half = utils.mul_packed_f32x2((0.5, 0.5), x)
        tanh_x_half = (tanh(x_half[0]), tanh(x_half[1]))
        return utils.fma_packed_f32x2(x_half, tanh_x_half, x_half)


@dsl_user_op
def swiglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    if const_expr(not isinstance(x, tuple)):
        return silu(x) * y
    else:
        return utils.mul_packed_f32x2(silu(x), y)


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
def swiglu_oai(
    x: F32_or_F32x2, y: F32_or_F32x2, alpha: float = 1.702, *, loc=None, ip=None
) -> F32_or_F32x2:
    """The swiglu variant used in gpt-oss, which has a scaling factor on x and bias of 1 to y.
    https://github.com/openai/gpt-oss/blob/7be9334950053a888e24887a57dac797a17d6e00/gpt_oss/torch/model.py#L249
    x * sigmoid(alpha * x) * (y + 1)
    Compile down to FMUL, FMUL, TANH, FFMA, FFMA
    """
    # Compute sigmoid(alpha * x) using tanh: sigmoid(z) = 0.5 * (1 + tanh(z/2))
    if const_expr(not isinstance(x, tuple)):
        x_half = 0.5 * x
        # silu_x = x_half * cute.math.tanh(alpha * x_half, fastmath=True) + x_half
        silu_x = x_half * tanh(alpha * x_half) + x_half
        return silu_x * y + silu_x
    else:
        x_half = utils.mul_packed_f32x2((0.5, 0.5), x)
        alpha_x_half = utils.mul_packed_f32x2((alpha, alpha), x_half)
        tanh_alpha_x_half = (tanh(alpha_x_half[0]), tanh(alpha_x_half[1]))
        silu_x = utils.fma_packed_f32x2(x_half, tanh_alpha_x_half, x_half)
        return utils.fma_packed_f32x2(silu_x, y, silu_x)


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
    # sigmoid_alpha_x = 0.5 + 0.5 * cute.math.tanh(alpha_x_half, fastmath=True)
    sigmoid_alpha_x = 0.5 + 0.5 * tanh(alpha_x_half)
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
def glu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """GLU: Gated Linear Unit
    glu(x, y) = sigmoid(x) * y
    Using tanh to compute sigmoid: sigmoid(x) = 0.5 * (1 + tanh(x/2))
    """
    if const_expr(not isinstance(x, tuple)):
        sigmoid_x = sigmoid(x)  # FMUL, MUFU.TANH, then FFMA
        return sigmoid_x * y  # FMUL
    else:
        sigmoid_x = sigmoid(x)
        return utils.mul_packed_f32x2(sigmoid_x, y)


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
def reglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """ReGLU: ReLU Gated Linear Unit
    reglu(x, y) = relu(x) * y = max(x, 0) * y
    """
    if const_expr(not isinstance(x, tuple)):
        return cute.arch.fmax(x, Float32(0.0)) * y
    else:
        relu_x = relu(x)
        return utils.mul_packed_f32x2(relu_x, y)


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
def geglu(x: F32_or_F32x2, y: F32_or_F32x2, *, loc=None, ip=None) -> F32_or_F32x2:
    """GeGLU: GELU Gated Linear Unit
    geglu(x, y) = gelu(x) * y
    Uses the tanh approximation of GELU
    """
    if const_expr(not isinstance(x, tuple)):
        return gelu_tanh_approx(x) * y
    else:
        return utils.mul_packed_f32x2(gelu_tanh_approx(x), y)


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
