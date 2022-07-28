
from typing import Tuple

import vapoursynth as vs
from vsexprtools import expr_func
from vsexprtools.util import aka_expr_available
from vskernels import MatrixCoefficients, Transfer
from vsutil import get_depth

__all__ = [
    'gamma2linear', 'linear2gamma'
]


def _linear_diff(cont: float, thr: float, diff: str = '') -> str:
    inv_op = f'{diff} -' if diff else ''

    return f'1 1 {cont} {thr} {inv_op} * exp + /'


def _sigmoid_x(sigmoid: bool, cont: float, thr: float) -> Tuple[str, str, str]:
    if not sigmoid:
        return '', '', ''

    header, x0, x1 = '', _linear_diff(cont, thr), _linear_diff(cont, thr, '1')

    if aka_expr_available:
        header = f'{x0} SX0! {x1} SX1!'
        x0, x1 = 'SX0@', 'SX1@'

    return header, x0, x1


def _clamp_converted(clip: vs.VideoNode, header: str, expr: str, curve: Transfer) -> vs.VideoNode:
    clamping = '0.0 1.0 clamp' if aka_expr_available else'0.0 max 1.0 min'

    linear = expr_func(clip, f'{header} {expr} {clamping}')

    return linear.std.SetFrameProps(_Transfer=curve.value)


def gamma2linear(
    clip: vs.VideoNode, curve: Transfer, gcor: float = 1.0,
    sigmoid: bool = False, thr: float = 0.5, cont: float = 6.5,
    epsilon: float = 1e-6
) -> vs.VideoNode:
    """Convert gamma to linear curve."""
    assert clip.format

    if get_depth(clip) != 32 and clip.format.sample_type != vs.FLOAT:
        raise ValueError('gamma2linear: Your clip must be 32bit float!')

    c = MatrixCoefficients.from_curve(curve)

    header, x0, x1 = _sigmoid_x(sigmoid, cont, thr)

    expr = f'x {c.k0} <= x {c.phi} / x {c.alpha} + 1 {c.alpha} + / {c.gamma} pow ? {gcor} pow'

    if sigmoid:
        expr = f'{thr} 1 {expr} {x1} {x0} - * {x0} + {epsilon} max / 1 - {epsilon} max log {cont} / -'

    return _clamp_converted(clip, header, expr, Transfer.LINEAR)


def linear2gamma(
    clip: vs.VideoNode, curve: Transfer, gcor: float = 1.0,
    sigmoid: bool = False, thr: float = 0.5, cont: float = 6.5
) -> vs.VideoNode:
    """Convert linear curve to gamma."""
    assert clip.format

    if get_depth(clip) != 32 and clip.format.sample_type != vs.FLOAT:
        raise ValueError('linear2gamma: Your clip must be 32bit float!')

    c = MatrixCoefficients.from_curve(curve)

    header, x0, x1 = _sigmoid_x(sigmoid, cont, thr)

    if sigmoid:
        lin = f"{_linear_diff(cont, thr, 'x')} {x0} - {x1} {x0} - / {gcor} pow"
    else:
        lin = f'x {gcor} pow'

    if aka_expr_available:
        header = f'{header} {lin} LIN!'
        lin = 'LIN@'

    expr = f'{lin} {c.k0} {c.phi} / <= {lin} {c.phi} * {lin} 1 {c.gamma} / pow {c.alpha} 1 + * {c.alpha} - ?'

    return _clamp_converted(clip, header, expr, curve)
