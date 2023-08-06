from __future__ import annotations

from vskernels import Catrom, Kernel, KernelT
from vstools import (
    FieldBased, FieldBasedT, FuncExceptT, core, get_w, get_y, vs
)
__all__ = [
    'descale_fields'
]


def descale_fields(
    clip: vs.VideoNode, width: int | None = None, height: int = 720,
    tff: bool | FieldBasedT = True, kernel: KernelT = Catrom,
    src_top: float | tuple[float, float] = 0.0,
    src_left: float | tuple[float, float] = 0.0,
    debug: bool = False, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Descale interwoven upscaled fields, also known as a cross conversion.

    ``src_top``, ``src_left`` allow you to to shift the clip prior to descaling.
    This may be useful, as sometimes clips are shifted before or after the original upscaling.

    :param clip:        Clip to process.
    :param width:       Native width. Will be automatically determined if set to `None`.
    :param height:      Native height. Will be divided by two internally.
    :param tff:         Top-field-first. `False` sets it to Bottom-Field-First.
    :param kernel:      py:class:`vskernels.Kernel` used for the descaling.
    :param src_top:     Shifts the clip vertically during the descaling.
                        Can be a tuple, defining the shift per-field.
    :param src_left:    Shifts the clip horizontally during the descaling.
                        Can be a tuple, defining the shift per-field.
    :param debug:       Set a frameprop with the kernel that was used.

    :return:            Descaled GRAY clip.
    """

    func = func or descale_fields

    height_field = int(height / 2)
    width = width or get_w(height, clip)

    kernel = Kernel.ensure_obj(kernel, func)

    clip = FieldBased.ensure_presence(clip, tff, func)

    y = get_y(clip).std.SeparateFields()

    if isinstance(src_top, tuple):
        ff_top, sf_top = src_top
    else:
        ff_top = sf_top = src_top

    if isinstance(src_left, tuple):
        ff_left, sf_left = src_left
    else:
        ff_left = sf_left = src_left

    if (ff_top, ff_left) == (sf_top, sf_left):
        descaled = kernel.descale(y, width, height_field, (ff_top, ff_left))
    else:
        descaled = core.std.Interleave([
            kernel.descale(y[::2], width, height_field, (ff_top, ff_left)),
            kernel.descale(y[1::2], width, height_field, (sf_top, sf_left))
        ])

    weave_y = descaled.std.DoubleWeave()

    if debug:
        weave_y = weave_y.std.SetFrameProp('scaler', data=f'{kernel.__class__.__name__} (Fields)')

    return weave_y.std.SetFieldBased(0)[::2]
