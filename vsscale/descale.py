from __future__ import annotations

from functools import partial
from math import log2
from typing import Callable, Iterable, List, Type

import vapoursynth as vs
from vsaa import Znedi3
from vskernels import Catrom, Kernel, Spline144, get_kernel, get_prop
from vskernels.kernels.abstract import Scaler
from vsmask.edge import EdgeDetect
from vsutil import depth, get_depth, join, split

from .mask import descale_detail_mask
from .scale import scale_var_clip
from .types import CreditMaskT, DescaleAttempt

core = vs.core


__all__ = [
    'get_select_descale',
    'descale',
]


def get_select_descale(
    clip: vs.VideoNode, descale_attempts: list[DescaleAttempt], threshold: float = 0.0
) -> tuple[Callable[[list[vs.VideoFrame], int], vs.VideoNode], list[vs.VideoNode]]:
    clips_by_height = {
        attempt.resolution.height: attempt
        for attempt in descale_attempts
    }

    diff_clips = [
        attempt.diff for attempt in clips_by_height.values()
    ]
    n_clips = len(diff_clips)

    def _get_descale_score(mapped_props: list[tuple[int, float]], i: int) -> float:
        height_log = log2(clip.height - mapped_props[i][0])
        pstats_avg = round(1 / max(mapped_props[i][1], 1e-12))

        return height_log * pstats_avg ** 0.2  # type: ignore

    def _parse_attemps(f: list[vs.VideoFrame]) -> tuple[vs.VideoNode, list[tuple[int, float]], int]:
        mapped_props = [
            (get_prop(frame, "descale_height", int), get_prop(frame, "PlaneStatsAverage", float)) for frame in f
        ]

        best_res = max(range(n_clips), key=partial(_get_descale_score, mapped_props))

        best_attempt = clips_by_height[mapped_props[best_res][0]]

        return best_attempt.descaled, mapped_props, best_res

    if threshold == 0:
        def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
            return _parse_attemps(f)[0]
    else:
        def _select_descale(f: list[vs.VideoFrame], n: int) -> vs.VideoNode:
            best_attempt, mapped_props, best_res = _parse_attemps(f)

            if mapped_props[best_res][1] > threshold:
                return clip

            return best_attempt

    return _select_descale, diff_clips


def descale(
    clip: vs.VideoNode,
    width: int | Iterable[int] | None = None,
    height: int | Iterable[int] = 720,
    upscaler: Scaler | None = Znedi3(),
    kernels: Kernel | Type[Kernel] | str | list[Kernel | Type[Kernel] | str] = Catrom(),
    thr: float = 0.0, shift: tuple[float, float] = (0, 0),
    mask: CreditMaskT | bool = descale_detail_mask,
    show_mask: bool = False
) -> vs.VideoNode:
    assert clip.format

    if not isinstance(kernels, List):
        kernels = [kernels]

    norm_kernels = [
        get_kernel(kernel)() if isinstance(kernel, str) else (
            kernel if isinstance(kernel, Kernel) else kernel()
        ) for kernel in kernels
    ]

    if isinstance(height, int):
        heights = [height]
    else:
        heights = list(height)

    if width is None:
        widths = [round(h * clip.width / clip.height) for h in heights]
    elif isinstance(width, int):
        widths = [width]
    else:
        widths = list(width)

    if len(widths) != len(heights):
        raise ValueError("descale: Number of heights and widths specified mismatch!")

    if not norm_kernels:
        raise ValueError("descale: You must specify at least one kernel!")

    multi_descale = len(widths) > 1

    work_clip, *chroma = split(clip)

    clip_y = work_clip.resize.Point(format=vs.GRAYS)

    n_kernels = len(norm_kernels)

    kernel_combinations = list(zip(norm_kernels, list(zip(widths, heights)) * n_kernels))

    descale_attempts = [
        DescaleAttempt.from_args(
            clip_y, width, height, shift, kernel,
            descale_attempt_idx=i,
            descale_height=height,
            descale_kernel=kernel.__class__.__name__
        )
        for i, (kernel, (width, height)) in enumerate(kernel_combinations)
    ]

    if multi_descale:
        var_res_clip = core.std.Splice([
            clip_y.std.BlankClip(length=len(clip_y) - 1, keep=True),
            clip_y.std.BlankClip(length=1, width=clip_y.width + 1, keep=True)
        ], mismatch=True)

        select_partial, prop_clips = get_select_descale(clip_y, descale_attempts, thr)

        descaled = var_res_clip.std.FrameEval(select_partial, prop_clips)
    else:
        descaled = descale_attempts[0].descaled

    if upscaler is None:
        upscaled = descaled
    else:
        upscaled = scale_var_clip(descaled, clip_y.width, clip_y.height, scaler=upscaler)

    if mask:
        if len(kernel_combinations) == 1:
            rescaled = descale_attempts[0].rescaled
        else:
            rescaled = clip_y.std.FrameEval(
                lambda f, n: descale_attempts[f.props.descale_attempt_idx].rescaled, descaled
            )

        if mask is True:
            mask = descale_detail_mask

        if isinstance(mask, EdgeDetect):
            mask = mask.edgemask(clip_y)
        elif callable(mask):
            mask = mask(clip_y, rescaled)

        if upscaler is None:
            mask = Spline144().scale(mask, upscaled.width, upscaled.height)
            clip_y = Spline144().scale(clip_y, upscaled.width, upscaled.height)

        if show_mask:
            return mask

        upscaled = upscaled.std.MaskedMerge(clip_y, mask)

    upscaled = depth(upscaled, get_depth(clip))

    if not chroma:
        return upscaled

    return join([upscaled, *chroma], clip.format.color_family)