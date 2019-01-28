from typing import Optional, List

import matplotlib.colors
import numpy as np
import pytest

from corrscope.channel import ChannelConfig
from corrscope.layout import LayoutConfig
from corrscope.outputs import RGB_DEPTH
from corrscope.renderer import RendererConfig, MatplotlibRenderer

WIDTH = 640
HEIGHT = 360

ALL_ZEROS = np.array([0, 0])
sloped_ys = np.array([-1, 1])

all_colors = pytest.mark.parametrize(
    "bg_str,fg_str,grid_str,antialiasing",
    [
        ("#000000", "#ffffff", None, True),
        ("#ffffff", "#000000", None, True),
        ("#0000aa", "#aaaa00", None, True),
        ("#aaaa00", "#0000aa", None, True),
        # Enabling gridlines enables Axes rectangles.
        # Make sure they don't draw *over* the global figure background.
        ("#0000aa", "#aaaa00", "#ff00ff", True),  # beautiful magenta gridlines
        ("#aaaa00", "#0000aa", "#ff00ff", True),
        # Disabling antialiasing ensures no other colors are present.
        ("#000000", "#ffffff", None, False),
        ("#0000aa", "#aaaa00", "#ff00ff", False),  # beautiful magenta gridlines
    ],
)

nplots = 2


@all_colors
def test_default_colors(bg_str, fg_str, grid_str, antialiasing):
    """ Test the default background/foreground colors. """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color=fg_str,
        grid_color=grid_str,
        antialiasing=antialiasing,
    )
    lcfg = LayoutConfig()

    r = MatplotlibRenderer(cfg, lcfg, nplots, None)
    verify(r, bg_str, fg_str, grid_str, antialiasing)

    # Ensure default ChannelConfig(line_color=None) does not override line color
    chan = ChannelConfig(wav_path="")
    channels = [chan] * nplots
    r = MatplotlibRenderer(cfg, lcfg, nplots, channels)
    verify(r, bg_str, fg_str, grid_str, antialiasing)


@all_colors
def test_line_colors(bg_str, fg_str, grid_str, antialiasing):
    """ Test channel-specific line color overrides """
    cfg = RendererConfig(
        WIDTH,
        HEIGHT,
        bg_color=bg_str,
        init_line_color="#888888",
        grid_color=grid_str,
        antialiasing=antialiasing,
    )
    lcfg = LayoutConfig()

    chan = ChannelConfig(wav_path="", line_color=fg_str)
    channels = [chan] * nplots
    r = MatplotlibRenderer(cfg, lcfg, nplots, channels)
    verify(r, bg_str, fg_str, grid_str, antialiasing)


VERIFY_AA = False
# too slow (1 second per test case)
# mplcairo grids are antialiased anyway.


def verify(
    r: MatplotlibRenderer, bg_str, fg_str, grid_str: Optional[str], antialiasing: bool
):
    r.render_frame([sloped_ys] * nplots)
    frame_colors: np.ndarray = np.frombuffer(r.get_frame(), dtype=np.uint8).reshape(
        (-1, RGB_DEPTH)
    )

    bg_u8 = to_bgra(bg_str)
    fg_u8 = to_bgra(fg_str)
    base_colors = [bg_u8, fg_u8]

    if grid_str:
        grid_u8 = to_bgra(grid_str)
        base_colors.append(grid_u8)

    # Ensure background is correct
    bg_frame = frame_colors[0]
    assert (
        bg_frame == bg_u8
    ).all(), f"incorrect background, it might be grid_str={grid_str}"

    # Ensure foreground is present
    assert np.prod(
        frame_colors == fg_u8, axis=-1
    ).any(), "incorrect foreground, it might be 136 = #888888"

    # Ensure grid color is present
    if grid_str:
        assert np.prod(frame_colors == grid_u8, axis=-1).any(), "Missing grid_str"

    # Ensure colors are right
    assert (np.amax(frame_colors, axis=0) == np.amax(base_colors, axis=0)).all()
    assert (np.amin(frame_colors, axis=0) == np.amin(base_colors, axis=0)).all()

    # Ensure blended colors present iff antialiasing enabled.
    if VERIFY_AA:  # probably False
        colors_found = np.unique(frame_colors, axis=0)
        for base_color in base_colors:
            assert base_color in colors_found

        blending_found = len(colors_found) > len(base_colors)
        assert blending_found == antialiasing, [colors_found, base_colors]


def to_bgra(c) -> List[int]:
    # https://github.com/anntzer/mplcairo/issues/12#issuecomment-451035973
    # Cairo outputs bgrA.
    # ffmpeg bgr0 and rgb32 are equivalent, but rgb32 uses less CPU in ffplay.
    rgb = [int(round(c * 255)) for c in matplotlib.colors.to_rgb(c)]
    bgra = rgb[::-1] + [255]
    return bgra
