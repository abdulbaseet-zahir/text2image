import math
import random
from typing import Callable

import numpy as np
from PIL import Image

__all__ = ["DistorsionGenerator"]


class DistorsionGenerator:
    """Utility class providing simple geometric distortions using sine, cosine or random offsets."""

    @classmethod
    def _apply_func_distorsion(
        cls,
        image: Image.Image,
        vertical: bool,
        horizontal: bool,
        max_offset: int,
        func: Callable[[int], int],
    ) -> Image.Image:
        """Apply a 1-D offset function either vertically, horizontally or both.

        Parameters
        ----------
        image: PIL.Image
            Input RGBA image.
        vertical: bool
            Whether to apply distortion along the *x* axis (vertical displacement).
        horizontal: bool
            Whether to apply distortion along the *y* axis (horizontal displacement).
        max_offset: int
            Maximum pixel offset returned by *func*.
        func: Callable[[int], int]
            Function returning an offset for each pixel position.
        """

        if not vertical and not horizontal:
            return image

        # Ensure input is in RGBA to keep transparency when compositing later on.
        rgba_img = image.convert("RGBA")
        arr = np.array(rgba_img)

        # Pre-compute offsets for speed.
        vertical_offsets = [func(i) for i in range(arr.shape[1])]
        horizontal_offsets = [func(i) for i in range(arr.shape[0])]

        # The new canvas must account for extra space after distortion.
        new_height = arr.shape[0] + (2 * max_offset if vertical else 0)
        new_width = arr.shape[1] + (2 * max_offset if horizontal else 0)
        new_arr = np.zeros((new_height, new_width, 4), dtype=np.uint8)

        # Vertical displacement (column-wise copy)
        if vertical:
            for x_coord, v_off in enumerate(vertical_offsets):
                new_arr[
                    max_offset + v_off : max_offset + v_off + arr.shape[0],
                    x_coord + (0 if not horizontal else max_offset),
                ] = arr[:, x_coord]

        # Horizontal displacement (row-wise copy)
        if horizontal:
            # Work on a copy so that vertical + horizontal can be combined.
            base_arr = new_arr if vertical else arr
            for y_coord, h_off in enumerate(horizontal_offsets):
                new_arr[
                    y_coord + (0 if not vertical else max_offset),
                    max_offset + h_off : max_offset + h_off + arr.shape[1],
                ] = base_arr[y_coord if not vertical else max_offset + y_coord]

        return Image.fromarray(new_arr, mode="RGBA")

    # Public helpers
    @classmethod
    def sin(
        cls, image: Image.Image, *, vertical: bool = False, horizontal: bool = False
    ) -> Image.Image:
        ratio = random.random() * 0.5
        max_offset = int(image.height**ratio)
        return cls._apply_func_distorsion(
            image,
            vertical,
            horizontal,
            max_offset,
            lambda x: int(math.sin(math.radians(x)) * max_offset),
        )

    @classmethod
    def cos(
        cls, image: Image.Image, *, vertical: bool = False, horizontal: bool = False
    ) -> Image.Image:
        ratio = random.random() * 0.5
        max_offset = int(image.height**ratio)
        return cls._apply_func_distorsion(
            image,
            vertical,
            horizontal,
            max_offset,
            lambda x: int(math.cos(math.radians(x)) * max_offset),
        )

    @classmethod
    def random(
        cls, image: Image.Image, *, vertical: bool = False, horizontal: bool = False
    ) -> Image.Image:
        ratio = random.random() * 0.5
        max_offset = int(image.height**ratio)
        return cls._apply_func_distorsion(
            image,
            vertical,
            horizontal,
            max_offset,
            lambda _x: random.randint(0, max_offset),
        )
