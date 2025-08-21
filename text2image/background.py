import math
import os
import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

__all__ = ["BackgroundGenerator"]


class BackgroundGenerator:
    """Generate a variety of simple random backgrounds for synthetic text rendering."""

    @classmethod
    def gaussian_noise(cls, height: int, width: int) -> Image.Image:
        """Return a plain background with Gaussian noise to mimic paper texture."""

        image = np.ones((height, width), dtype=np.float32) * 255
        cv2.randn(image, 235, 10)
        return Image.fromarray(image.astype(np.uint8)).convert("RGB")

    @classmethod
    def plain_white(cls, height: int, width: int) -> Image.Image:
        """Return a pure white background."""
        return Image.new("RGB", (width, height), (255, 255, 255))

    @classmethod
    def quasicrystal(cls, height: int, width: int) -> Image.Image:
        """Create an abstract quasi-crystal background (slow!)."""
        image = Image.new("L", (width, height))
        pixels = image.load()

        frequency = random.random() * 30 + 20
        phase = random.random() * 2 * math.pi
        rotations = random.randint(10, 20)

        for x in range(width):
            y_coord = float(x) / (width - 1) * 4 * math.pi - 2 * math.pi
            for y in range(height):
                x_coord = float(y) / (height - 1) * 4 * math.pi - 2 * math.pi
                z_val = 0.0
                for i in range(rotations):
                    r = math.hypot(x_coord, y_coord)
                    a = math.atan2(y_coord, x_coord) + i * math.pi * 2.0 / rotations
                    z_val += math.cos(r * math.sin(a) * frequency + phase)
                color = int(255 - round(255 * z_val / rotations))
                pixels[x, y] = color
        return image.convert("RGB")

    @classmethod
    def picture(
        cls,
        height: int,
        width: int,
        background_path: str | Path,
        fit_to_size: bool = True,
    ) -> Tuple[Image.Image, str]:
        """Load and process a specific background image for synthetic text rendering."""

        background_path = Path(background_path)
        if not background_path.is_file():
            raise FileNotFoundError(
                f"Background image '{background_path}' does not exist or is not a file."
            )

        picture = Image.open(background_path).convert("RGB")
        if fit_to_size:
            picture = picture.resize((width, height), Image.Resampling.LANCZOS)
        else:
            # Resize or thumbnail to make sure picture is at least target size.
            if picture.width < width:
                new_height = int(picture.height * (width / picture.width))
                picture = picture.resize((width, new_height), Image.Resampling.LANCZOS)
            if picture.height < height:
                new_width = int(picture.width * (height / picture.height))
                picture.thumbnail((new_width, height), Image.Resampling.LANCZOS)

        # Simple crop from top-left corner (could randomize in the future).
        x = 0
        y = 0
        cropped = picture.crop((x, y, x + width, y + height))
        return cropped, str(background_path)
