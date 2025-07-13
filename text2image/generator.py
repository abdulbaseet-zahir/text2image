from __future__ import annotations

import io
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Union, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib.pyplot as plt

from .distortion import DistorsionGenerator
from .background import BackgroundGenerator

__all__ = ["text_to_image", "equation_to_image"]

# Default parameters
IMAGE_SIZE: Tuple[int, int] = (1000, 500)


def _transform(
    image: Image.Image,
    *,
    distortion: bool = True,
    background: bool = True,
    background_pictures_dir: str | Path | None = None,
) -> Image.Image:
    """Apply random post-processing (distortion, blur, blending with background)."""

    dg = DistorsionGenerator()
    bg_gen = BackgroundGenerator()

    # Distortion
    if distortion and random.choice([True, False]):
        dist_type = random.choice(["random", "sin", "cos"])
        if dist_type == "random":
            image = dg.random(image, vertical=False, horizontal=False)
        elif dist_type == "sin":
            image = dg.sin(image, vertical=True)
        else:
            image = dg.cos(image, vertical=True)

        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white_bg, image).convert("RGBA")

    # Blur
    if random.choice([True, False]):
        image = image.filter(
            ImageFilter.GaussianBlur(
                radius=random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            )
        )

    # Background blend
    if background and background_pictures_dir is not None:
        try:
            bg_img, _ = bg_gen.picture(
                image.size[1], image.size[0], background_pictures_dir
            )
            bg_img = bg_img.convert("RGBA")
            image = Image.blend(image, bg_img, alpha=random.uniform(0.1, 0.7))
        except Exception:
            # fallback to white background on failure
            pass

    return image.convert("RGB")


def text_to_image(
    text: str,
    *,
    fonts_dir: Union[str, Path, List[Union[str, Path]]],
    font_sizes: Union[int, List[int]] | None = None,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    background_pictures_dir: str | Path | None = None,
) -> Image.Image | None:
    """Render *text* (assumed RTL for Kurdish/Arabic) into an RGB image.

    Parameters
    ----------
    text: str
        The text to render.
    fonts_dir: str | Path | list[str | Path]
        Directory containing `.ttf` font files, or list of specific font file paths.
    font_sizes: int | list[int] | None
        Font size(s) to use. If int, uses that specific size.
        If list, randomly chooses from the sequence.
        If None, uses default range of sizes.
    image_size: tuple[int, int]
        Size of canvas before cropping text bounding box.
    background_pictures_dir: Optional[str | Path]
        If provided, blend resulting image with a random picture from this directory.
    """

    # Handle fonts_dir parameter - can be directory path or list of font files
    if isinstance(fonts_dir, (str, Path)):
        # Single directory path
        fonts_dir = Path(fonts_dir)
        font_files = list(fonts_dir.glob("*.ttf"))
        if not font_files:
            raise FileNotFoundError(f"No .ttf fonts found in directory '{fonts_dir}'.")
    elif isinstance(fonts_dir, list):
        # List of font file paths
        font_files = []
        for font_path in fonts_dir:
            font_path = Path(font_path)
            if not font_path.exists():
                print(f"Warning: Font file '{font_path}' not found, skipping.")
                continue
            if font_path.suffix.lower() != '.ttf':
                print(f"Warning: Font file '{font_path}' is not a .ttf file, skipping.")
                continue
            font_files.append(font_path)
        
        if not font_files:
            raise FileNotFoundError("No valid .ttf font files found in the provided list.")
    else:
        raise TypeError("fonts_dir must be a string/Path (directory) or list of font file paths.")

    # Handle font_sizes parameter
    if font_sizes is None:
        font_sizes = [
            10,
            12,
            14,
            16,
            18,
            20,
            22,
            24,
            26,
            28,
            30,
            32,
            34,
            36,
            38,
            40,
            42,
        ]
    elif isinstance(font_sizes, int):
        font_sizes = [font_sizes]

    try:
        image = Image.new("RGB", image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        font_path = random.choice(font_files)
        font_size = random.choice(font_sizes)
        font = ImageFont.truetype(str(font_path), font_size)

        # Center text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (image_size[0] - text_width) / 2
        text_y = (image_size[1] - text_height) / 2
        draw.text((text_x, text_y), text, fill="black", font=font, direction="rtl")

        # Crop to bounding box with padding
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = 255 * (gray < 128).astype(np.uint8)
        coords = cv2.findNonZero(gray)
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)

        # Random padding
        w += random.randint(15, 30)
        h += random.randint(15, 30)
        x = max(0, x - random.randint(7, 15))
        y = max(0, y - random.randint(7, 15))
        cropped = np.array(image)[y : y + h, x : x + w]
        img_rgba = Image.fromarray(cropped).convert("RGBA")

        return _transform(img_rgba, background_pictures_dir=background_pictures_dir)
    except Exception as exc:
        print(exc)
        print(f"Skipped text '{text}'")
        return None


def equation_to_image(
    equation: str,
    *,
    dpi: int = 300,
    padding: int = 15,
    background_pictures_dir: str | Path | None = None,
) -> Image.Image | None:
    """Render a LaTeX/MathText *equation* into an RGB image.

    By default it uses *matplotlib*'s internal mathtext engine so that no LaTeX
    installation is required.
    """

    try:
        fig = plt.figure(
            figsize=(0.01, 0.01)
        )  # size will be adjusted by bbox_inches="tight"
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.5, f"${equation}$", fontsize=24, ha="center", va="center")
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf).convert("RGBA")

        # Add padding around non-transparent pixels
        arr = np.array(img)
        alpha = arr[:, :, 3]
        coords = cv2.findNonZero((alpha > 0).astype(np.uint8))
        if coords is None:
            return None
        x, y, w, h = cv2.boundingRect(coords)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(arr.shape[1] - x, w + 2 * padding)
        h = min(arr.shape[0] - y, h + 2 * padding)
        cropped = arr[y : y + h, x : x + w]
        img_rgba = Image.fromarray(cropped).convert("RGBA")

        return _transform(img_rgba, background_pictures_dir=background_pictures_dir)
    except Exception as exc:
        print(exc)
        print(f"Skipped equation '{equation}'")
        return None
