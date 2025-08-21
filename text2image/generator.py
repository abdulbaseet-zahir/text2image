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
    distortion_type: str | None = None,
    background_image_path: str | Path | None = None,
    blur_radius: float | None = None,
) -> Image.Image:
    """Apply post-processing (distortion, blur, blending with background)."""

    dg = DistorsionGenerator()
    bg_gen = BackgroundGenerator()

    # Distortion
    if distortion_type is not None:
        if distortion_type == "random":
            image = dg.random(image, vertical=False, horizontal=False)
        elif distortion_type == "sin":
            image = dg.sin(image, vertical=True)
        elif distortion_type == "cos":
            image = dg.cos(image, vertical=True)
        else:
            raise ValueError(
                f"Invalid distortion type: {distortion_type}. Must be 'random', 'sin', or 'cos'."
            )

        white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white_bg, image).convert("RGBA")

    # Blur
    if blur_radius is not None:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Background blend
    if background_image_path is not None:
        try:
            bg_img, _ = bg_gen.picture(
                image.size[1],
                image.size[0],
                background_image_path,
                fit_to_size=random.choice([True, False]),
            )
            bg_img = bg_img.convert("RGBA")
            image = Image.blend(image, bg_img, alpha=random.uniform(0.1, 0.7))
        except (FileNotFoundError, ValueError) as e:
            # Re-raise validation errors
            raise e
        except Exception:
            # fallback to white background on failure for other errors (PIL, etc.)
            pass

    return image.convert("RGB")


def text_to_image(
    text: str,
    *,
    font_path: Union[str, Path],
    font_size: int,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    background_image_path: str | Path | None = None,
    text_colors: str = "black",
    distortion_type: str | None = None,
    blur_radius: float | None = None,
    text_dir: str = "rtl",
    text_align: str = "center",
) -> Image.Image | None:
    """Render *text* (assumed RTL for Kurdish/Arabic) into an RGB image.

    Parameters
    ----------
    text: str
        The text to render.
    font_path: str | Path
        Path to a specific .ttf font file.
    font_size: int
        Font size to use. If None, uses default range of sizes.
    image_size: tuple[int, int]
        Size of canvas before cropping text bounding box.
    background_image_path: Optional[str | Path]
        If provided, blend resulting image with a background image. Must be a path to a single image file.
    text_colors: str
        Color of the text.
    distortion_type: str
        Type of distortion: "random", "sin", or "cos" (default: None).
    blur_radius: float | None
        Radius for Gaussian blur (default: None).
    text_dir: str
        Text direction: "rtl" (right-to-left) or "ltr" (left-to-right) (default: "rtl").
    text_align: str
        Text alignment: "left", "center", or "right" (default: "center").
    """

    # Handle font_path parameter - must be a valid font file
    font_path = Path(font_path)
    if not font_path.exists():
        raise FileNotFoundError(f"Font file '{font_path}' not found.")
    if font_path.suffix.lower() != ".ttf":
        raise ValueError(f"Font file '{font_path}' is not a .ttf file.")

    try:
        image = Image.new("RGB", image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(str(font_path), font_size)

        # Center text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = (image_size[0] - text_width) / 2
        text_y = (image_size[1] - text_height) / 2

        # Try to use text direction and alignment, fallback gracefully if not supported
        try:
            draw.text(
                (text_x, text_y),
                text,
                fill=text_colors,
                font=font,
                direction=text_dir,
                align=text_align,
            )
        except Exception:
            # Fallback without direction and alignment if not supported
            draw.text(
                (text_x, text_y),
                text,
                fill=text_colors,
                font=font,
            )

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

        return _transform(
            img_rgba,
            background_image_path=background_image_path,
            distortion_type=distortion_type,
            blur_radius=blur_radius,
        )
    except Exception as exc:
        print(exc)
        print(f"Skipped text '{text}'")
        return None


def equation_to_image(
    equation: str,
    *,
    dpi: int = 300,
    padding: int = 15,
    background_image_path: str | Path | None = None,
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

        return _transform(img_rgba, background_image_path=background_image_path)
    except Exception as exc:
        print(exc)
        print(f"Skipped equation '{equation}'")
        return None
