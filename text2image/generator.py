from __future__ import annotations

import io
import pathlib
import random
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
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
            image = dg.random(
                image,
                vertical=False,
                horizontal=False,
            )
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
            image = ImageChops.multiply(image, bg_img)
        except (FileNotFoundError, ValueError) as e:
            # Re-raise validation errors
            raise e
        except Exception:
            # fallback to white background on failure for other errors (PIL, etc.)
            pass

    return image.convert("RGB")


def _calculate_text_dimensions(
    text: str,
    font_path: Union[str, Path],
    font_size: int,
    text_dir: str,
    text_align: str,
) -> Tuple[int, int]:
    """Calculate the dimensions needed to render text with given font and settings.

    Returns
    -------
    Tuple[int, int]
        (width, height) needed to render the text
    """
    # Create a temporary image to measure text
    temp_image = Image.new("RGB", (1, 1), color=(255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_image)
    font = ImageFont.truetype(str(font_path), font_size)

    # Get text bounding box
    text_bbox = temp_draw.textbbox(
        (0, 0),
        text,
        font=font,
        direction=text_dir,
        align=text_align,
    )
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    return int(text_width), int(text_height)


def _calculate_optimal_canvas_size(
    text_width: int,
    text_height: int,
    padding: int = 50,
    min_size: Tuple[int, int] = (200, 100),
    max_size: Tuple[int, int] = (2000, 1000),
) -> Tuple[int, int]:
    """Calculate optimal canvas size based on text dimensions.

    Parameters
    ----------
    text_width, text_height : int
        Dimensions of the text
    padding : int
        Padding to add around the text
    min_size : Tuple[int, int]
        Minimum canvas size (width, height)
    max_size : Tuple[int, int]
        Maximum canvas size (width, height)

    Returns
    -------
    Tuple[int, int]
        Optimal canvas size (width, height)
    """
    # Calculate required size with padding
    required_width = text_width + (2 * padding)
    required_height = text_height + (2 * padding)

    # Apply min/max constraints
    canvas_width = max(min_size[0], min(required_width, max_size[0]))
    canvas_height = max(min_size[1], min(required_height, max_size[1]))

    return canvas_width, canvas_height


def text_to_image(
    text: str,
    *,
    font_path: Union[str, Path],
    font_size: int,
    image_size: Tuple[int, int] | None = None,
    auto_size: bool = False,
    padding: int = 50,
    min_canvas_size: Tuple[int, int] = (200, 100),
    max_canvas_size: Tuple[int, int] = (5000, 3000),
    background_image_path: str | Path | None = None,
    text_colors: str = "black",
    distortion_type: str | None = None,
    blur_radius: float | None = None,
    text_dir: str = "rtl",
    text_align: str = "right",
    random_padding: bool | dict[str, int | tuple[int, int] | list[int]] = True,
    random_faded_areas: bool = False,
    num_faded_areas: int = 2,
    max_area_size: float = 0.4,
    faded_opacity: float = 0.3,
) -> tuple[Image.Image, str]:
    """Render *text* (assumed RTL for Kurdish/Arabic) into an RGB image.

    Parameters
    ----------
    text: str
        The text to render.
    font_path: str | Path
        Path to a specific .ttf font file.
    font_size: int
        Font size to use.
    image_size: tuple[int, int] | None
        Size of canvas before cropping text bounding box. If None and auto_size=False, uses default IMAGE_SIZE.
    auto_size: bool
        If True, automatically calculate canvas size based on text dimensions to prevent text overflow.
        If False, use image_size parameter (default: False).
    padding: int
        Padding around text when auto_size=True (default: 50).
    min_canvas_size: tuple[int, int]
        Minimum canvas size when auto_size=True (default: (200, 100)).
    max_canvas_size: tuple[int, int]
        Maximum canvas size when auto_size=True (default: (2000, 1000)).
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
        Text alignment: "left", "center", or "right" (default: "right").
    random_padding: bool | dict
        If True, adds a small random padding on all sides. If False, no padding.
        If a dict, you can control padding per side with pixel values or ranges:
        keys: "left", "right", "top", "bottom", "horizontal", "vertical", "all".
        - Each value can be an int (exact pixels) or a (min, max) tuple/list for a random range.
        - "all" applies to every side unless overridden by a more specific key.
        - "horizontal" applies to left and right when they are not explicitly set.
        - "vertical" applies to top and bottom when they are not explicitly set.
    random_faded_areas: bool
        If True, automatically generates random faded areas (rectangles or ellipses) to mimic bad printer effect.
        If False, no areas are faded (default: False).
    num_faded_areas: int
        Number of random faded areas to generate when random_faded_areas=True (default: 2).
    max_area_size: float
        Maximum size of each random faded area as fraction of text dimensions (0.0-1.0).
        Default: 0.4 (40% of text width/height).
    faded_opacity: float
        Opacity level for faded areas, between 0.0 (completely transparent) and 1.0 (fully opaque).
        Default: 0.3 (30% opacity).
    """

    # Handle font_path parameter - must be a valid font file
    font_path = Path(font_path)
    if not font_path.exists():
        raise FileNotFoundError(f"Font file '{font_path}' not found.")
    if font_path.suffix.lower() != ".ttf":
        raise ValueError(f"Font file '{font_path}' is not a .ttf file.")

    try:
        # Determine canvas size
        if auto_size:
            # Calculate text dimensions first
            text_width, text_height = _calculate_text_dimensions(
                text, font_path, font_size, text_dir, text_align
            )
            # Calculate optimal canvas size
            image_size = _calculate_optimal_canvas_size(
                text_width, text_height, padding, min_canvas_size, max_canvas_size
            )

        elif image_size is None:
            # Use default size if neither auto_size nor image_size is specified
            image_size = IMAGE_SIZE

        image = Image.new("RGB", image_size, color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        font = ImageFont.truetype(str(font_path), font_size)

        # Handle random faded areas functionality
        if random_faded_areas:
            # Create RGBA image for opacity control
            image_rgba = Image.new("RGBA", image_size, color=(255, 255, 255, 255))
            draw_rgba = ImageDraw.Draw(image_rgba)

            # Get text dimensions for positioning
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Center text
            text_x = (image_size[0] - text_width) / 2
            text_y = (image_size[1] - text_height) / 2

            # Draw the full text first with normal opacity
            draw_rgba.text(
                (text_x, text_y),
                text,
                fill=text_colors if text_colors != "transparent" else "black",
                font=font,
                direction=text_dir,
                align=text_align,
            )

            # Create a mask for faded areas
            fade_mask = Image.new("L", image_size, 255)  # Start with full opacity
            fade_draw = ImageDraw.Draw(fade_mask)

            # Generate random faded areas
            for _ in range(num_faded_areas):
                # Generate random area dimensions
                area_width = random.uniform(0.1, max_area_size) * text_width
                area_height = random.uniform(0.1, max_area_size) * text_height

                # Generate random position within text bounds
                max_x = text_width - area_width
                max_y = text_height - area_height

                # Ensure area stays within text bounds
                if max_x > 0 and max_y > 0:
                    area_x = text_x + random.uniform(0, max_x)
                    area_y = text_y + random.uniform(0, max_y)

                    # Randomly choose between rectangle and ellipse (rounded) shape
                    shape_type = random.choice(["rectangle", "ellipse"])

                    if shape_type == "rectangle":
                        # Draw rectangular faded area
                        fade_draw.rectangle(
                            [area_x, area_y, area_x + area_width, area_y + area_height],
                            fill=int(255 * faded_opacity),
                        )
                    else:
                        # Draw elliptical (rounded) faded area
                        fade_draw.ellipse(
                            [area_x, area_y, area_x + area_width, area_y + area_height],
                            fill=int(255 * faded_opacity),
                        )

            # Apply the mask to create the faded effect
            image_rgba.putalpha(fade_mask)

            # Convert RGBA back to RGB
            image = Image.new("RGB", image_size, color=(255, 255, 255))
            image.paste(
                image_rgba, mask=image_rgba.split()[-1]
            )  # Use alpha channel as mask

        else:
            # Original text rendering for non-faded text
            # Get text dimensions for positioning
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Center text
            text_x = (image_size[0] - text_width) / 2
            text_y = (image_size[1] - text_height) / 2

            draw.text(
                (text_x, text_y),
                text,
                fill=text_colors if text_colors != "transparent" else "black",
                font=font,
                direction=text_dir,
                align=text_align,
            )

        # Crop to bounding box with padding
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        gray = 255 * (gray < 128).astype(np.uint8)
        coords = cv2.findNonZero(gray)
        if coords is None:
            return None, None
        x, y, w, h = cv2.boundingRect(coords)

        if text_colors == "transparent":
            image = Image.new("RGB", image_size, color=(255, 255, 255))

        # Padding logic (bool for legacy behavior, dict for per-side control)
        pad_left = pad_right = pad_top = pad_bottom = 0

        if isinstance(random_padding, bool):
            if random_padding:
                pad_left = random.randint(3, 10)
                pad_right = random.randint(3, 10)
                pad_top = random.randint(3, 10)
                pad_bottom = random.randint(3, 10)
        elif isinstance(random_padding, dict):

            def _parse_pad(value):
                if isinstance(value, tuple) and len(value) == 2:
                    a, b = value
                    return random.randint(min(a, b), max(a, b))
                if isinstance(value, list) and len(value) == 2:
                    a, b = value
                    return random.randint(min(a, b), max(a, b))
                if isinstance(value, int):
                    return value
                return 0

            all_pad = random_padding.get("all", None)
            horiz_pad = random_padding.get("horizontal", None)
            vert_pad = random_padding.get("vertical", None)

            if all_pad is not None:
                pad_left = pad_right = pad_top = pad_bottom = _parse_pad(all_pad)

            if horiz_pad is not None:
                if pad_left == 0:
                    pad_left = _parse_pad(horiz_pad)
                if pad_right == 0:
                    pad_right = _parse_pad(horiz_pad)

            if vert_pad is not None:
                if pad_top == 0:
                    pad_top = _parse_pad(vert_pad)
                if pad_bottom == 0:
                    pad_bottom = _parse_pad(vert_pad)

            if "left" in random_padding:
                pad_left = _parse_pad(random_padding["left"])
            if "right" in random_padding:
                pad_right = _parse_pad(random_padding["right"])
            if "top" in random_padding:
                pad_top = _parse_pad(random_padding["top"])
            if "bottom" in random_padding:
                pad_bottom = _parse_pad(random_padding["bottom"])

        # Apply padding and clamp to image bounds
        x = max(0, x - pad_left)
        y = max(0, y - pad_top)
        w = min(image_size[0] - x, w + pad_left + pad_right)
        h = min(image_size[1] - y, h + pad_top + pad_bottom)

        cropped = np.array(image)[y : y + h, x : x + w]
        img_rgba = Image.fromarray(cropped).convert("RGBA")

        return _transform(
            img_rgba,
            background_image_path=background_image_path,
            distortion_type=distortion_type,
            blur_radius=blur_radius,
        ), (text if text_colors != "transparent" else "NOTEXT")
    except Exception as exc:
        print(exc)
        print(f"Skipped text '{text}'")
        return None, None


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


def text_to_random_image(
    text: str,
    font_dir: str,
    background_dir: str | None = None,
) -> Image.Image | None:

    fonts = list(pathlib.Path(font_dir).glob("*.ttf"))
    backgrounds = (
        list(pathlib.Path(background_dir).glob("*.png")) + [None] * 10
        if background_dir is not None
        else [None] * 10
    )

    text_colors = [
        "red",
        "blue",
        "green",
        "black",
        "orange",
        "crimson",
        "purple",
        "brown",
        "magenta",
        "teal",
        "indigo",
        "maroon",
        "navy",
        "olive",
        "darkblue",
        "transparent",
    ] + ["black"] * 50

    background_image_path = (
        random.choice(backgrounds) if background_dir is not None else None
    )
    if background_image_path is not None:
        background_image_path = background_image_path.as_posix()

    font_path = random.choice(fonts)
    if font_path is not None:
        font_path = font_path.as_posix()

    return text_to_image(
        text,
        font_path=font_path,
        font_size=random.randint(16, 26),
        auto_size=True,
        background_image_path=background_image_path,
        text_colors=random.choice(text_colors),
        distortion_type=random.choice(["random", "sin", "cos"] + [None] * 10),
        blur_radius=random.uniform(0.1, 0.8) if random.choice([True, False]) else None,
        text_dir=random.choice(["rtl", "ltr"]),
        text_align=random.choice(["left", "center", "right"]),
        random_padding=random.choice([True, False]),
        random_faded_areas=random.choice([True, False]),
        num_faded_areas=random.randint(1, 10),
        max_area_size=random.uniform(0.1, 0.5),
        faded_opacity=random.uniform(0.1, 0.8),
    )
