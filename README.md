# text2image

A Python package for generating synthetic images from text and LaTeX equations. Convert text in Kurdish, Arabic, English, and mathematical expressions into stunning visual representations with customizable fonts, backgrounds, and distortion effects.

## Features

- **Multi-language Support**: Generate images from Kurdish, Arabic, English text
- **LaTeX Equation Rendering**: Convert mathematical equations to images using matplotlib
- **Customizable Fonts**: Use your own TTF font files
- **Background Effects**: Apply various background styles including Gaussian noise, quasicrystal patterns, and image blending
- **Distortion Effects**: Add realistic distortion using sine, cosine, or random transformations
- **Post-processing**: Automatic blur effects and background blending for more realistic results
- **Text Control**: Control text direction (RTL/LTR) and alignment (left/center/right)

## Installation

### From GitHub

```bash
pip install git+https://github.com/abdulbaseet-zahir/text2image.git
```

## Quick Start

### Basic Text to Image

```python
from text2image import text_to_image
from PIL import Image

# Generate an image from text
image = text_to_image(
    text="Hello World",
    font_path="fonts/arial.ttf",  # Path to specific .ttf font file
    font_size=24,  # Font size as integer
    image_size=(800, 400)  # Optional: specify canvas size
)

if image:
    image.save("output.png")
```

### Advanced Text to Image with All Parameters

```python
from text2image import text_to_image

# Generate text with all customization options
image = text_to_image(
    text="سڵاو، باشی بەڕێز!!",  # Kurdish text
    font_path="fonts/kurdish.ttf",
    font_size=28,
    image_size=(1200, 600),
    background_image_path="backgrounds/paper_texture.png",  # Single background image file
    text_colors="red",  # Text color
    distortion_type="sin",  # "random", "sin", or "cos"
    blur_radius=0.3,
    text_dir="rtl",  # "rtl" (right-to-left) or "ltr" (left-to-right)
    text_align="center"  # "left", "center", or "right"
)
```

### LaTeX Equation to Image

```python
from text2image import equation_to_image

# Generate an image from a LaTeX equation
image = equation_to_image(
    equation="x^2 + y^2 = r^2",
    dpi=300,  # Optional: specify DPI
    padding=20,  # Optional: specify padding around equation
    background_image_path="backgrounds/paper_texture.png"  # Optional: background image file
)

if image:
    image.save("equation.png")
```

## Requirements

- Python >= 3.8
- Pillow >= 9.0.0
- NumPy >= 1.20
- OpenCV-Python >= 4.6
- Matplotlib >= 3.5

## Usage Examples

### Text Direction and Alignment

```python
from text2image import text_to_image

# Right-to-left text (Arabic/Kurdish) with right alignment
image = text_to_image(
    text="سڵاو، باشی بەڕێز!!",
    font_path="fonts/kurdish.ttf",
    font_size=22,
    text_dir="rtl",
    text_align="right"
)

# Left-to-right text with left alignment
image = text_to_image(
    text="Hello World",
    font_path="fonts/arial.ttf",
    font_size=24,
    text_dir="ltr",
    text_align="left"
)
```

### Distortion Effects

```python
from text2image import text_to_image

# Sine wave distortion
image = text_to_image(
    text="Distorted Text",
    font_path="fonts/times.ttf",
    font_size=32,
    distortion_type="sin"
)

# Cosine wave distortion
image = text_to_image(
    text="Wave Text",
    font_path="fonts/arial.ttf",
    font_size=28,
    distortion_type="cos"
)

# Random distortion
image = text_to_image(
    text="Random Text",
    font_path="fonts/calibri.ttf",
    font_size=26,
    distortion_type="random"
)
```

### Blur Effects

```python
from text2image import text_to_image

# Heavy blur
image = text_to_image(
    text="Blurry Text",
    font_path="fonts/verdana.ttf",
    font_size=30,
    blur_radius=0.8
)

# No blur
image = text_to_image(
    text="Sharp Text",
    font_path="fonts/georgia.ttf",
    font_size=26,
    blur=False
)
```

### Background Effects

```python
from text2image import text_to_image

# With background image
image = text_to_image(
    text="Text with Background",
    font_path="fonts/arial.ttf",
    font_size=24,
    background_image_path="backgrounds/paper_texture.png"  # Specific image file
)

# Without background
image = text_to_image(
    text="Text without Background",
    font_path="fonts/arial.ttf",
    font_size=24,
)
```

## API Reference

### `text_to_image(text, font_path, font_size, **kwargs)`

Generate an image from text.

**Parameters:**
- `text` (str): The text to render
- `font_path` (str|Path): Path to a specific .ttf font file
- `font_size` (int): Font size to use
- `image_size` (tuple[int, int], optional): Canvas size before cropping (default: (1000, 500))
- `background_image_path` (str|Path, optional): Path to a single background image file
- `text_colors` (str, optional): Color of the text (default: "black")
- `distortion_type` (str, optional): Type of distortion: "random", "sin", or "cos" (default: None)
- `blur_radius` (float, optional): Radius for Gaussian blur (default: 0.3)
- `text_dir` (str, optional): Text direction: "rtl" or "ltr" (default: "rtl")
- `text_align` (str, optional): Text alignment: "left", "center", or "right" (default: "center")

**Returns:**
- `PIL.Image.Image` or `None`: Generated image or None if generation failed

### `equation_to_image(equation, **kwargs)`

Generate an image from a LaTeX equation.

**Parameters:**
- `equation` (str): LaTeX equation string
- `dpi` (int, optional): DPI for rendering (default: 300)
- `padding` (int, optional): Padding around equation (default: 15)
- `background_image_path` (str|Path, optional): Path to a single background image file

**Returns:**
- `PIL.Image.Image` or `None`: Generated image or None if generation failed

## Background Effects

The package includes several background generators:

- **Gaussian Noise**: Mimics paper texture
- **Plain White**: Clean white background
- **Quasicrystal**: Abstract geometric patterns
- **Picture Blending**: Blend with user-provided images

#### Using Background Images

You can use a single background image file:

```python
from text2image import text_to_image

# Use a single background image
image = text_to_image(
    text="Hello World",
    font_path="fonts/arial.ttf",
    font_size=24,
    background_image_path="backgrounds/paper_texture.png"  # Single image file
)
```

## Distortion Effects

Apply realistic distortions to make images more natural:

- **Sine Distortion**: Smooth wave-like distortion
- **Cosine Distortion**: Alternative wave pattern
- **Random Distortion**: Random pixel offsets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Abdulbasit Zahir
