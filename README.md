# text2image

A Python package for generating synthetic images from text and LaTeX equations. Convert text in Kurdish, Arabic, English, and mathematical expressions into stunning visual representations with customizable fonts, backgrounds, and distortion effects.

## Features

- **Multi-language Support**: Generate images from Kurdish, Arabic, English text
- **LaTeX Equation Rendering**: Convert mathematical equations to images using matplotlib
- **Customizable Fonts**: Use your own TTF font files
- **Background Effects**: Apply various background styles including Gaussian noise, quasicrystal patterns, and image blending
- **Distortion Effects**: Add realistic distortion using sine, cosine, or random transformations
- **Post-processing**: Automatic blur effects and background blending for more realistic results

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
    fonts_dir="path/to/your/fonts",  # Directory containing .ttf files
    font_sizes=[24, 28, 32],  # Optional: specify font sizes as list
    image_size=(800, 400)  # Optional: specify canvas size
)

# Or specify individual font files
image = text_to_image(
    text="Hello World",
    fonts_dir=["fonts/arial.ttf", "fonts/times.ttf"],  # List of specific font files
    font_sizes=24
)

if image:
    image.save("output.png")
```

### LaTeX Equation to Image

```python
from text2image import equation_to_image

# Generate an image from a LaTeX equation
image = equation_to_image(
    equation="x^2 + y^2 = r^2",
    dpi=300,  # Optional: specify DPI
    padding=20  # Optional: specify padding around equation
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

### Advanced Usage with Background Effects

```python
from text2image import text_to_image

# Generate text with background pictures
image = text_to_image(
    text="مرحبا بالعالم",  # Arabic text
    fonts_dir="fonts/",
    background_pictures_dir="backgrounds/"  # Directory with background images
)
```

### Custom Font Sizes and Image Sizes

```python
from text2image import text_to_image

# Generate with specific font sizes and image dimensions
image = text_to_image(
    text="Kurdish Text: سڵاو",
    fonts_dir="fonts/",
    font_sizes=[16, 18, 20, 22],  # Only use these font sizes
    image_size=(1200, 600)  # Larger canvas
)

# Or use specific font files with different sizes
image = text_to_image(
    text="Custom fonts",
    fonts_dir=["fonts/arabic.ttf", "fonts/kurdish.ttf"],
    font_sizes=[20, 24, 28],  # Randomly choose from these sizes
    image_size=(800, 400)
)

# Or use a single font size
image = text_to_image(
    text="Fixed size text",
    fonts_dir="fonts/",
    font_sizes=24,  # Use only this font size
    image_size=(800, 400)
)
```

## API Reference

### `text_to_image(text, fonts_dir, **kwargs)`

Generate an image from text.

**Parameters:**
- `text` (str): The text to render
- `fonts_dir` (str|Path|list[str|Path]): Directory containing .ttf font files, or list of specific font file paths
- `font_sizes` (int|list[int], optional): Font size(s) to use. If int, uses that specific size. If list, randomly chooses from the sequence
- `image_size` (tuple[int, int], optional): Canvas size before cropping
- `background_pictures_dir` (str|Path, optional): Directory with background images

**Returns:**
- `PIL.Image.Image` or `None`: Generated image or None if generation failed

### `equation_to_image(equation, **kwargs)`

Generate an image from a LaTeX equation.

**Parameters:**
- `equation` (str): LaTeX equation string
- `dpi` (int, optional): DPI for rendering (default: 300)
- `padding` (int, optional): Padding around equation (default: 15)
- `background_pictures_dir` (str|Path, optional): Directory with background images

**Returns:**
- `PIL.Image.Image` or `None`: Generated image or None if generation failed

## Background Effects

The package includes several background generators:

- **Gaussian Noise**: Mimics paper texture
- **Plain White**: Clean white background
- **Quasicrystal**: Abstract geometric patterns
- **Picture Blending**: Blend with user-provided images

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
