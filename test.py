from text2image import text_to_random_image

from PIL import features

if not features.check("raqm"):
    print("raqm is not installed")
    exit()

text = """
ژمارەی ئەوانەی لە ئەنجامی بوومەلەرزەکەی شەوی رابردووی
"""


# Example: Using a specific background image file with random faded areas (rectangles & ellipses) to mimic bad printer
image, text = text_to_random_image(
    text=text,
    font_dir="fonts",
    background_dir="backgrounds",
)

if image is not None:
    image.save("output.png")
    print("Image saved as output.png")
    print(text)
else:
    print("No image generated")
