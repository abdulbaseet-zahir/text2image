from text2image import text_to_image

# Example: Using a specific background image file
image, text = text_to_image(
    text="١٩٩٨/٣/٤",
    font_path="fonts/rudawregular2.ttf",  # Specific font file
    font_size=24,
    image_size=(800, 400),
    background_image_path="backgrounds/398a4342ff1c4dff9d4bac23f295d340.png",  # Specific image file
    text_colors="black",
    distortion_type="sin",
    blur_radius=0.7,
    text_dir="ltr",
    text_align="center",
    random_padding={"top": 10, "bottom": 10, "left": 10, "right": 10},
)

if image is not None:
    image.save("output.png")
    print("Image saved as output.png")
    print(text)
else:
    print("No image generated")
