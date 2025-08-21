from text2image import text_to_image

# Example: Using a specific background image file
image = text_to_image(
    text="سڵاو، باشی بەڕێز!!",
    font_path="fonts/rudawregular2.ttf",  # Specific font file
    font_size=24,
    image_size=(800, 400),
    background_image_path="backgrounds/398a4342ff1c4dff9d4bac23f295d340.png",  # Specific image file
    text_colors="red",
    distortion_type="sin",
    blur_radius=0.3,
    text_dir="rtl",
    text_align="right",
)

if image is not None:
    image.save("output.png")
    print("Image saved as output.png")
else:
    print("No image generated")
