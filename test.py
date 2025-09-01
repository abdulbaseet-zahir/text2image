from text2image import text_to_image

text = """
ژمارەی ئەوانەی لە ئەنجامی بوومەلەرزەکەی شەوی رابردووی ئەفغانستان گیانیان لەدەستدا گەیشتووەتە لانیکەم 812 کەس و زیاتر لە دوو هەزار و 800ی دیکەش بریندارن. 
 
بوومەلەرزەکە بە گوڕی 6 پلە بوو و زۆرترین زیانی بە پارێزگەی کونەر لە باکووری رۆژهەڵاتی ئەفغانستان گەیاند و ژمارەی گیانلەدەستدان و برینداران لە بەرزبوونەوەی بەردەوامدایە. 
 
بوومەلەرزەکە کاژێر 11:47ـی شەوی یەکشەممە روویدا، لە کاتێکدا بەشێکی خەڵک نووستبوون. چەقی بوومەلەرزەکە لە نزیک جەلال ئابادی پایتەختی پارێزگای نەنگەرهار و بە قووڵایی 14 کیلۆمەتر بوو.
 
لە گوندە دوورەدەستەکانی پارێزگەی کونەری باکووری رۆژهەڵات، کە هاوشێوەی زۆربەی ئەفغانستان خانووەکانی بە خۆڵ و بەرد پێکەوەنراون، بوومەلەرزەکە وێرانکاریی هەرە گەورەی ناوەتەوە. 
"""


# text = "لاتین ئەوانە"

# Example: Using a specific background image file
image, text = text_to_image(
    text=text,
    font_path="fonts/rudawregular2.ttf",  # Specific font file
    font_size=24,
    # image_size=(800, 400),
    auto_size=True,
    background_image_path="backgrounds/398a4342ff1c4dff9d4bac23f295d340.png",  # Specific image file
    text_colors="black",
    distortion_type="sin",
    blur_radius=0.7,
    text_dir="rtl",
    text_align="right",
    random_padding=True,
)

if image is not None:
    image.save("output.png")
    print("Image saved as output.png")
    print(text)
else:
    print("No image generated")
