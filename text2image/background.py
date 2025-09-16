from pathlib import Path
from typing import Tuple

from PIL import Image

__all__ = ["BackgroundGenerator"]


class BackgroundGenerator:
    """Generate a variety of backgrounds for synthetic text rendering."""

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
