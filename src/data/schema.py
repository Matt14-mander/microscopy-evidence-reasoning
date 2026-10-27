from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class Region:
    """Pixel coordinates [x0, y0, x1, y1), attached to one image."""

    image_index: int
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    question: str
    choices: tuple[str, ...]
    answer_index: int
    image_paths: tuple[Path, ...]
    regions: tuple[Region, ...] = ()
    metadata: dict = field(default_factory=dict)

    def load_images(self):
        images = []
        for path in self.image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        return images
