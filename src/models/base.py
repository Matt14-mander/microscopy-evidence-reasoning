from dataclasses import dataclass
from typing import Protocol

from PIL import Image
from src.models.protocols import render_prompt


@dataclass(frozen=True)
class ModelRequest:
    """Intentionally excludes labels, annotations and privileged metadata."""

    sample_id: str
    prompt: str
    images: tuple[Image.Image, ...]
    num_choices: int
    seed: int


class Model(Protocol):
    def generate(self, request: ModelRequest) -> str:
        """Return the unmodified model response for parsing and audit."""
        ...


def build_prompt(question, choices, protocol=None):
    return render_prompt(question, choices, protocol)
