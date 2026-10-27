from dataclasses import dataclass
from typing import Protocol

from PIL import Image


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


def build_prompt(question, choices):
    options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    return (
        "Answer the microscopy multiple-choice question using the provided evidence.\n"
        "Return only a JSON object with keys answer, confidence, abstain.\n"
        'answer: one option letter; confidence: a number in [0,1] or null.\n'
        'If evidence is insufficient, return {"answer": null, "confidence": null, "abstain": true}.\n'
        'Otherwise set abstain to false. Do not include explanations.\n\n'
        f"Question: {question}\n{options}"
    )
