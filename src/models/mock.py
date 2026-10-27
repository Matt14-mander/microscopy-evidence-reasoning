"""Deterministic infrastructure test double; never a scientific baseline."""

import json

from src.utils import digest, image_digest


class MockModel:
    def generate(self, request):
        value = int(digest([request.prompt, request.seed,
                            [image_digest(image) for image in request.images]])[:16], 16)
        index = value % request.num_choices
        return json.dumps({"answer": chr(65 + index), "confidence": 1 / request.num_choices,
                           "abstain": False})
