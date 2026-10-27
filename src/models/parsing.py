"""Strict parsing: format errors are distinct from deliberate abstention."""

import json
import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    answer_index: int | None
    confidence: float | None
    abstain: bool
    valid: bool
    error: str | None = None


def parse_prediction(raw, num_choices):
    try:
        text = raw.strip()
        if text.startswith("```"):
            match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
            if not match:
                raise ValueError("Malformed JSON fence")
            text = match.group(1)
        obj = json.loads(text)
        if not isinstance(obj, dict) or not {"answer", "confidence", "abstain"} <= obj.keys():
            raise ValueError("Expected JSON object with answer, confidence and abstain")
        if type(obj["abstain"]) is not bool:
            raise ValueError("abstain must be boolean")
        confidence = obj["confidence"]
        if confidence is not None:
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be null or a finite number in [0,1]")
            confidence = float(confidence)
        answer = obj["answer"]
        if obj["abstain"]:
            if answer is not None:
                raise ValueError("Abstention requires a null answer")
            return Prediction(None, confidence, True, True)
        if not isinstance(answer, str) or len(answer.strip()) != 1:
            raise ValueError("answer must be a single option letter")
        index = ord(answer.strip().upper()) - 65
        if not 0 <= index < num_choices:
            raise ValueError("Option letter outside the choice range")
        return Prediction(index, confidence, False, True)
    except (ValueError, TypeError, AttributeError) as error:
        return Prediction(None, None, False, False, str(error))
