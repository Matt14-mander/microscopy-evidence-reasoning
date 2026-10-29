"""Strict parsing: format errors are distinct from deliberate abstention."""

import json
import math
import re
from dataclasses import dataclass
from src.models.protocols import resolve_protocol


@dataclass(frozen=True)
class Prediction:
    answer_index: int | None
    confidence: float | None
    abstain: bool
    valid: bool
    error: str | None = None
    error_kind: str | None = None


class ProtocolViolation(ValueError):
    pass


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def parse_prediction(raw, num_choices, protocol=None):
    spec = resolve_protocol(protocol)  # Configuration errors must fail, not become model errors.
    try:
        text = raw.strip()
        if text.startswith("```"):
            match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
            if not match:
                raise ValueError("Malformed JSON fence")
            text = match.group(1)
        obj = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(obj, dict) or {"answer", "confidence", "abstain"} != obj.keys():
            raise ValueError("Expected JSON object with answer, confidence and abstain")
        if type(obj["abstain"]) is not bool:
            raise ValueError("abstain must be boolean")
        confidence = obj["confidence"]
        if confidence is not None:
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("confidence must be null or a finite number in [0,1]")
            confidence = float(confidence)
        answer = obj["answer"]
        if spec["name"] == "forced_choice" and (obj["abstain"] or answer is None):
            raise ProtocolViolation("forced_choice requires an option letter and abstain=false")
        if obj["abstain"]:
            if answer is not None:
                raise ProtocolViolation("Abstention requires a null answer")
            if confidence is not None:
                raise ProtocolViolation("Abstention requires null confidence")
            return Prediction(None, confidence, True, True)
        if not isinstance(answer, str) or len(answer.strip()) != 1:
            raise ValueError("answer must be a single option letter")
        index = ord(answer.strip().upper()) - 65
        if not 0 <= index < num_choices:
            raise ValueError("Option letter outside the choice range")
        return Prediction(index, confidence, False, True)
    except (ValueError, TypeError, AttributeError) as error:
        return Prediction(None, None, False, False, str(error),
                          "protocol" if isinstance(error, ProtocolViolation) else "format")
