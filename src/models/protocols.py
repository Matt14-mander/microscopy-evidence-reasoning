"""Versioned answer protocols. Released templates must not be edited in place."""

import hashlib


DEFAULT_PROTOCOL = {"name": "allow_abstain", "version": "1.0"}
_COMMON = (
    "Answer the microscopy multiple-choice question using the provided evidence.\n"
    "Return only a JSON object with keys answer, confidence, abstain.\n"
    "answer: one option letter; confidence: a number in [0,1] or null.\n"
)
TEMPLATES = {
    ("allow_abstain", "1.0"): _COMMON + (
        'If evidence is insufficient, return {"answer": null, "confidence": null, "abstain": true}.\n'
        "Otherwise set abstain to false. Do not include explanations.\n\n"
        "Question: {question}\n{options}"
    ),
    ("forced_choice", "1.0"): _COMMON + (
        "Select the best available option even if evidence is insufficient or you are uncertain.\n"
        "You must provide an option letter and set abstain to false. Do not include explanations.\n\n"
        "Question: {question}\n{options}"
    ),
}


def text_sha256(text):
    """SHA256 of the exact UTF-8 text, without JSON quoting or newline normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_protocol(config=None):
    config = DEFAULT_PROTOCOL if config is None else config
    if not isinstance(config, dict) or set(config) != {"name", "version"}:
        raise ValueError("protocol requires exactly name and version")
    name, version = config["name"], config["version"]
    if not isinstance(name, str) or not isinstance(version, str) or (name, version) not in TEMPLATES:
        raise ValueError(f"Unsupported protocol/version: {name!r}/{version!r}")
    return {"name": name, "version": version, "protocol_id": f"{name}@{version}",
            "parser_version": "strict-json-1.0",
            "template_sha256": text_sha256(TEMPLATES[name, version])}


def render_prompt(question, choices, config=None):
    protocol = resolve_protocol(config)
    options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    # Literal JSON braces are part of the template; only the final question/options slot is expanded.
    template = TEMPLATES[protocol["name"], protocol["version"]]
    prefix = template.removesuffix("Question: {question}\n{options}")
    return prefix + f"Question: {question}\n{options}"
