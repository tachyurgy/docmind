import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """Remove control characters, normalize unicode, and clean whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    return normalize_whitespace(text)


def truncate(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length, appending suffix if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences using basic heuristics."""
    pattern = r"(?<=[.!?])\s+"
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]
