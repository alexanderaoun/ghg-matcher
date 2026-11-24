import re
import unicodedata

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_SKU = re.compile(r"\b[a-z]{1,3}-?\d{3,}\b")
_QTY = re.compile(r"\b\d+([.,]\d+)?\s?(stk|pcs|x|ml|l|kg|g|ul|tb|gb|m|v|mm)?\b")
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower().translate(_UMLAUTS)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _SKU.sub(" ", text)
    text = _QTY.sub(" ", text)
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def with_vendor(description: str, vendor: str | None) -> str:
    return f"{description} vendor_{normalize(vendor).replace(' ', '_')}" if vendor else description
