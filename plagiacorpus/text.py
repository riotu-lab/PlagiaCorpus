"""Text utilities: language heuristics, word counting and Class 2 obfuscation."""

import re

_ARABIC_CHAR = re.compile(r"[؀-ۿ]")
_WORD = re.compile(r"\b\w+\b")

TATWEEL = "ـ"
LEETSPEAK = str.maketrans({"e": "3", "a": "@", "i": "1", "o": "0"})


def is_arabic(text: str) -> bool:
    """True when more than 10% of the characters belong to the Arabic Unicode block."""
    return len(_ARABIC_CHAR.findall(text)) > 0.1 * len(text)


def language_of(text: str) -> str:
    """Return 'ar' or 'en' using the `is_arabic` heuristic."""
    return "ar" if is_arabic(text) else "en"


def count_words(text: str) -> int:
    """Word count used for all length statistics (regex word tokens)."""
    return len(_WORD.findall(text or ""))


def obfuscate(text: str, lang: str) -> str:
    """Class 2 rule-based obfuscation.

    English: leetspeak substitutions e->3, a->@, i->1, o->0 (lower-case only).
    Arabic:  a tatweel (U+0640) is inserted after the first character of every
             word longer than two characters (whitespace is normalised to single spaces).
    """
    if lang == "en":
        return text.translate(LEETSPEAK)
    return " ".join(w[0] + TATWEEL + w[1:] if len(w) > 2 else w for w in text.split())
