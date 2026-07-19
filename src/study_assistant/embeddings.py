from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
PHRASE_ALIASES = {
    "maths": "mathematics math",
    "math": "mathematics math",
    "algebraic": "algebra variables expressions",
    "variables": "algebra variables",
    "variable": "algebra variables",
    "expressions": "algebra expressions",
    "equations": "algebra equations",
    "quadratics": "quadratic equations roots parabola",
    "quadratic": "quadratic equations roots parabola",
    "roots": "quadratic equations roots",
    "formula": "quadratic equations formula",
    "factorisation": "quadratic algebra factorisation",
    "factorization": "quadratic algebra factorisation",
    "science": "science physics chemistry",
    "physics": "science physics",
    "light": "light optics reflection refraction",
    "optics": "light optics reflection refraction",
    "ray": "light optics ray diagrams",
    "rays": "light optics ray diagrams",
    "mirror": "light reflection mirror",
    "mirrors": "light reflection mirror",
    "lens": "light refraction lens",
    "lenses": "light refraction lens",
    "reflection": "light reflection optics",
    "refraction": "light refraction optics",
    "weak": "weak struggle improve low score",
    "weaker": "weak struggle improve low score",
    "struggling": "weak struggle improve low score",
    "struggle": "weak struggle improve low score",
    "difficult": "weak struggle improve low score",
    "poor": "weak low score improve",
    "improve": "weak improve practice",
    "priority": "prioritize priority important urgent",
    "prioritize": "prioritize priority important urgent",
    "first": "prioritize first",
    "next": "next study plan",
    "week": "week study schedule plan",
    "weekly": "week study schedule plan test",
    "study": "study learn revise practice",
    "prepare": "prepare revise practice test",
    "preparation": "prepare revise practice test",
    "test": "test exam assessment quiz",
    "exam": "test exam assessment quiz",
    "assessment": "test exam assessment quiz",
    "coming": "upcoming future soon",
    "upcoming": "upcoming future soon",
    "material": "material resource notes video",
    "notes": "material resource notes",
    "video": "material resource video",
}


class LocalSemanticEmbedder:
    """Deterministic embedding model with alias expansion and hashed vectors."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        tokens = self._expanded_tokens(text)
        features = Counter(tokens)
        for token in list(tokens):
            for ngram in self._character_ngrams(token):
                features[f"char:{ngram}"] += 0.25

        vector = [0.0] * self.dimensions
        for feature, weight in features.items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            integer = int.from_bytes(digest, "big")
            index = integer % self.dimensions
            sign = 1.0 if (integer >> 8) % 2 == 0 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _expanded_tokens(self, text: str) -> list[str]:
        normalized = text.lower().replace("-", " ")
        tokens = TOKEN_PATTERN.findall(normalized)
        expanded = list(tokens)

        token_text = " ".join(tokens)
        for alias, replacement in PHRASE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", token_text):
                expanded.extend(TOKEN_PATTERN.findall(replacement))

        return expanded

    def _character_ngrams(self, token: str) -> list[str]:
        padded = f"_{token}_"
        ngrams: list[str] = []
        for size in (3, 4):
            if len(padded) >= size:
                ngrams.extend(
                    padded[index : index + size]
                    for index in range(len(padded) - size + 1)
                )
        return ngrams


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(left_value * right_value for left_value, right_value in zip(left, right))

