import random

def word_dropout(text: str, p: float = 0.2) -> str:
    words = text.split()
    kept = [w for w in words if random.random() > p]
    if not kept:
        return text
    return " ".join(kept)
