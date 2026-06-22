import random

def typo_noise(text: str) -> str:
    chars = list(text)
    if len(chars) < 4:
        return text
    idx = random.randint(0, len(chars) - 2)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)
