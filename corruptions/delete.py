import random 

def delete_char(text: str) -> str:
    if not text:
        return text 
    idx = random.randint(0, len(text) - 1)
    return text[:idx] + text[idx + 1:]
