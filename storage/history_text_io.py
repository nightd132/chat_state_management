from pathlib import Path
def save_text(text, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)

def load_text(path):
    with open(path, "r") as f:
        text = f.read()
    return text

def concatenate_texts(texts):
    return "\n".join(texts)