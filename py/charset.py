# license plate chars
CHARS = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

itos = {i: ch for i, ch in enumerate(CHARS)}
stoi = {ch: i for i, ch in enumerate(CHARS)}

def vocab_size():
    return len(CHARS)
