# infer.py
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from charset import itos, vocab_size
from train import CRNN, W, H, BLANK, DEVICE
import torch.nn.functional as F

tx = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def greedy_decode(logits):  # [T,B,C]
    out = logits.argmax(-1)
    T, B = out.shape
    res = []
    for b in range(B):
        prev = BLANK
        s = []
        for t in range(T):
            k = out[t, b].item()
            if k != BLANK and k != prev:
                s.append(itos[k])
            prev = k
        res.append("".join(s))
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--ckpt", default="ckpts/crnn.pt")
    args = ap.parse_args()

    assert Path(args.ckpt).exists(), "train first"
    m = CRNN(vocab_size()).to(DEVICE)
    ckpt = torch.load(args.ckpt, map_location=DEVICE)
    m.load_state_dict(ckpt["model"])
    m.eval()

    x = tx(Image.open(args.img).convert("L")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = m(x)
    print("Prediction:", greedy_decode(logits)[0])
