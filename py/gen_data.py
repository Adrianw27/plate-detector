# small data generator
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random, string

OUT = Path("data")
TRAIN_N = 2000
VAL_N = 400
W, H = 160, 48

def get_font(size=34):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def sample_text():
    # ABC-123 format only for now
    L = "".join(random.choice(string.ascii_uppercase) for _ in range(3))
    D = "".join(random.choice(string.digits) for _ in range(3))
    return f"{L}-{D}"

def render(txt):
    img = Image.new("L", (W, H), color=240)
    d = ImageDraw.Draw(img)
    font = get_font()
    tw, th = d.textsize(txt, font=font)
    x = (W - tw) // 2
    y = (H - th) // 2
    d.text((x, y), txt, fill=0, font=font)
    return img

def write_split(root, n):
    root.mkdir(parents=True, exist_ok=True)
    with open(root/"labels.txt", "w", encoding="utf-8") as f:
        for i in range(n):
            t = sample_text()
            im = render(t)
            name = f"{i:06d}.png"
            im.save(root/name)
            f.write(f"{name}\t{t}\n")

if __name__ == "__main__":
    write_split(OUT/"train", TRAIN_N)
    write_split(OUT/"val", VAL_N)
    print("done")
