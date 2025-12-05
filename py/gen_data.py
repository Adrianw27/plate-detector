# gen_data.py
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random, string, math

OUT = Path("data")
TRAIN_N = 6000
VAL_N = 1200
W, H = 160, 48  # fixed, small
MAX_ANGLE = 3

def get_font(size=34):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def sample_text():
    # ABC-123 format only
    L = "".join(random.choice(string.ascii_uppercase) for _ in range(3))
    D = "".join(random.choice(string.digits) for _ in range(3))
    return f"{L}-{D}"

def render(txt):
    img = Image.new("L", (W, H), color=240)
    d = ImageDraw.Draw(img)
    font = get_font()
    # Use getbbox instead of deprecated textsize (removed in Pillow 10.0.0)
    bbox = d.textbbox((0, 0), txt, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (W - tw) // 2
    y = (H - th) // 2
    d.text((x, y), txt, fill=0, font=font)

    # small random blur/rotate so it's not too clean
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    if random.random() < 0.5:
        angle = random.uniform(-MAX_ANGLE, MAX_ANGLE)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
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
