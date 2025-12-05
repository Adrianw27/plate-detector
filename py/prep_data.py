
# Usage:
#   python prep_data.py --src real_raw

import argparse, random, re, shutil
from pathlib import Path
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def try_open(p: Path) -> bool:
    try:
        Image.open(p).convert("L")
        return True
    except Exception:
        return False

def label_from_name(stem: str) -> str | None:
    m = re.search(r"[A-Z0-9]+-?[A-Z0-9]+", stem.upper())
    return m.group(0) if m else None

def load_pairs_from_names(src: Path):
    pairs = []
    for p in src.iterdir():
        if p.suffix.lower() not in EXTS: 
            continue
        if not try_open(p): 
            continue
        lab = label_from_name(p.stem)
        if lab:
            pairs.append((p, lab))
    return pairs

def write_split(out_dir: Path, pairs):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
        for i, (src, lab) in enumerate(pairs):
            dst = out_dir / f"{i:06d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
            f.write(f"{dst.name}\t{lab}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder with CROPPED real plate images")
    ap.add_argument("--csv", help="optional CSV file: filename,label")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: source folder '{src}' not found")
        return

    if args.csv:
        pairs = load_pairs_from_csv(src, Path(args.csv))
    else:
        pairs = load_pairs_from_names(src)

    pairs = [(p, l) for (p, l) in pairs if l]
    if not pairs:
        print("ERROR: No usable (image,label) pairs found. Check filenames or CSV.")
        return

    random.Random(args.seed).shuffle(pairs)
    k = int(len(pairs) * (1 - args.val_ratio))
    train_pairs = pairs[:k]
    val_pairs   = pairs[k:]

    out = Path("data")
    write_split(out / "train", train_pairs)
    write_split(out / "val",   val_pairs)
    print(f"Done. train={len(train_pairs)}, val={len(val_pairs)}")

if __name__ == "__main__":
    main()
