import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
from charset import stoi, itos, vocab_size

W, H = 160, 48
BLANK = vocab_size()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LPDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.items = []
        with open(self.root/"labels.txt", "r", encoding="utf-8") as f:
            for line in f:
                n, t = line.strip().split("\t")
                self.items.append((self.root/n, t))
        self.tx = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, t = self.items[i]
        img = self.tx(Image.open(p).convert("L"))
        tgt = torch.tensor([stoi[c] for c in t], dtype=torch.long)
        return img, tgt

def collate(batch):
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs, 0)
    tgt_lens = torch.tensor([t.size(0) for t in tgts], dtype=torch.long)
    tgts = torch.cat(tgts, 0)
    in_lens = torch.full((imgs.size(0),), W // 4, dtype=torch.long)
    return imgs, tgts, in_lens, tgt_lens

class CRNN(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(256*(H//4), 256, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, K+1)
    def forward(self, x):
        f = self.cnn(x)  # [B,C,H/4,W/4]
        B, C, H4, W4 = f.size()
        f = f.permute(0,3,1,2).contiguous().view(B, W4, C*H4)
        y,_ = self.lstm(f)
        y = self.fc(y)          # [B,T,C]
        return y.permute(1,0,2) # [T,B,C] for CTC

def main():
    train_ds = LPDataset("data/train")
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate)

    m = CRNN(vocab_size()).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    for ep in range(3):
        for imgs, tgts, in_lens, tgt_lens in train_dl:
            imgs, tgts, in_lens, tgt_lens = imgs.to(DEVICE), tgts.to(DEVICE), in_lens.to(DEVICE), tgt_lens.to(DEVICE)
            logp = F.log_softmax(m(imgs), dim=-1)
            loss = crit(logp, tgts, in_lens, tgt_lens)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("epoch", ep+1, "loss", float(loss))

    Path("ckpts").mkdir(exist_ok=True)
    torch.save({"model": m.state_dict()}, "ckpts/crnn.pt")
    print("saved ckpts/crnn.pt")

if __name__ == "__main__":
    main()
