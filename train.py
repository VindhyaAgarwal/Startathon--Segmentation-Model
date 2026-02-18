import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import segmentation_models_pytorch as smp

value_map = {
    0: 0, 100: 1, 200: 2, 300: 3, 500: 4,
    550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.mask_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.ids = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.mask_dir, name)))

        image = self.transform(image)

        mask = T.Resize(
            (image.shape[1], image.shape[2]),
            interpolation=T.InterpolationMode.NEAREST
        )(mask)

        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Conv2d(256, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        return self.classifier(x)

def compute_iou(pred, target):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(n_classes):
        intersection = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        if union == 0:
            continue
        ious.append((intersection / union).item())
    return np.mean(ious)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    batch_size = 2
    accumulation_steps = 2
    lr_backbone = 1e-5
    lr_head = 3e-4
    n_epochs = 30

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    train_transform = T.Compose([
        T.Resize((h, w)),
        T.RandomHorizontalFlip(0.5),
        T.RandomVerticalFlip(0.3),
        T.RandomRotation(10),
        T.ColorJitter(0.3,0.3,0.3),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_transform = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = "/content/dataset/Offroad_Segmentation_Training_Dataset/train"
    val_dir = "/content/dataset/Offroad_Segmentation_Training_Dataset/val"

    train_loader = DataLoader(MaskDataset(train_dir, train_transform),
                              batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(MaskDataset(val_dir, val_transform),
                            batch_size=2, shuffle=False)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.to(device)

    for name, param in backbone.named_parameters():
        if "blocks.10" in name or "blocks.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    sample = next(iter(train_loader))[0].to(device)
    with torch.no_grad():
        tokens = backbone.forward_features(sample)["x_norm_patchtokens"]

    embed_dim = tokens.shape[2]
    model = SegmentationHead(embed_dim, n_classes, w // 14, h // 14).to(device)

    ce_loss = nn.CrossEntropyLoss()
    dice_loss = smp.losses.DiceLoss(mode="multiclass")

    def loss_fn(pred, target):
        return ce_loss(pred, target) + dice_loss(pred, target)

    optimizer = optim.AdamW([
        {"params": model.parameters(), "lr": lr_head},
        {"params": [p for p in backbone.parameters() if p.requires_grad], "lr": lr_backbone}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = torch.cuda.amp.GradScaler()

    best_iou = 0

    for epoch in range(n_epochs):
        model.train()
        backbone.train()
        optimizer.zero_grad()
        total_loss = 0

        for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                outputs = model(feats)
                outputs = F.interpolate(outputs, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                loss = loss_fn(outputs, labels) / accumulation_steps

            scaler.scale(loss).backward()

            if (i+1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()

        scheduler.step()

        model.eval()
        backbone.eval()
        val_ious = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = backbone.forward_features(imgs)["x_norm_patchtokens"]
                outputs = model(feats)
                outputs = F.interpolate(outputs, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                val_ious.append(compute_iou(outputs, labels))

        mean_iou = np.mean(val_ious)
        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Val IoU: {mean_iou:.4f}")

        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save(model.state_dict(), "segmentation_head.pt")
            print("🔥 Saved Best Model")

    print("\nBest IoU:", best_iou)

if __name__ == "__main__":
    main()
