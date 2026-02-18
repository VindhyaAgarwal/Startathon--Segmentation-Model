import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Class mapping from problem statement
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
n_classes = len(value_map)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class MaskDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_transform=None):
        self.image_dir, self.masks_dir = image_dir, mask_dir
        self.transform, self.mask_transform = transform, mask_transform
        self.data_ids = sorted(os.listdir(self.image_dir))
    def __len__(self): return len(self.data_ids)
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))
        if self.transform:
            image = self.transform(image)
            mask = (self.mask_transform(mask) * 255).long()
        return image, mask

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.block = nn.Sequential(
            nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 1), nn.GELU(),
            nn.Conv2d(512, 256, 1), nn.GELU()
        )
        self.classifier = nn.Conv2d(256, out_channels, 1)
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

def get_iou(pred, target):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(n_classes):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union > 0: ious.append(inter / union)
    return np.mean(ious) if ious else 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # RESOLUTION INCREASED TO 518
    batch_size, w, h, lr, n_epochs = 4, 518, 518, 2e-4, 12

    transform = transforms.Compose([
        transforms.Resize((h, w)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([transforms.Resize((h, w), interpolation=Image.NEAREST), transforms.ToTensor()])

    train_img = '/content/dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images'
    train_mask = '/content/dataset/Offroad_Segmentation_Training_Dataset/train/Segmentation'
    val_img = '/content/dataset/Offroad_Segmentation_Training_Dataset/val/Color_Images'
    val_mask = '/content/dataset/Offroad_Segmentation_Training_Dataset/val/Segmentation'
    
    train_loader = DataLoader(MaskDataset(train_img, train_mask, transform, mask_transform), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(MaskDataset(val_img, val_mask, transform, mask_transform), batch_size=batch_size, shuffle=False)

    print("Loading DINOv2 BASE Backbone...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
    classifier = SegmentationHeadConvNeXt(768, n_classes, w//14, h//14).to(device)
    
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = GradScaler()
    
    best_iou = 0.0
    for epoch in range(n_epochs):
        classifier.train()
        train_ious = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device).squeeze(1)
            optimizer.zero_grad()
            with autocast():
                with torch.no_grad(): features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(features)
                logits = F.interpolate(logits, size=(h, w), mode="bilinear")
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_ious.append(get_iou(logits, labels))
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{np.mean(train_ious):.3f}")
        
        scheduler.step()
        classifier.eval()
        val_ious = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device).squeeze(1)
                features = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits = F.interpolate(classifier(features), size=(h, w), mode="bilinear")
                val_ious.append(get_iou(logits, labels))
        
        avg_val_iou = np.mean(val_ious)
        print(f"Epoch {epoch+1} Val IoU: {avg_val_iou:.4f}")
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(classifier.state_dict(), "best_model.pth")
    print(f"Final Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()