import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import os
from tqdm import tqdm
import cv2

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
    return new_arr

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

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_dir = "/content/dataset/Offroad_Segmentation_Training_Dataset/val"
    image_dir = os.path.join(val_dir, "Color_Images")
    mask_dir = os.path.join(val_dir, "Segmentation")

    ids = sorted(os.listdir(image_dir))

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    backbone.eval().to(device)

    sample_img = transform(Image.open(os.path.join(image_dir, ids[0])).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        tokens = backbone.forward_features(sample_img)["x_norm_patchtokens"]

    embed_dim = tokens.shape[2]
    model = SegmentationHead(embed_dim, n_classes, w // 14, h // 14).to(device)
    model.load_state_dict(torch.load("segmentation_head.pt", map_location=device))
    model.eval()

    all_ious = []
    os.makedirs("predictions", exist_ok=True)

    with torch.no_grad():
        for name in tqdm(ids):
            img = transform(Image.open(os.path.join(image_dir, name)).convert("RGB")).unsqueeze(0).to(device)
            raw_mask = convert_mask(Image.open(os.path.join(mask_dir, name)))

            feats = backbone.forward_features(img)["x_norm_patchtokens"]
            outputs = model(feats)
            outputs = F.interpolate(outputs, size=img.shape[2:], mode="bilinear", align_corners=False)

            pred_mask = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            gt_mask = torch.from_numpy(raw_mask).long().to(device)

            all_ious.append(compute_iou(outputs, gt_mask.unsqueeze(0)))

            cv2.imwrite(f"predictions/{name}", pred_mask)

    print("\nFINAL MEAN IoU:", np.mean(all_ious))

if __name__ == "__main__":
    main()
