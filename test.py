import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from tqdm import tqdm

# Mapping from raw pixel values to new class IDs
value_map = {0: 0, 100: 1, 200: 2, 300: 3, 500: 4, 550: 5, 700: 6, 800: 7, 7100: 8, 10000: 9}
class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes', 'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky']
n_classes = len(value_map)

color_palette = np.array([[0,0,0], [34,139,34], [0,255,0], [210,180,140], [139,90,43], [128,128,0], [139,69,19], [128,128,128], [160,82,45], [135,206,235]], dtype=np.uint8)

def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = sorted(os.listdir(self.image_dir))
    def __len__(self): return len(self.data_ids)
    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))
        if self.transform:
            image = self.transform(image)
            mask = (self.mask_transform(mask) * 255).long()
        return image, mask, data_id

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        self.stem = nn.Sequential(nn.Conv2d(in_channels, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU())
        self.block = nn.Sequential(nn.Conv2d(256, 256, 7, padding=3, groups=256), nn.BatchNorm2d(256), nn.GELU(), nn.Conv2d(256, 512, 1), nn.GELU(), nn.Conv2d(512, 256, 1), nn.GELU())
        self.classifier = nn.Conv2d(256, out_channels, 1)
    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.classifier(self.block(self.stem(x)))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '/content/best_model.pth'
    data_dir = '/content/dataset/Offroad_Segmentation_testImages'
    output_dir = '/content/predictions'
    os.makedirs(output_dir, exist_ok=True)

    w, h = 518, 518 
    transform = transforms.Compose([transforms.Resize((h, w)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_transform = transforms.Compose([transforms.Resize((h, w), interpolation=Image.NEAREST), transforms.ToTensor()])

    valset = MaskDataset(data_dir=data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=4, shuffle=False)

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
    classifier = SegmentationHeadConvNeXt(768, n_classes, w//14, h//14).to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=device))
    classifier.eval()

    print(f"Evaluating {len(valset)} test images...")
    with torch.no_grad():
        for imgs, _, data_ids in tqdm(val_loader):
            imgs = imgs.to(device)
            features = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits = F.interpolate(classifier(features), size=(540, 960), mode="bilinear")
            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

            for i in range(imgs.shape[0]):
                Image.fromarray(preds[i]).save(os.path.join(output_dir, data_ids[i]))

    print(f"Predictions saved to {output_dir}")

if __name__ == "__main__":
    main()