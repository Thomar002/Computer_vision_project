"""Dataset classes for RESIDE ITS, OTS, SOTS, and O-HAZE."""
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class RESIDETrainDataset(Dataset):
    """RESIDE ITS/OTS training dataset.

    GT filenames: {scene_id}.png (e.g. 1000.png)
    Hazy filenames: {scene_id}_{param1}_{param2}.png (e.g. 1000_10_0.74905.png)

    Each GT image is paired with all its hazy variants for training.
    """

    def __init__(self, gt_dir, hazy_dir, image_size=256):
        self.gt_dir = gt_dir
        self.hazy_dir = hazy_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.pairs = []
        if os.path.exists(hazy_dir) and os.path.exists(gt_dir):
            gt_files = {os.path.splitext(f)[0]: f
                        for f in os.listdir(gt_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
            for hazy_name in sorted(os.listdir(hazy_dir)):
                if not hazy_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                scene_id = hazy_name.split('_')[0]
                if scene_id in gt_files:
                    self.pairs.append((hazy_name, gt_files[scene_id]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_name, gt_name = self.pairs[idx]
        hazy_img = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        hazy_t = self.transform(hazy_img)
        torch.manual_seed(seed)
        gt_t = self.transform(gt_img)
        return hazy_t, gt_t


class RESIDETestDataset(Dataset):
    """RESIDE SOTS test dataset (indoor or outdoor).

    Indoor: GT={id}.png, Hazy={id}_{param}.png
    Outdoor: GT={id}.png, Hazy={id}_{param1}_{param2}.jpg
    """

    def __init__(self, hazy_dir, gt_dir, image_size=256):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.pairs = []
        if os.path.exists(hazy_dir) and os.path.exists(gt_dir):
            gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
            for hazy_name in sorted(os.listdir(hazy_dir)):
                if not hazy_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue
                scene_id = hazy_name.split('_')[0]
                if scene_id in gt_files:
                    self.pairs.append((hazy_name, gt_files[scene_id]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        hazy_name, gt_name = self.pairs[idx]
        hazy_img = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')
        return self.transform(hazy_img), self.transform(gt_img), hazy_name


class OHAZEDataset(Dataset):
    """O-HAZE dataset with real hazy/GT pairs."""

    def __init__(self, hazy_dir, gt_dir, image_size=256):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        hazy_files = sorted(os.listdir(hazy_dir))
        gt_files = set(os.listdir(gt_dir))
        self.pairs = []
        for hf in hazy_files:
            if hf.lower().endswith(('.jpg', '.png', '.jpeg')):
                if hf in gt_files:
                    self.pairs.append(hf)
                else:
                    hf_lower = hf.lower()
                    for gf in gt_files:
                        if gf.lower() == hf_lower:
                            self.pairs.append((hf, gf))
                            break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item = self.pairs[idx]
        if isinstance(item, tuple):
            hazy_name, gt_name = item
        else:
            hazy_name = gt_name = item
        hazy_img = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')
        hazy_name_str = hazy_name if isinstance(hazy_name, str) else hazy_name
        return self.transform(hazy_img), self.transform(gt_img), hazy_name_str
