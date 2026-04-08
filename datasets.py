"""Dataset classes for RESIDE and O-HAZE."""
import os
import re
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class RESIDEOutdoorTrainDataset(Dataset):
    """RESIDE outdoor training set (RESIDE OUT train).

    GT filenames like 0001_0.85_0.04.jpg where 0001 is the scene ID.
    Hazy images are in the SOTS hazy folder with same naming convention.
    """

    def __init__(self, gt_dir, hazy_dir, image_size=256):
        self.gt_dir = gt_dir
        self.hazy_dir = hazy_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # Build pairs: each GT image has a matching hazy image
        self.pairs = []
        gt_files = sorted(os.listdir(gt_dir))
        hazy_files = set(os.listdir(hazy_dir))
        for gt_name in gt_files:
            if gt_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                # Try to find matching hazy image with same name
                if gt_name in hazy_files:
                    self.pairs.append((gt_name, gt_name))
                else:
                    # GT name is the scene ID, find any hazy version
                    scene_id = gt_name.split('_')[0]
                    for hf in hazy_files:
                        if hf.startswith(scene_id + '_'):
                            self.pairs.append((gt_name, hf))
                            break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_name, hazy_name = self.pairs[idx]
        gt_img = Image.open(os.path.join(self.gt_dir, gt_name)).convert('RGB')
        hazy_img = Image.open(os.path.join(self.hazy_dir, hazy_name)).convert('RGB')
        return self.transform(hazy_img), self.transform(gt_img)


class RESIDEOutdoorTestDataset(Dataset):
    """RESIDE SOTS outdoor test set.

    Hazy: 0001_0.8_0.2.jpg  ->  GT: 0001.png
    """

    def __init__(self, hazy_dir, gt_dir, image_size=256):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.pairs = []
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
                # Same filename in both folders
                if hf in gt_files:
                    self.pairs.append(hf)
                else:
                    # Try case-insensitive match
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


class MatchedPairDataset(Dataset):
    """Dataset where hazy and GT images have the same filename."""

    def __init__(self, hazy_dir, gt_dir, image_size=256, augment=False):
        self.hazy_dir = hazy_dir
        self.gt_dir = gt_dir

        base_transform = [transforms.Resize((image_size, image_size))]
        if augment:
            base_transform.append(transforms.RandomHorizontalFlip())
        base_transform.append(transforms.ToTensor())

        self.transform = transforms.Compose(base_transform)
        self.augment = augment

        # Find matching filenames
        hazy_files = set(os.listdir(hazy_dir))
        gt_files = set(os.listdir(gt_dir))
        common = sorted(hazy_files & gt_files)
        self.filenames = [f for f in common
                          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        hazy_img = Image.open(os.path.join(self.hazy_dir, name)).convert('RGB')
        gt_img = Image.open(os.path.join(self.gt_dir, name)).convert('RGB')
        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            hazy_t = self.transform(hazy_img)
            torch.manual_seed(seed)
            gt_t = self.transform(gt_img)
        else:
            hazy_t = self.transform(hazy_img)
            gt_t = self.transform(gt_img)
        return hazy_t, gt_t


class RESIDETrainDataset(Dataset):
    """RESIDE OUT training dataset for deep learning methods.

    Uses RESIDE OUT train GT images paired with RESIDE OUT test hazy images,
    or generates training pairs from the available data.
    """

    def __init__(self, gt_dir, hazy_dir, image_size=256):
        self.gt_dir = gt_dir
        self.hazy_dir = hazy_dir
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.transform_val = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.pairs = []
        if os.path.exists(hazy_dir):
            gt_files = {os.path.splitext(f)[0].split('_')[0]: f
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
        # Use same random seed for both transforms to get same augmentation
        seed = torch.randint(0, 2**32, (1,)).item()
        torch.manual_seed(seed)
        hazy_t = self.transform(hazy_img)
        torch.manual_seed(seed)
        gt_t = self.transform(gt_img)
        return hazy_t, gt_t
