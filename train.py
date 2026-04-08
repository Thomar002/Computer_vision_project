"""Training script for AOD-Net, DCPDN, and Color-Constrained Dehazing models."""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from config import (RESIDE_SOTS_HAZY, RESIDE_SOTS_GT, RESIDE_OUT_TRAIN_GT,
                    RESIDE_OUT_TEST_HAZY, RESIDE_OUT_TEST_GT,
                    CHECKPOINTS_DIR, TRAIN_CONFIG)
from datasets import RESIDEOutdoorTrainDataset, RESIDEOutdoorTestDataset, MatchedPairDataset
from metrics import compute_psnr, compute_ssim
from methods.aodnet import AODNet
from methods.dcpdn import DCPDN, Discriminator
from methods.color_dehaze import ColorConstrainedDehaze, ColorConsistencyLoss


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def build_train_dataset(image_size):
    """Build training dataset from available RESIDE data.

    Uses RESIDE OUT test set (matched hazy-GT pairs with same filenames)
    for training, since the train folder only has GT images.
    Falls back to SOTS outdoor if unavailable.
    """
    # RESIDE OUT test has matched pairs (same filename in GT/ and hazy/)
    out_test_hazy = RESIDE_OUT_TEST_HAZY
    out_test_gt = RESIDE_OUT_TEST_GT
    if os.path.exists(out_test_hazy) and os.path.exists(out_test_gt):
        dataset = MatchedPairDataset(out_test_hazy, out_test_gt, image_size,
                                     augment=True)
        if len(dataset) > 0:
            print(f"Using RESIDE OUT test set for training: {len(dataset)} pairs")
            return dataset

    # Fallback: SOTS outdoor (hazy name -> scene_id.png in clear)
    dataset = RESIDEOutdoorTestDataset(
        RESIDE_SOTS_HAZY, RESIDE_SOTS_GT, image_size)
    print(f"Using RESIDE SOTS dataset: {len(dataset)} pairs")

    class TrainWrapper:
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            hazy, gt, _ = self.ds[idx]
            return hazy, gt

    return TrainWrapper(dataset)


def train_aodnet(args):
    """Train AOD-Net."""
    device = get_device()
    print(f"Training AOD-Net on {device}")

    dataset = build_train_dataset(args.image_size)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = AODNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_psnr = 0
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        for hazy, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            hazy, gt = hazy.to(device), gt.to(device)
            output = model(hazy)
            loss = criterion(output, gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_psnr, val_ssim = 0, 0
        with torch.no_grad():
            for hazy, gt in val_loader:
                hazy, gt = hazy.to(device), gt.to(device)
                output = model(hazy)
                for i in range(output.shape[0]):
                    val_psnr += compute_psnr(output[i], gt[i])
                    val_ssim += compute_ssim(output[i], gt[i])

        n_val_samples = len(val_ds)
        val_psnr /= n_val_samples
        val_ssim /= n_val_samples

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
              f"Val PSNR={val_psnr:.2f}, Val SSIM={val_ssim:.4f}")

        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINTS_DIR, "aodnet_best.pth"))
            print(f"  Saved best model (PSNR={best_psnr:.2f})")

    # Save final model
    torch.save(model.state_dict(),
               os.path.join(CHECKPOINTS_DIR, "aodnet_final.pth"))
    print(f"Training complete. Best PSNR: {best_psnr:.2f}")


def train_dcpdn(args):
    """Train DCPDN with GAN loss."""
    device = get_device()
    print(f"Training DCPDN on {device}")

    dataset = build_train_dataset(args.image_size)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = DCPDN().to(device)
    discriminator = Discriminator().to(device)

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_bce = nn.BCEWithLogitsLoss()

    optimizer_g = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr * 0.5, betas=(0.5, 0.999))
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=5, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=5, gamma=0.5)

    best_psnr = 0
    for epoch in range(args.epochs):
        model.train()
        discriminator.train()
        train_loss_g, train_loss_d = 0, 0

        for hazy, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            hazy, gt = hazy.to(device), gt.to(device)

            # Forward pass
            dehazed, t_map, A = model(hazy)

            # --- Train Discriminator ---
            # Real: GT image with a "perfect" transmission map (ones)
            real_t = torch.ones_like(t_map)
            pred_real = discriminator(gt, real_t)
            pred_fake = discriminator(dehazed.detach(), t_map.detach())

            label_real = torch.ones_like(pred_real)
            label_fake = torch.zeros_like(pred_fake)

            loss_d = (criterion_bce(pred_real, label_real) +
                      criterion_bce(pred_fake, label_fake)) * 0.5

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # --- Train Generator ---
            pred_fake_g = discriminator(dehazed, t_map)
            loss_gan = criterion_bce(pred_fake_g, label_real)

            # Reconstruction losses
            loss_recon = criterion_mse(dehazed, gt)
            loss_l1 = criterion_l1(dehazed, gt)

            # Edge-preserving loss (gradient loss)
            dehazed_dx = dehazed[:, :, :, 1:] - dehazed[:, :, :, :-1]
            dehazed_dy = dehazed[:, :, 1:, :] - dehazed[:, :, :-1, :]
            gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
            gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
            loss_edge = criterion_l1(dehazed_dx, gt_dx) + criterion_l1(dehazed_dy, gt_dy)

            loss_g = loss_recon + 0.5 * loss_l1 + 0.01 * loss_gan + 0.1 * loss_edge

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            train_loss_g += loss_g.item()
            train_loss_d += loss_d.item()

        train_loss_g /= len(train_loader)
        train_loss_d /= len(train_loader)

        # Validation
        model.eval()
        val_psnr, val_ssim = 0, 0
        with torch.no_grad():
            for hazy, gt in val_loader:
                hazy, gt = hazy.to(device), gt.to(device)
                dehazed, _, _ = model(hazy)
                for i in range(dehazed.shape[0]):
                    val_psnr += compute_psnr(dehazed[i], gt[i])
                    val_ssim += compute_ssim(dehazed[i], gt[i])

        n_val_samples = len(val_ds)
        val_psnr /= n_val_samples
        val_ssim /= n_val_samples

        print(f"Epoch {epoch+1}: G_Loss={train_loss_g:.4f}, D_Loss={train_loss_d:.4f}, "
              f"Val PSNR={val_psnr:.2f}, Val SSIM={val_ssim:.4f}")

        scheduler_g.step()
        scheduler_d.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINTS_DIR, "dcpdn_best.pth"))
            print(f"  Saved best model (PSNR={best_psnr:.2f})")

    torch.save(model.state_dict(),
               os.path.join(CHECKPOINTS_DIR, "dcpdn_final.pth"))
    print(f"Training complete. Best PSNR: {best_psnr:.2f}")


def train_color_constrained(args):
    """Train Color-Constrained Dehazing Model."""
    device = get_device()
    print(f"Training Color-Constrained Dehazing on {device}")

    dataset = build_train_dataset(args.image_size)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ColorConstrainedDehaze().to(device)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    color_loss_fn = ColorConsistencyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_psnr = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for hazy, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            hazy, gt = hazy.to(device), gt.to(device)
            dehazed, atm_light = model(hazy)

            # Reconstruction loss
            loss_mse = criterion_mse(dehazed, gt)
            loss_l1 = criterion_l1(dehazed, gt)

            # Color consistency loss
            loss_color = color_loss_fn(dehazed, gt)

            # Total loss
            loss = loss_mse + 0.5 * loss_l1 + 0.1 * loss_color

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_psnr, val_ssim = 0, 0
        with torch.no_grad():
            for hazy, gt in val_loader:
                hazy, gt = hazy.to(device), gt.to(device)
                dehazed, _ = model(hazy)
                for i in range(dehazed.shape[0]):
                    val_psnr += compute_psnr(dehazed[i], gt[i])
                    val_ssim += compute_ssim(dehazed[i], gt[i])

        n_val_samples = len(val_ds)
        val_psnr /= n_val_samples
        val_ssim /= n_val_samples

        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, "
              f"Val PSNR={val_psnr:.2f}, Val SSIM={val_ssim:.4f}")

        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINTS_DIR, "color_dehaze_best.pth"))
            print(f"  Saved best model (PSNR={best_psnr:.2f})")

    torch.save(model.state_dict(),
               os.path.join(CHECKPOINTS_DIR, "color_dehaze_final.pth"))
    print(f"Training complete. Best PSNR: {best_psnr:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dehazing models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["aodnet", "dcpdn", "color"],
                        help="Model to train")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["num_epochs"])
    parser.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"])
    parser.add_argument("--image_size", type=int, default=TRAIN_CONFIG["image_size"])
    parser.add_argument("--num_workers", type=int, default=TRAIN_CONFIG["num_workers"])
    args = parser.parse_args()

    if args.model == "aodnet":
        train_aodnet(args)
    elif args.model == "dcpdn":
        train_dcpdn(args)
    elif args.model == "color":
        train_color_constrained(args)
