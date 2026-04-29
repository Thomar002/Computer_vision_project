"""Load trained checkpoints and save readable evaluation results to a text file."""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from config import (SOTS_INDOOR_HAZY, SOTS_INDOOR_GT, SOTS_OUTDOOR_HAZY, SOTS_OUTDOOR_GT,
                    OHAZE_HAZY, OHAZE_GT, CHECKPOINTS_DIR, OUTPUTS_DIR, DCP_CONFIG)
from datasets import RESIDETestDataset, OHAZEDataset
from metrics import compute_psnr, compute_ssim
from methods.dcp import DarkChannelPrior
from methods.aodnet import AODNet
from methods.dcpdn import DCPDN
from methods.color_dehaze import ColorConstrainedDehaze


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def best_checkpoint_path(model_name, train_set):
    """Return best-checkpoint path using the same split suffix as train.py."""
    suffix = "" if train_set in ("auto", "its") else f"_{train_set}"
    return os.path.join(CHECKPOINTS_DIR, f"{model_name}{suffix}_best.pth")


def eval_dcp(dcp, dataloader, desc):
    psnr_list, ssim_list = [], []
    for hazy, gt, _ in tqdm(dataloader, desc=desc):
        for i in range(hazy.shape[0]):
            hazy_np = hazy[i].numpy().transpose(1, 2, 0)
            gt_np = gt[i].numpy().transpose(1, 2, 0)
            dehazed = dcp.dehaze(hazy_np)
            psnr_list.append(compute_psnr(dehazed, gt_np))
            ssim_list.append(compute_ssim(dehazed, gt_np))
    return np.array(psnr_list), np.array(ssim_list)


def eval_model(model, dataloader, model_name, device, desc):
    model.eval()
    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for hazy, gt, _ in tqdm(dataloader, desc=desc):
            hazy_dev = hazy.to(device)
            if model_name == "DCPDN":
                output, _, _ = model(hazy_dev)
            elif model_name == "Color-Constrained":
                output, _ = model(hazy_dev)
            else:
                output = model(hazy_dev)
            output = output.cpu()
            for i in range(output.shape[0]):
                psnr_list.append(compute_psnr(output[i], gt[i]))
                ssim_list.append(compute_ssim(output[i], gt[i]))
    return np.array(psnr_list), np.array(ssim_list)


def main():
    parser = argparse.ArgumentParser(description="Show dehazing evaluation results")
    parser.add_argument("--train_set", type=str, default="ots",
                        choices=["auto", "its", "ots"],
                        help="Checkpoint split to load: ots loads *_ots_best.pth")
    args = parser.parse_args()

    device = get_device()
    image_size = 256
    batch_size = 4

    # Load datasets
    datasets = {}
    if os.path.exists(SOTS_INDOOR_HAZY) and os.path.exists(SOTS_INDOOR_GT):
        ds = RESIDETestDataset(SOTS_INDOOR_HAZY, SOTS_INDOOR_GT, image_size)
        if len(ds) > 0:
            datasets["SOTS-Indoor"] = DataLoader(ds, batch_size=batch_size, num_workers=0)
    if os.path.exists(SOTS_OUTDOOR_HAZY) and os.path.exists(SOTS_OUTDOOR_GT):
        ds = RESIDETestDataset(SOTS_OUTDOOR_HAZY, SOTS_OUTDOOR_GT, image_size)
        if len(ds) > 0:
            datasets["SOTS-Outdoor"] = DataLoader(ds, batch_size=batch_size, num_workers=0)
    if os.path.exists(OHAZE_HAZY) and os.path.exists(OHAZE_GT):
        ds = OHAZEDataset(OHAZE_HAZY, OHAZE_GT, image_size)
        if len(ds) > 0:
            datasets["O-HAZE"] = DataLoader(ds, batch_size=batch_size, num_workers=0)

    if not datasets:
        print("No datasets found. Check config.py paths.")
        return

    # Check which models have checkpoints
    models = {}

    aod_path = best_checkpoint_path("aodnet", args.train_set)
    if os.path.exists(aod_path):
        aodnet = AODNet().to(device)
        aodnet.load_state_dict(torch.load(aod_path, map_location=device, weights_only=True))
        models["AOD-Net"] = aodnet
        print(f"Loaded AOD-Net from {aod_path}")
    else:
        print(f"AOD-Net checkpoint not found at {aod_path} - skipping")

    dcpdn_path = best_checkpoint_path("dcpdn", args.train_set)
    if os.path.exists(dcpdn_path):
        dcpdn = DCPDN().to(device)
        dcpdn.load_state_dict(torch.load(dcpdn_path, map_location=device, weights_only=True))
        models["DCPDN"] = dcpdn
        print(f"Loaded DCPDN from {dcpdn_path}")
    else:
        print(f"DCPDN checkpoint not found at {dcpdn_path} - skipping")

    color_path = best_checkpoint_path("color_dehaze", args.train_set)
    if os.path.exists(color_path):
        color_model = ColorConstrainedDehaze().to(device)
        color_model.load_state_dict(torch.load(color_path, map_location=device, weights_only=True))
        models["Color-Constrained"] = color_model
        print(f"Loaded Color-Constrained from {color_path}")
    else:
        print(f"Color-Constrained checkpoint not found at {color_path} - skipping")

    if not models:
        print("No trained models found. Run training first.")
        return

    dcp = DarkChannelPrior(**DCP_CONFIG)

    # Run evaluation
    all_results = {}
    for ds_name, loader in datasets.items():
        # DCP baseline
        psnr, ssim = eval_dcp(dcp, loader, f"DCP on {ds_name}")
        all_results[("DCP", ds_name)] = (psnr, ssim)

        # Trained models
        for model_name, model in models.items():
            psnr, ssim = eval_model(model, loader, model_name, device, f"{model_name} on {ds_name}")
            all_results[(model_name, ds_name)] = (psnr, ssim)

    # Build report
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUTS_DIR, f"results_{args.train_set}.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  DEHAZING EVALUATION RESULTS")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Device: {device}")
    lines.append(f"  Image size: {image_size}x{image_size}")
    lines.append(f"  Checkpoint split: {args.train_set}")
    lines.append("=" * 70)
    lines.append("")

    # Summary table
    lines.append(f"{'Method':<20} {'Dataset':<15} {'PSNR (dB)':<12} {'SSIM':<10} {'Num Images':<10}")
    lines.append("-" * 70)
    for (method, ds_name), (psnr, ssim) in all_results.items():
        lines.append(f"{method:<20} {ds_name:<15} {psnr.mean():<12.2f} {ssim.mean():<10.4f} {len(psnr):<10}")
    lines.append("-" * 70)
    lines.append("")

    # Per-dataset detailed stats
    for ds_name in datasets:
        lines.append(f"{'=' * 70}")
        lines.append(f"  DATASET: {ds_name}")
        lines.append(f"{'=' * 70}")
        lines.append("")
        lines.append(f"{'Method':<20} {'Mean PSNR':<12} {'Std PSNR':<12} {'Min PSNR':<12} {'Max PSNR':<12}")
        lines.append("-" * 70)
        for (method, dn), (psnr, _) in all_results.items():
            if dn == ds_name:
                lines.append(f"{method:<20} {psnr.mean():<12.2f} {psnr.std():<12.2f} {psnr.min():<12.2f} {psnr.max():<12.2f}")
        lines.append("")
        lines.append(f"{'Method':<20} {'Mean SSIM':<12} {'Std SSIM':<12} {'Min SSIM':<12} {'Max SSIM':<12}")
        lines.append("-" * 70)
        for (method, dn), (_, ssim) in all_results.items():
            if dn == ds_name:
                lines.append(f"{method:<20} {ssim.mean():<12.4f} {ssim.std():<12.4f} {ssim.min():<12.4f} {ssim.max():<12.4f}")
        lines.append("")

    # Model info
    lines.append("=" * 70)
    lines.append("  MODEL INFO")
    lines.append("=" * 70)
    lines.append("")
    for model_name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        if model_name == "AOD-Net":
            ckpt = aod_path
        elif model_name == "DCPDN":
            ckpt = dcpdn_path
        else:
            ckpt = color_path
        size_mb = os.path.getsize(ckpt) / (1024 * 1024)
        lines.append(f"  {model_name}:")
        lines.append(f"    Parameters:      {params:,}")
        lines.append(f"    Checkpoint size:  {size_mb:.2f} MB")
        lines.append(f"    Checkpoint path:  {ckpt}")
        lines.append("")

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
