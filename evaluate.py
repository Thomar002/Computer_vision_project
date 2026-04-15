"""Evaluation script: run all 4 methods on RESIDE SOTS and O-HAZE, compute PSNR/SSIM."""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

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


def count_images(directory):
    """Count image files in a directory."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1 for name in os.listdir(directory)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def evaluate_dcp(dataloader, dataset_name):
    """Evaluate Dark Channel Prior on a dataset."""
    dcp = DarkChannelPrior(**DCP_CONFIG)
    psnr_list, ssim_list = [], []
    results = []

    for hazy, gt, name in tqdm(dataloader, desc=f"DCP on {dataset_name}"):
        for i in range(hazy.shape[0]):
            hazy_np = hazy[i].numpy().transpose(1, 2, 0)
            gt_np = gt[i].numpy().transpose(1, 2, 0)

            dehazed = dcp.dehaze(hazy_np)

            p = compute_psnr(dehazed, gt_np)
            s = compute_ssim(dehazed, gt_np)
            psnr_list.append(p)
            ssim_list.append(s)

            if len(results) < 5:
                fname = name[i] if isinstance(name, (list, tuple)) else name
                results.append((fname, hazy_np, dehazed, gt_np))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    return avg_psnr, avg_ssim, results


def evaluate_model(model, dataloader, dataset_name, model_name, device):
    """Evaluate a PyTorch dehazing model on a dataset."""
    model.eval()
    psnr_list, ssim_list = [], []
    results = []

    with torch.no_grad():
        for hazy, gt, name in tqdm(dataloader, desc=f"{model_name} on {dataset_name}"):
            hazy_dev = hazy.to(device)

            if model_name == "DCPDN":
                output, _, _ = model(hazy_dev)
            elif model_name == "Color-Constrained":
                output, _ = model(hazy_dev)
            else:
                output = model(hazy_dev)

            output = output.cpu()
            for i in range(output.shape[0]):
                p = compute_psnr(output[i], gt[i])
                s = compute_ssim(output[i], gt[i])
                psnr_list.append(p)
                ssim_list.append(s)

                if len(results) < 5:
                    fname = name[i] if isinstance(name, (list, tuple)) else name
                    results.append((
                        fname,
                        hazy[i].numpy().transpose(1, 2, 0),
                        output[i].numpy().transpose(1, 2, 0),
                        gt[i].numpy().transpose(1, 2, 0),
                    ))

    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    return avg_psnr, avg_ssim, results


def load_model(model_class, checkpoint_name, device):
    """Load a trained model from checkpoint."""
    model = model_class().to(device)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, checkpoint_name)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}, using random weights!")
    return model


def best_checkpoint_name(model_name, train_set):
    """Return best-checkpoint filename using the same split suffix as train.py."""
    suffix = "" if train_set == "auto" else f"_{train_set}"
    return f"{model_name}{suffix}_best.pth"


def save_visual_comparison(all_results, dataset_name, output_dir):
    """Save visual comparison grid for the first few images."""
    os.makedirs(output_dir, exist_ok=True)

    method_names = list(all_results.keys())
    n_examples = min(5, min(len(r) for r in all_results.values()))
    if n_examples == 0:
        return

    fig, axes = plt.subplots(n_examples, len(method_names) + 2,
                             figsize=(4 * (len(method_names) + 2), 4 * n_examples))
    if n_examples == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_examples):
        # Hazy input (from first method)
        _, hazy, _, gt = all_results[method_names[0]][i]
        axes[i, 0].imshow(np.clip(hazy, 0, 1))
        axes[i, 0].set_title("Hazy Input" if i == 0 else "")
        axes[i, 0].axis('off')

        # Ground truth
        axes[i, 1].imshow(np.clip(gt, 0, 1))
        axes[i, 1].set_title("Ground Truth" if i == 0 else "")
        axes[i, 1].axis('off')

        # Each method's result
        for j, method in enumerate(method_names):
            _, _, dehazed, _ = all_results[method][i]
            axes[i, j + 2].imshow(np.clip(dehazed, 0, 1))
            axes[i, j + 2].set_title(method if i == 0 else "")
            axes[i, j + 2].axis('off')

    plt.suptitle(f"Dehazing Comparison - {dataset_name}", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"comparison_{dataset_name}.png"),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison figure: comparison_{dataset_name}.png")


def main():
    parser = argparse.ArgumentParser(description="Evaluate dehazing methods")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["sots_indoor", "sots_outdoor", "ohaze", "all"])
    parser.add_argument("--train_set", type=str, default="ots",
                        choices=["auto", "its", "ots"],
                        help="Checkpoint split to load: ots loads *_ots_best.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Evaluating on {device}")

    # Prepare datasets
    datasets_to_eval = {}
    skipped_datasets = []
    if args.dataset in ("sots_indoor", "all"):
        ds = RESIDETestDataset(SOTS_INDOOR_HAZY, SOTS_INDOOR_GT, args.image_size)
        if len(ds) > 0:
            datasets_to_eval["SOTS-Indoor"] = DataLoader(
                ds, batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            skipped_datasets.append(
                f"SOTS-Indoor: 0 matched pairs "
                f"(hazy files={count_images(SOTS_INDOOR_HAZY)}, "
                f"gt files={count_images(SOTS_INDOOR_GT)})"
            )

    if args.dataset in ("sots_outdoor", "all"):
        ds = RESIDETestDataset(SOTS_OUTDOOR_HAZY, SOTS_OUTDOOR_GT, args.image_size)
        if len(ds) > 0:
            datasets_to_eval["SOTS-Outdoor"] = DataLoader(
                ds, batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            skipped_datasets.append(
                f"SOTS-Outdoor: 0 matched pairs "
                f"(hazy files={count_images(SOTS_OUTDOOR_HAZY)}, "
                f"gt files={count_images(SOTS_OUTDOOR_GT)})"
            )

    if args.dataset in ("ohaze", "all"):
        ohaze_ds = OHAZEDataset(OHAZE_HAZY, OHAZE_GT, args.image_size)
        if len(ohaze_ds) > 0:
            datasets_to_eval["O-HAZE"] = DataLoader(
                ohaze_ds, batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            skipped_datasets.append(
                f"O-HAZE: 0 matched pairs "
                f"(hazy files={count_images(OHAZE_HAZY)}, "
                f"gt files={count_images(OHAZE_GT)})"
            )

    if skipped_datasets:
        print("Dataset checks:")
        for line in skipped_datasets:
            print(f"  - {line}")

    if not datasets_to_eval:
        raise RuntimeError(
            "No evaluation datasets available. Populate the expected hazy/GT folders "
            "before running evaluate.py."
        )

    # Load models
    aodnet = load_model(AODNet, best_checkpoint_name("aodnet", args.train_set), device)
    dcpdn = load_model(DCPDN, best_checkpoint_name("dcpdn", args.train_set), device)
    color_model = load_model(
        ColorConstrainedDehaze, best_checkpoint_name("color_dehaze", args.train_set), device)

    # Evaluate all methods on all datasets
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'Dataset':<15} {'PSNR (dB)':<12} {'SSIM':<10}")
    print("=" * 70)

    for ds_name, dataloader in datasets_to_eval.items():
        all_results = {}

        # 1. DCP
        psnr, ssim, results = evaluate_dcp(dataloader, ds_name)
        print(f"{'DCP':<25} {ds_name:<15} {psnr:<12.2f} {ssim:<10.4f}")
        all_results["DCP"] = results

        # 2. AOD-Net
        psnr, ssim, results = evaluate_model(
            aodnet, dataloader, ds_name, "AOD-Net", device)
        print(f"{'AOD-Net':<25} {ds_name:<15} {psnr:<12.2f} {ssim:<10.4f}")
        all_results["AOD-Net"] = results

        # 3. DCPDN
        psnr, ssim, results = evaluate_model(
            dcpdn, dataloader, ds_name, "DCPDN", device)
        print(f"{'DCPDN':<25} {ds_name:<15} {psnr:<12.2f} {ssim:<10.4f}")
        all_results["DCPDN"] = results

        # 4. Color-Constrained
        psnr, ssim, results = evaluate_model(
            color_model, dataloader, ds_name, "Color-Constrained", device)
        print(f"{'Color-Constrained':<25} {ds_name:<15} {psnr:<12.2f} {ssim:<10.4f}")
        all_results["Color-Constrained"] = results

        print("-" * 70)

        # Save visual comparison
        save_visual_comparison(all_results, ds_name, OUTPUTS_DIR)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
