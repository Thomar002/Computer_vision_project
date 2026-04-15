"""Run the full dehazing experiment: train all models, then evaluate and compare."""
import os
import subprocess
import sys
import argparse


def run_command(cmd, description):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"WARNING: {description} finished with return code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full dehazing experiment")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training, only evaluate")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_training:
        # Train AOD-Net
        code = run_command(
            f'{python} train.py --model aodnet --epochs {args.epochs} '
            f'--batch_size {args.batch_size} --image_size {args.image_size}',
            "Training AOD-Net"
        )
        if code != 0:
            sys.exit(code)

        # Train DCPDN
        code = run_command(
            f'{python} train.py --model dcpdn --epochs {args.epochs} '
            f'--batch_size {args.batch_size} --image_size {args.image_size}',
            "Training DCPDN"
        )
        if code != 0:
            sys.exit(code)

        # Train Color-Constrained
        code = run_command(
            f'{python} train.py --model color --epochs {args.epochs} '
            f'--batch_size {args.batch_size} --image_size {args.image_size}',
            "Training Color-Constrained Dehazing"
        )
        if code != 0:
            sys.exit(code)

    # Evaluate all methods
    code = run_command(
        f'{python} evaluate.py --image_size {args.image_size} '
        f'--batch_size 4 --dataset all',
        "Evaluating all methods on SOTS (indoor/outdoor) and O-HAZE"
    )
    if code != 0:
        sys.exit(code)

    print("\n" + "="*60)
    print("  EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: outputs/")
    print("Check outputs/ for comparison images (SOTS-Indoor, SOTS-Outdoor, O-HAZE)")


if __name__ == "__main__":
    main()
