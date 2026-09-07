"""Lab 5 Part B — Applying a real pre-trained ResNet-50.

NOT part of the CI suite. This needs PyTorch (a large install) and downloads
about 100 MB of ImageNet weights plus a test image, so it cannot run in the
offline, no-network tier that gates every push. Run it by hand.

    pip install torch torchvision pillow requests
    python solution/resnet_demo.py --image path/to/photo.jpg

Part A implements convolution from scratch precisely so that this file is not
magic: resnet50 is the same operation, stacked fifty times with learned kernels.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="path to a JPEG or PNG")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()

    try:
        import torch
        from PIL import Image
        from torchvision import models, transforms
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print("install with:  pip install torch torchvision pillow", file=sys.stderr)
        return 1

    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.eval()  # inference mode: disables dropout, freezes batch-norm statistics

    total = sum(p.numel() for p in model.parameters())
    print(f"ResNet-50 parameters: {total:,}")
    print(f"First layer: {model.conv1}")
    print("  That is Part A's convolve2d, with 64 learned 7x7 kernels over 3 channels.")
    print()

    # The preprocessing must match what the network was trained on. Wrong
    # normalisation is a silent failure: the model still returns confident
    # predictions, they are just wrong.
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(a.image).convert("RGB")
    batch = preprocess(image).unsqueeze(0)  # add the batch dimension

    with torch.no_grad():
        probs = torch.nn.functional.softmax(model(batch)[0], dim=0)

    top = torch.topk(probs, a.topk)
    categories = weights.meta["categories"]
    print(f"Top {a.topk} predictions for {a.image}:")
    for score, idx in zip(top.values, top.indices):
        print(f"  {categories[idx]:<32} {score.item():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
