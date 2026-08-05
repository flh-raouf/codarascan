#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from barcode_benchmark.restoration import TinyQRRestorer


class QRDNPairs(Dataset):
    def __init__(self, root: Path, patch_size: int, seed: int) -> None:
        self.patch_size = patch_size
        self.seed = seed
        self.pairs: list[tuple[Path, Path]] = []
        for input_folder, target_folder in (
            ("extracted One", "target"),
            ("extracted Voted", "target"),
            ("extracted Quad", "target quad"),
        ):
            for input_path in sorted((root / input_folder / "train").glob("*.jpg")):
                target_path = root / target_folder / "train" / input_path.name
                if not target_path.is_file():
                    raise FileNotFoundError(target_path)
                self.pairs.append((input_path, target_path))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_path, target_path = self.pairs[index]
        noisy = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
        if noisy is None or target is None:
            raise ValueError(f"cannot read pair: {input_path}, {target_path}")
        generator = random.Random(self.seed + 104729 * index + random.randrange(1 << 20))
        patch = min(self.patch_size, noisy.shape[0], noisy.shape[1])
        top = generator.randrange(noisy.shape[0] - patch + 1)
        left = generator.randrange(noisy.shape[1] - patch + 1)
        noisy = noisy[top:top + patch, left:left + patch]
        target = target[top:top + patch, left:left + patch]
        rotations = generator.randrange(4)
        noisy = np.rot90(noisy, rotations).copy()
        target = np.rot90(target, rotations).copy()
        if generator.random() < 0.5:
            noisy = np.fliplr(noisy).copy()
            target = np.fliplr(target).copy()
        return (
            torch.from_numpy(noisy.astype(np.float32)[None] / 255.0),
            torch.from_numpy(target.astype(np.float32)[None] / 255.0),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tiny QR-DN restorer")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=1729)
    arguments = parser.parse_args()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    dataset = QRDNPairs(arguments.root.resolve(), arguments.patch_size, arguments.seed)
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    device = torch.device(arguments.device)
    model = TinyQRRestorer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    loss_function = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(1, arguments.epochs + 1):
        total_loss = 0.0
        for step, (noisy, target) in enumerate(loader, start=1):
            noisy = noisy.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(noisy)
            loss = loss_function(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            if step % 100 == 0:
                print(
                    f"epoch={epoch}/{arguments.epochs} step={step}/{len(loader)} "
                    f"loss={total_loss / step:.6f}",
                    flush=True,
                )
        print(
            f"epoch={epoch}/{arguments.epochs} mean_loss={total_loss / len(loader):.6f}",
            flush=True,
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.to("cpu").state_dict(),
        "architecture": "TinyQRRestorer",
        "width": 8,
        "training": {
            "dataset": "QR-DN1.0 train split",
            "pairs": len(dataset),
            "epochs": arguments.epochs,
            "patch_size": arguments.patch_size,
            "seed": arguments.seed,
        },
    }, arguments.output)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
