#!/usr/bin/env python3
"""
前立腺上皮セグメンテーションのベースライン。

ResNet34エンコーダ付きのU-Netで、Epithelium(クラス2)のみを学習する。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


@dataclass
class Config:
    """実行設定。"""

    input_dir: Path
    output_dir: Path
    batch_size: int
    epochs: int
    lr: float
    num_workers: int
    img_size: int
    device: str
    seed: int
    debug: bool


class SegmentationDataset(Dataset):
    """Epithelium(クラス2)のみを抽出するデータセット。"""

    def __init__(self, image_dir: Path, label_dir: Optional[Path], transform: Optional[object]) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_ids = sorted([f.stem for f in self.image_dir.glob("*.jpg")])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_id = self.image_ids[idx]
        image = np.array(Image.open(self.image_dir / f"{image_id}.jpg"))

        if self.label_dir is not None:
            label = np.array(Image.open(self.label_dir / f"{image_id}.png"))
            mask = (label == 2).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, image_id


def get_transforms(train: bool) -> Optional[object]:
    """前処理を返す。"""

    if not ALBUMENTATIONS_AVAILABLE:
        return None

    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


class DiceLoss(nn.Module):
    """Dice損失。"""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Dice + BCE の混合損失。"""

    def __init__(self, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(pred, target) + (1 - self.dice_weight) * self.bce(
            pred, target
        )


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Diceスコア。"""

    pred_bin = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection) / (pred_bin.sum() + target.sum() + 1e-8)


def rle_encode(mask: np.ndarray) -> str:
    """RLE形式に変換する。"""

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def set_seed(seed: int, debug: bool) -> None:
    """乱数種を固定する。"""

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if debug:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks, _ in tqdm(loader, desc="Train"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_dice += float(dice_score(outputs, masks).item())

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks, _ in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += float(loss.item())
        total_dice += float(dice_score(outputs, masks).item())

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, str]:
    model.eval()
    predictions: Dict[str, str] = {}

    for images, _, image_ids in tqdm(loader, desc="Predict"):
        images = images.to(device)
        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        for pred, image_id in zip(preds, image_ids):
            mask = pred.squeeze().astype(np.uint8)
            rle = rle_encode(mask)
            predictions[image_id] = rle

    return predictions


def parse_args() -> Config:
    """引数を読み取って設定を返す。"""

    parser = argparse.ArgumentParser(description="Prostate Epithelium Segmentation Baseline")
    parser.add_argument("--input-dir", type=Path, default=Path("input"), help="入力データのルート")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="出力先ディレクトリ")
    parser.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=1, help="学習エポック数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学習率")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoaderのワーカー数")
    parser.add_argument("--img-size", type=int, default=320, help="入力画像サイズ")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス",
    )
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--debug", action="store_true", help="デバッグモード")

    args = parser.parse_args()
    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        img_size=args.img_size,
        device=args.device,
        seed=args.seed,
        debug=args.debug,
    )


def resolve_model() -> nn.Module:
    """モデルを構築する。"""

    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch が見つかりません")
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


def maybe_limit_debug_dataset(dataset: Dataset, debug: bool) -> Dataset:
    """デバッグ時は小さなsubsetを返す。"""

    if not debug:
        return dataset
    limit = min(32, len(dataset))
    indices = list(range(limit))
    return torch.utils.data.Subset(dataset, indices)


def save_submission(predictions: Dict[str, str], output_path: Path) -> None:
    """提出ファイルを保存する。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Expected"])
        for image_id in sorted(predictions.keys()):
            writer.writerow([image_id, predictions[image_id]])


def load_best_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    """最良モデルの重みを読み込む。"""

    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def main() -> None:
    config = parse_args()
    set_seed(config.seed, config.debug)

    device = torch.device(config.device)
    print(f"Device: {device}")

    if config.debug:
        print("Debug mode: データ数を制限し、num_workers=0で実行します。")
        config = Config(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            batch_size=config.batch_size,
            epochs=config.epochs,
            lr=config.lr,
            num_workers=0,
            img_size=config.img_size,
            device=config.device,
            seed=config.seed,
            debug=config.debug,
        )

    if (config.input_dir / "train").exists():
        data_dir = config.input_dir
    else:
        data_dir = config.input_dir / "data"
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)

    if train_transform is None or val_transform is None:
        print("albumentations が見つからないため、基本変換のみを使用します。")

    train_dataset = SegmentationDataset(
        data_dir / "train" / "images",
        data_dir / "train" / "labels",
        transform=train_transform,
    )
    train_dataset = maybe_limit_debug_dataset(train_dataset, config.debug)

    n_train = int(len(train_dataset) * 0.8)
    n_val = len(train_dataset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

    model = resolve_model().to(device)
    criterion = DiceBCELoss()
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_dice = 0.0
    best_path = config.output_dir / "best_model.pth"
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (Dice: {best_dice:.4f})")

    print("\nGenerating predictions...")
    load_best_weights(model, best_path, device)

    test_dataset = SegmentationDataset(
        data_dir / "test" / "images",
        label_dir=None,
        transform=val_transform,
    )
    test_dataset = maybe_limit_debug_dataset(test_dataset, config.debug)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    predictions = predict(model, test_loader, device)

    submission_path = config.output_dir / "submission.csv"
    save_submission(predictions, submission_path)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
