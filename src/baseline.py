#!/usr/bin/env python3
"""
前立腺上皮セグメンテーションの強化ベースライン。

U-Net系やDeepLab系を選択でき、Epithelium(クラス2)のみを学習する。
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

try:
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast

    AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

    def autocast_context(enabled: bool) -> amp_autocast:
        return amp_autocast(device_type=AMP_DEVICE_TYPE, enabled=enabled)

    GradScaler = AmpGradScaler
except ImportError:  # pragma: no cover - 古いPyTorch向け
    from torch.cuda.amp import GradScaler as AmpGradScaler
    from torch.cuda.amp import autocast as amp_autocast

    AMP_DEVICE_TYPE = None

    def autocast_context(enabled: bool) -> amp_autocast:
        return amp_autocast(enabled=enabled)

    GradScaler = AmpGradScaler



try:
    import segmentation_models_pytorch as smp

    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False

try:
    if SMP_AVAILABLE:
        from segmentation_models_pytorch.base import SegmentationHead
        from segmentation_models_pytorch.base import model as smp_base_model
        from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
        from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
        from segmentation_models_pytorch.encoders import get_encoder
except Exception:  # pragma: no cover - オプション依存
    pass

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


@dataclass
class Config:
    """実行設定。

    Attributes:
        input_dir: 入力データのルート。
        output_dir: 出力先ディレクトリ。
        batch_size: バッチサイズ。
        epochs: 学習エポック数。
        lr: 学習率。
        weight_decay: Weight decay。
        num_workers: DataLoaderのワーカー数。
        img_size: 入力画像サイズ。
        device: 使用デバイス。
        seed: 乱数シード。
        debug: デバッグモード。
        val_ratio: 検証データの割合。
        arch: モデルアーキテクチャ。
        encoder: エンコーダ名。
        encoder_weights: エンコーダ重み。
        loss_name: 損失関数の種類。
        dice_weight: Diceの重み。
        focal_alpha: Focal lossのalpha。
        focal_gamma: Focal lossのgamma。
        tversky_alpha: Tversky lossのalpha。
        tversky_beta: Tversky lossのbeta。
        threshold: 推論時の二値化しきい値。
        min_area: 予測マスクの最小面積。
        tta: 推論時TTAの有無。
        amp: AMPの有無。
        scheduler: 学習率スケジューラ。
        pin_memory: DataLoaderのpin_memory。
        context_slices: 前後スライス数。
        auto_threshold: 検証でしきい値探索するか。
        thresholds: しきい値候補。
        sampler: サンプラー種類。
        pos_boost: 正例サンプルの重み強化。
    """

    input_dir: Path
    output_dir: Path
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    num_workers: int
    img_size: int
    device: str
    seed: int
    debug: bool
    val_ratio: float
    arch: str
    encoder: str
    encoder_weights: Optional[str]
    loss_name: str
    dice_weight: float
    focal_alpha: float
    focal_gamma: float
    tversky_alpha: float
    tversky_beta: float
    threshold: float
    min_area: int
    tta: bool
    amp: bool
    scheduler: str
    pin_memory: bool
    context_slices: int
    auto_threshold: bool
    thresholds: List[float]
    sampler: str
    pos_boost: float
    onecycle_pct_start: float
    onecycle_div_factor: float
    onecycle_final_div_factor: float


class SegmentationDataset(Dataset):
    """Epithelium(クラス2)のみを抽出するデータセット。

    Args:
        image_dir: 画像ディレクトリ。
        label_dir: ラベルディレクトリ。
        image_ids: 画像ID一覧。
        transform: 画像変換。
        img_size: 画像サイズ。
        metadata: メタデータ。
        context_slices: 前後スライス数。
    """

    def __init__(
        self,
        image_dir: Path,
        label_dir: Optional[Path],
        image_ids: List[str],
        transform: Optional[object],
        img_size: int,
        metadata: Dict[str, Dict[str, int]],
        context_slices: int,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_ids = image_ids
        self.transform = transform
        self.img_size = img_size
        self.metadata = metadata
        self.context_slices = context_slices
        self.crop_index = build_crop_index(metadata)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        image_id = self.image_ids[idx]
        image_ids = resolve_context_ids(
            image_id=image_id,
            metadata=self.metadata,
            crop_index=self.crop_index,
            context_slices=self.context_slices,
        )
        image_stack = [self._load_image_array(img_id) for img_id in image_ids]
        image_np = np.concatenate(image_stack, axis=2)

        if self.label_dir is not None:
            mask = self._load_mask_array(image_id)
        else:
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.float32)

        if self.transform is not None:
            transformed = self.transform(image=image_np, mask=mask)
            image_tensor = transformed["image"]
            mask_tensor = transformed["mask"]
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor, image_id

    def _resize_image(self, image: Image.Image) -> Image.Image:
        if image.size != (self.img_size, self.img_size):
            return image.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        return image

    def _resize_mask(self, mask: Image.Image) -> Image.Image:
        if mask.size != (self.img_size, self.img_size):
            return mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)
        return mask

    def _load_image_array(self, image_id: str) -> np.ndarray:
        image_path = self.image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            image_path = self.image_dir / f"{self.image_ids[0]}.jpg"
        image = Image.open(image_path).convert("RGB")
        image = self._resize_image(image)
        return np.array(image)

    def _load_mask_array(self, image_id: str) -> np.ndarray:
        if self.label_dir is None:
            raise ValueError("ラベルディレクトリが指定されていません。")
        label_path = self.label_dir / f"{image_id}.png"
        label = Image.open(label_path)
        label = self._resize_mask(label)
        return (np.array(label) == 2).astype(np.float32)


def get_transforms(train: bool, img_size: int, num_channels: int) -> Optional[object]:
    """前処理パイプラインを生成する。

    Args:
        train: 学習用かどうか。
        img_size: 画像サイズ。
        num_channels: 入力チャンネル数。

    Returns:
        変換パイプライン。利用不可ならNone。
    """

    if not ALBUMENTATIONS_AVAILABLE:
        return None

    mean, std = build_normalization_stats(num_channels)

    if train:
        transforms: List[object] = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
            A.OneOf(
                [
                    A.ElasticTransform(alpha=50, sigma=6, p=1.0),
                    A.GridDistortion(p=1.0),
                ],
                p=0.3,
            ),
        ]
        if num_channels == 3:
            transforms.extend(
                [
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(p=1.0),
                            A.HueSaturationValue(p=1.0),
                            A.RandomGamma(p=1.0),
                            A.CLAHE(p=1.0),
                        ],
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.GaussNoise(p=1.0),
                            A.GaussianBlur(p=1.0),
                        ],
                        p=0.2,
                    ),
                ]
            )
        transforms.extend(
            [
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )
        return A.Compose(transforms)

    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


class DiceLoss(nn.Module):
    """Dice損失。

    Args:
        smooth: 平滑化項。
    """

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


class FocalLoss(nn.Module):
    """Focal損失。

    Args:
        alpha: クラス重み。
        gamma: 難易度係数。
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        prob = torch.sigmoid(pred)
        p_t = prob * target + (1 - prob) * (1 - target)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky損失。

    Args:
        alpha: 偽陽性の重み。
        beta: 偽陰性の重み。
        smooth: 平滑化項。
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class MixedLoss(nn.Module):
    """複数損失の加重和。

    Args:
        losses: (loss, weight) の一覧。
    """

    def __init__(self, losses: List[Tuple[nn.Module, float]]) -> None:
        super().__init__()
        self.losses = losses

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for loss_fn, weight in self.losses:
            total += weight * loss_fn(pred, target)
        return total


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    """Diceスコアを計算する。

    Args:
        pred: 予測ロジット。
        target: 正解マスク。
        threshold: 二値化しきい値。

    Returns:
        Diceスコア。
    """

    pred_bin = (torch.sigmoid(pred) > threshold).float()
    pred_sum = pred_bin.sum()
    target_sum = target.sum()
    if pred_sum.item() == 0 and target_sum.item() == 0:
        return torch.tensor(1.0, device=pred.device)
    intersection = (pred_bin * target).sum()
    return (2.0 * intersection) / (pred_sum + target_sum + 1e-8)


def resize_logits_like(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ロジットをターゲットの空間サイズに合わせる。

    Args:
        logits: 予測ロジット。
        target: 参照テンソル。

    Returns:
        リサイズ済みロジット。
    """

    if logits.shape[-2:] == target.shape[-2:]:
        return logits
    return F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)


def rle_encode(mask: np.ndarray) -> str:
    """RLE形式に変換する。

    Args:
        mask: 二値マスク。

    Returns:
        RLE文字列。
    """

    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def set_seed(seed: int, debug: bool) -> None:
    """乱数種を固定する。

    Args:
        seed: 乱数シード。
        debug: デバッグモード。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if debug:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def load_metadata(csv_path: Path) -> Dict[str, Dict[str, int]]:
    """メタデータを読み込む。

    Args:
        csv_path: CSVパス。

    Returns:
        image_id -> {"crop_id": int, "slice_id": int} の辞書。
    """

    if not csv_path.exists():
        return {}
    metadata: Dict[str, Dict[str, int]] = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("image_id")
            crop_id = row.get("crop_id")
            slice_id = row.get("slice_id")
            if not image_id or crop_id is None or slice_id is None:
                continue
            try:
                metadata[image_id] = {
                    "crop_id": int(crop_id),
                    "slice_id": int(slice_id),
                }
            except ValueError:
                continue
    return metadata


def build_crop_index(metadata: Dict[str, Dict[str, int]]) -> Dict[int, Dict[int, str]]:
    """crop_id -> slice_id -> image_id の辞書を作る。

    Args:
        metadata: メタデータ。

    Returns:
        crop_id辞書。
    """

    crop_index: Dict[int, Dict[int, str]] = {}
    for image_id, info in metadata.items():
        crop_id = info.get("crop_id")
        slice_id = info.get("slice_id")
        if crop_id is None or slice_id is None:
            continue
        crop_index.setdefault(crop_id, {})[slice_id] = image_id
    return crop_index


def resolve_context_ids(
    image_id: str,
    metadata: Dict[str, Dict[str, int]],
    crop_index: Dict[int, Dict[int, str]],
    context_slices: int,
) -> List[str]:
    """前後スライスのIDを解決する。

    Args:
        image_id: 中心画像ID。
        metadata: メタデータ。
        crop_index: crop_id辞書。
        context_slices: 前後スライス数。

    Returns:
        スタックする画像ID一覧。
    """

    if context_slices <= 0:
        return [image_id]

    info = metadata.get(image_id)
    if info is None:
        return [image_id] * (context_slices * 2 + 1)

    crop_id = info["crop_id"]
    slice_id = info["slice_id"]
    stack_ids: List[str] = []
    for offset in range(-context_slices, context_slices + 1):
        target_slice = slice_id + offset
        target_id = crop_index.get(crop_id, {}).get(target_slice, image_id)
        stack_ids.append(target_id)
    return stack_ids


def build_normalization_stats(num_channels: int) -> Tuple[List[float], List[float]]:
    """正規化の平均と分散を作る。

    Args:
        num_channels: チャンネル数。

    Returns:
        (mean, std) のリスト。
    """

    base_mean = [0.485, 0.456, 0.406]
    base_std = [0.229, 0.224, 0.225]
    repeats = max(1, (num_channels + 2) // 3)
    mean = (base_mean * repeats)[:num_channels]
    std = (base_std * repeats)[:num_channels]
    return mean, std


def normalize_encoder_name(encoder: str) -> str:
    """エンコーダ名を正規化する。

    Args:
        encoder: 入力エンコーダ名。

    Returns:
        正規化後のエンコーダ名。
    """

    if encoder.startswith(("tu-", "timm-")):
        return encoder
    convnext_aliases = {
        "convnext_tiny": "tu-convnext_tiny",
        "convnext_small": "tu-convnext_small",
        "convnext_base": "tu-convnext_base",
        "convnext_large": "tu-convnext_large",
    }
    if encoder in convnext_aliases:
        return convnext_aliases[encoder]
    if encoder.startswith("convnextv2_"):
        return f"tu-{encoder}"
    if encoder.startswith("convnextv3_"):
        return f"tu-{encoder}"
    return encoder


def is_timm_model_available(name: str) -> bool:
    """timmモデルの存在を確認する。

    Args:
        name: timmモデル名。

    Returns:
        利用可能ならTrue。
    """

    try:
        import timm
    except Exception:
        return False
    return name in timm.list_models(name)


class FilteredEncoder(nn.Module):
    """特徴マップから0チャンネルを除外するラッパー。

    Args:
        encoder: 元のエンコーダ。
        keep_indices: 残すインデックス。
    """

    def __init__(self, encoder: nn.Module, keep_indices: List[int]) -> None:
        super().__init__()
        self.encoder = encoder
        self.keep_indices = keep_indices
        self.out_channels = [encoder.out_channels[i] for i in keep_indices]
        self.output_stride = getattr(encoder, "output_stride", 32)
        self.name = getattr(encoder, "name", encoder.__class__.__name__)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = self.encoder(x)
        return [features[i] for i in self.keep_indices]


class SimpleSegmentationModel(smp_base_model.SegmentationModel):
    """簡易セグメンテーションモデル。

    Args:
        encoder: エンコーダ。
        decoder: デコーダ。
        segmentation_head: セグメンテーションヘッド。
        name: モデル名。
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        segmentation_head: nn.Module,
        name: str,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.segmentation_head = segmentation_head
        self.classification_head = None
        self.name = name
        self.initialize()


def filter_zero_channel_encoder(encoder: nn.Module) -> nn.Module:
    """0チャンネルの特徴を除外したエンコーダにする。

    Args:
        encoder: 元のエンコーダ。

    Returns:
        フィルタ済みエンコーダ。
    """

    out_channels = list(getattr(encoder, "out_channels", []))
    keep_indices = [idx for idx, ch in enumerate(out_channels) if ch > 0]
    if len(keep_indices) == len(out_channels):
        return encoder
    return FilteredEncoder(encoder, keep_indices)


def build_decoder_channels(depth: int) -> Tuple[int, ...]:
    """デコーダチャンネルを生成する。

    Args:
        depth: デコーダブロック数。

    Returns:
        デコーダチャンネル。
    """

    base = (256, 128, 64, 32, 16)
    if depth <= 0:
        raise ValueError("decoder depth が不正です。")
    return tuple(base[:depth])


def build_custom_model(
    arch: str,
    encoder_name: str,
    encoder_weights: Optional[str],
    in_channels: int,
) -> nn.Module:
    """カスタムでモデルを構築する。

    Args:
        arch: アーキテクチャ名。
        encoder_name: エンコーダ名。
        encoder_weights: エンコーダ重み。
        in_channels: 入力チャンネル数。

    Returns:
        モデル。
    """

    encoder = get_encoder(
        encoder_name,
        in_channels=in_channels,
        depth=5,
        weights=encoder_weights,
    )
    encoder = filter_zero_channel_encoder(encoder)
    depth = len(encoder.out_channels) - 1
    decoder_channels = build_decoder_channels(depth)

    if arch == "unetplusplus":
        decoder = UnetPlusPlusDecoder(
            encoder_channels=encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=depth,
            use_norm="batchnorm",
            center=encoder_name.startswith("vgg"),
            attention_type=None,
            interpolation_mode="nearest",
        )
    elif arch == "unet":
        decoder = UnetDecoder(
            encoder_channels=encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=depth,
            use_norm="batchnorm",
            attention_type=None,
            add_center_block=encoder_name.startswith("vgg"),
            interpolation_mode="nearest",
        )
    else:
        raise ValueError("このアーキテクチャはConvNeXtに未対応です。")

    segmentation_head = SegmentationHead(
        in_channels=decoder_channels[-1],
        out_channels=1,
        activation=None,
        kernel_size=3,
    )
    model_name = f"{arch}-{encoder_name}"
    return SimpleSegmentationModel(encoder, decoder, segmentation_head, model_name)


def compute_sample_weights(
    image_ids: List[str],
    label_dir: Path,
    img_size: int,
    pos_boost: float,
) -> List[float]:
    """サンプル重みを計算する。

    Args:
        image_ids: 画像ID一覧。
        label_dir: ラベルディレクトリ。
        img_size: 画像サイズ。
        pos_boost: 正例サンプルの重み強化。

    Returns:
        重みリスト。
    """

    weights: List[float] = []
    for image_id in image_ids:
        label_path = label_dir / f"{image_id}.png"
        if not label_path.exists():
            weights.append(1.0)
            continue
        label = Image.open(label_path)
        if label.size != (img_size, img_size):
            label = label.resize((img_size, img_size), resample=Image.NEAREST)
        mask = np.array(label) == 2
        has_pos = float(mask.any())
        weights.append(1.0 + pos_boost * has_pos)
    return weights


def split_by_crop_id(
    image_ids: List[str], metadata: Dict[str, Dict[str, int]], val_ratio: float, seed: int
) -> Tuple[List[str], List[str]]:
    """crop_id単位で分割する。

    Args:
        image_ids: 画像ID一覧。
        metadata: メタデータ。
        val_ratio: 検証割合。
        seed: 乱数シード。

    Returns:
        (train_ids, val_ids)。
    """

    if not metadata:
        ids = image_ids[:]
        random.Random(seed).shuffle(ids)
        val_count = max(1, int(len(ids) * val_ratio))
        return ids[val_count:], ids[:val_count]

    groups: Dict[str, List[str]] = {}
    for image_id in image_ids:
        crop_info = metadata.get(image_id)
        if crop_info is None:
            group_key = f"unknown_{image_id}"
        else:
            group_key = str(crop_info["crop_id"])
        groups.setdefault(group_key, []).append(image_id)

    group_keys = list(groups.keys())
    random.Random(seed).shuffle(group_keys)
    target_val = max(1, int(len(image_ids) * val_ratio))

    val_ids: List[str] = []
    for key in group_keys:
        if len(val_ids) >= target_val:
            break
        val_ids.extend(groups[key])

    val_set = set(val_ids)
    train_ids = [image_id for image_id in image_ids if image_id not in val_set]
    return train_ids, val_ids


def build_loss(config: Config) -> nn.Module:
    """損失関数を構築する。

    Args:
        config: 設定。

    Returns:
        損失関数。
    """

    if config.loss_name == "dice_bce":
        return MixedLoss(
            [
                (DiceLoss(), config.dice_weight),
                (nn.BCEWithLogitsLoss(), 1 - config.dice_weight),
            ]
        )
    if config.loss_name == "dice_focal":
        return MixedLoss(
            [
                (DiceLoss(), config.dice_weight),
                (FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma), 1 - config.dice_weight),
            ]
        )
    if config.loss_name == "tversky":
        return TverskyLoss(alpha=config.tversky_alpha, beta=config.tversky_beta)
    if config.loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    raise ValueError("損失関数の指定が不正です。")


def resolve_model(config: Config, in_channels: int) -> nn.Module:
    """モデルを構築する。

    Args:
        config: 設定。
        in_channels: 入力チャンネル数。

    Returns:
        モデル。
    """

    if not SMP_AVAILABLE:
        raise ImportError("segmentation_models_pytorch が見つかりません")

    arch = config.arch.lower()
    encoder_name = normalize_encoder_name(config.encoder)
    available_encoders: List[str] = []
    if hasattr(smp, "encoders") and hasattr(smp.encoders, "get_encoder_names"):
        available_encoders = smp.encoders.get_encoder_names()
    if (
        available_encoders
        and encoder_name not in available_encoders
        and not encoder_name.startswith("tu-")
        and not encoder_name.startswith("timm-")
    ):
        fallback = "resnet34" if "resnet34" in available_encoders else available_encoders[0]
        print(
            "指定エンコーダが未対応のため、"
            f"{encoder_name} -> {fallback} に切り替えます。"
        )
        print("ConvNeXtを使う場合は、segmentation_models_pytorch と timm の更新が必要です。")
        encoder_name = fallback
    if arch == "unet":
        model_cls = smp.Unet
    elif arch == "unetplusplus":
        model_cls = smp.UnetPlusPlus
    elif arch == "efficientunetplusplus":
        if not hasattr(smp, "EfficientUnetPlusPlus"):
            raise ValueError("EfficientUnetPlusPlus が利用できません。")
        model_cls = smp.EfficientUnetPlusPlus
    elif arch == "deeplabv3plus":
        model_cls = smp.DeepLabV3Plus
    elif arch == "fpn":
        model_cls = smp.FPN
    else:
        raise ValueError("アーキテクチャの指定が不正です。")

    try:
        if encoder_name.startswith(("tu-", "timm-")):
            timm_name = encoder_name.replace("timm-", "").replace("tu-", "")
            if not is_timm_model_available(timm_name):
                fallback = "resnet34" if "resnet34" in available_encoders else available_encoders[0]
                print(
                    "指定timmモデルが見つからないため、"
                    f"{encoder_name} -> {fallback} に切り替えます。"
                )
                encoder_name = fallback
            else:
                try:
                    return build_custom_model(arch, encoder_name, config.encoder_weights, in_channels)
                except ValueError as exc:
                    fallback = "resnet34" if "resnet34" in available_encoders else available_encoders[0]
                    print(
                        "ConvNeXt互換モデルの構築に失敗したため、"
                        f"{encoder_name} -> {fallback} に切り替えます。"
                    )
                    encoder_name = fallback
        model = model_cls(
            encoder_name=encoder_name,
            encoder_weights=config.encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError("モデルの初期化に失敗しました。") from exc
    return model


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: Config, steps_per_epoch: int
) -> Tuple[Optional[torch.optim.lr_scheduler._LRScheduler], bool]:
    """学習率スケジューラを構築する。

    Args:
        optimizer: オプティマイザ。
        config: 設定。
        steps_per_epoch: 1エポックあたりの更新回数。

    Returns:
        (スケジューラ, バッチごとにstepするか)。
    """

    if config.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs), False
    if config.scheduler == "onecycle":
        return (
            OneCycleLR(
                optimizer,
                max_lr=config.lr,
                epochs=config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=config.onecycle_pct_start,
                div_factor=config.onecycle_div_factor,
                final_div_factor=config.onecycle_final_div_factor,
            ),
            True,
        )
    if config.scheduler == "none":
        return None, False
    raise ValueError("スケジューラの指定が不正です。")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    threshold: float,
    scaler: GradScaler,
    use_amp: bool,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step_per_batch: bool,
) -> Tuple[float, float]:
    """1エポック学習する。

    Args:
        model: モデル。
        loader: DataLoader。
        criterion: 損失関数。
        optimizer: オプティマイザ。
        device: デバイス。
        threshold: Diceしきい値。
        scaler: GradScaler。
        use_amp: AMPの有無。
        scheduler: スケジューラ。
        step_per_batch: バッチごとにstepするか。

    Returns:
        (loss, dice)。
    """

    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks, _ in tqdm(loader, desc="Train"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with autocast_context(use_amp):
            outputs = model(images)
            outputs = resize_logits_like(outputs, masks)
            loss = criterion(outputs, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None and step_per_batch:
            scheduler.step()

        total_loss += float(loss.item())
        total_dice += float(dice_score(outputs, masks, threshold).item())

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    thresholds: List[float],
    auto_threshold: bool,
    use_amp: bool,
) -> Tuple[float, float, float]:
    """検証を実行する。

    Args:
        model: モデル。
        loader: DataLoader。
        criterion: 損失関数。
        device: デバイス。
        threshold: Diceしきい値。
        thresholds: しきい値候補。
        auto_threshold: しきい値探索の有無。
        use_amp: AMPの有無。

    Returns:
        (loss, dice, best_threshold)。
    """

    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    best_threshold = threshold
    if auto_threshold:
        threshold_stats = {
            "inter": torch.zeros(len(thresholds), device=device),
            "pred": torch.zeros(len(thresholds), device=device),
            "target": torch.tensor(0.0, device=device),
        }

    for images, masks, _ in tqdm(loader, desc="Val"):
        images = images.to(device)
        masks = masks.to(device)
        with autocast_context(use_amp):
            outputs = model(images)
            outputs = resize_logits_like(outputs, masks)
            loss = criterion(outputs, masks)
        total_loss += float(loss.item())
        if auto_threshold:
            probs = torch.sigmoid(outputs)
            threshold_stats["target"] += masks.sum()
            for idx, value in enumerate(thresholds):
                pred = (probs > value).float()
                threshold_stats["inter"][idx] += (pred * masks).sum()
                threshold_stats["pred"][idx] += pred.sum()
        else:
            total_dice += float(dice_score(outputs, masks, threshold).item())

    n = len(loader)
    if auto_threshold:
        target_sum = threshold_stats["target"]
        dices = (
            2.0 * threshold_stats["inter"]
        ) / (threshold_stats["pred"] + target_sum + 1e-8)
        best_idx = int(torch.argmax(dices).item())
        best_threshold = thresholds[best_idx]
        total_dice = float(dices[best_idx].item())
    else:
        total_dice = total_dice / n
    return total_loss / n, total_dice, best_threshold


def apply_tta_ops(tensor: torch.Tensor, ops: List[str]) -> torch.Tensor:
    """TTAの操作列を適用する。

    Args:
        tensor: 入力テンソル。
        ops: 操作列。

    Returns:
        変換後テンソル。
    """

    out = tensor
    for op in ops:
        if op == "h":
            out = torch.flip(out, dims=[3])
        elif op == "v":
            out = torch.flip(out, dims=[2])
        elif op == "t":
            out = out.transpose(2, 3)
    return out


def tta_ops_from_mode(mode: Optional[str]) -> List[str]:
    """TTAモードから操作列を作る。

    Args:
        mode: モード文字列。

    Returns:
        操作列。
    """

    mapping = {
        None: [],
        "h": ["h"],
        "v": ["v"],
        "hv": ["h", "v"],
        "t": ["t"],
        "th": ["t", "h"],
        "tv": ["t", "v"],
        "thv": ["t", "h", "v"],
    }
    return mapping.get(mode, [])


@torch.no_grad()
def predict_probabilities(
    model: nn.Module, images: torch.Tensor, use_tta: bool, use_amp: bool
) -> torch.Tensor:
    """確率マップを推論する。

    Args:
        model: モデル。
        images: 入力バッチ。
        use_tta: TTAの有無。
        use_amp: AMPの有無。

    Returns:
        確率マップ。
    """

    if not use_tta:
        with autocast_context(use_amp):
            logits = model(images)
        return torch.sigmoid(logits)

    modes = [None, "h", "v", "hv", "t", "th", "tv", "thv"]
    probs = []
    for mode in modes:
        ops = tta_ops_from_mode(mode)
        augmented = apply_tta_ops(images, ops)
        with autocast_context(use_amp):
            logits = model(augmented)
        prob = torch.sigmoid(logits)
        prob = apply_tta_ops(prob, list(reversed(ops)))
        probs.append(prob)
    return torch.stack(probs).mean(0)


def postprocess_mask(mask: np.ndarray, min_area: int) -> np.ndarray:
    """小領域を除外する。

    Args:
        mask: 二値マスク。
        min_area: 最小面積。

    Returns:
        後処理済みマスク。
    """

    if min_area <= 0:
        return mask
    if mask.sum() < min_area:
        return np.zeros_like(mask)
    return mask


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    min_area: int,
    use_tta: bool,
    use_amp: bool,
) -> Dict[str, str]:
    """推論してRLEを生成する。

    Args:
        model: モデル。
        loader: DataLoader。
        device: デバイス。
        threshold: 二値化しきい値。
        min_area: 最小面積。
        use_tta: TTAの有無。
        use_amp: AMPの有無。

    Returns:
        image_id -> RLE の辞書。
    """

    model.eval()
    predictions: Dict[str, str] = {}

    for images, _, image_ids in tqdm(loader, desc="Predict"):
        images = images.to(device)
        probs = predict_probabilities(model, images, use_tta, use_amp)
        if probs.shape[-2:] != images.shape[-2:]:
            probs = F.interpolate(probs, size=images.shape[-2:], mode="bilinear", align_corners=False)
        preds = (probs > threshold).cpu().numpy()

        for pred, image_id in zip(preds, image_ids):
            mask = pred.squeeze().astype(np.uint8)
            mask = postprocess_mask(mask, min_area)
            rle = rle_encode(mask)
            predictions[image_id] = rle

    return predictions


def parse_args() -> Config:
    """引数を読み取って設定を返す。

    Returns:
        設定。
    """

    parser = argparse.ArgumentParser(description="Prostate Epithelium Segmentation Baseline")
    parser.add_argument("--input-dir", type=Path, default=Path("input"), help="入力データのルート")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="出力先ディレクトリ")
    parser.add_argument("--batch-size", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--epochs", type=int, default=10, help="学習エポック数")
    parser.add_argument("--lr", type=float, default=3e-4, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="weight decay")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.99, help="AdamW beta2")
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
    parser.add_argument("--val-ratio", type=float, default=0.2, help="検証データの割合")
    parser.add_argument(
        "--arch",
        type=str,
        default="unetplusplus",
        choices=["unet", "unetplusplus", "efficientunetplusplus", "deeplabv3plus", "fpn"],
        help="アーキテクチャ",
    )
    parser.add_argument("--encoder", type=str, default="tu-convnext_tiny", help="エンコーダ名")
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="エンコーダ重み (noneで無効)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="dice_bce",
        choices=["dice_bce", "dice_focal", "tversky", "bce"],
        help="損失関数",
    )
    parser.add_argument("--dice-weight", type=float, default=0.7, help="Diceの重み")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal alpha")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal gamma")
    parser.add_argument("--tversky-alpha", type=float, default=0.7, help="Tversky alpha")
    parser.add_argument("--tversky-beta", type=float, default=0.3, help="Tversky beta")
    parser.add_argument("--threshold", type=float, default=0.5, help="推論しきい値")
    parser.add_argument("--auto-threshold", action="store_true", help="検証でしきい値を探索する")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="しきい値候補 (例: 0.35 0.4 0.45)",
    )
    parser.add_argument("--threshold-min", type=float, default=0.3, help="しきい値探索の最小値")
    parser.add_argument("--threshold-max", type=float, default=0.7, help="しきい値探索の最大値")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="しきい値探索の刻み")
    parser.add_argument("--min-area", type=int, default=0, help="最小面積フィルタ")
    parser.add_argument("--tta", action="store_true", help="推論時TTAを使う")
    parser.add_argument("--amp", action="store_true", help="AMPを使う")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["cosine", "onecycle", "none"],
        help="学習率スケジューラ",
    )
    parser.add_argument("--onecycle-pct-start", type=float, default=0.1, help="OneCycleの上昇割合")
    parser.add_argument(
        "--onecycle-div-factor",
        type=float,
        default=25.0,
        help="OneCycleの初期lrスケール",
    )
    parser.add_argument(
        "--onecycle-final-div-factor",
        type=float,
        default=1e4,
        help="OneCycleの最終lrスケール",
    )
    parser.add_argument("--pin-memory", action="store_true", help="pin_memoryを有効化")
    parser.add_argument(
        "--context-slices",
        type=int,
        default=1,
        help="前後スライス数 (0で2D)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="none",
        choices=["none", "weighted"],
        help="学習サンプラー",
    )
    parser.add_argument("--pos-boost", type=float, default=2.0, help="正例サンプルの重み強化")

    args = parser.parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("val_ratio は 0〜1 の範囲で指定してください。")
    if args.context_slices < 0:
        raise ValueError("context_slices は 0 以上で指定してください。")
    if args.threshold_min >= args.threshold_max:
        raise ValueError("threshold-min は threshold-max より小さくしてください。")
    if args.threshold_step <= 0:
        raise ValueError("threshold-step は正の値で指定してください。")

    thresholds = args.thresholds
    if thresholds is None:
        thresholds = []
        value = args.threshold_min
        while value <= args.threshold_max + 1e-8:
            thresholds.append(round(value, 4))
            value += args.threshold_step
    if not thresholds:
        raise ValueError("thresholds が空です。")

    encoder_weights = None if args.encoder_weights == "none" else args.encoder_weights
    return Config(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        num_workers=args.num_workers,
        img_size=args.img_size,
        device=args.device,
        seed=args.seed,
        debug=args.debug,
        val_ratio=args.val_ratio,
        arch=args.arch,
        encoder=args.encoder,
        encoder_weights=encoder_weights,
        loss_name=args.loss,
        dice_weight=args.dice_weight,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        threshold=args.threshold,
        min_area=args.min_area,
        tta=args.tta,
        amp=args.amp,
        scheduler=args.scheduler,
        pin_memory=args.pin_memory,
        context_slices=args.context_slices,
        auto_threshold=args.auto_threshold,
        thresholds=thresholds,
        sampler=args.sampler,
        pos_boost=args.pos_boost,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
    )


def save_submission(predictions: Dict[str, str], output_path: Path) -> None:
    """提出ファイルを保存する。

    Args:
        predictions: 予測辞書。
        output_path: 出力先パス。
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Expected"])
        for image_id in sorted(predictions.keys()):
            writer.writerow([image_id, predictions[image_id]])


def load_best_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    """最良モデルの重みを読み込む。

    Args:
        model: モデル。
        path: 重みパス。
        device: デバイス。
    """

    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def resolve_data_root(input_dir: Path) -> Path:
    """データルートを決定する。

    Args:
        input_dir: 入力ディレクトリ。

    Returns:
        データルート。
    """

    if (input_dir / "train").exists():
        return input_dir
    return input_dir / "data"


def list_image_ids(image_dir: Path) -> List[str]:
    """画像ID一覧を作成する。

    Args:
        image_dir: 画像ディレクトリ。

    Returns:
        画像ID一覧。
    """

    return sorted([f.stem for f in image_dir.glob("*.jpg")])


def limit_ids(image_ids: List[str], limit: int) -> List[str]:
    """ID一覧を制限する。

    Args:
        image_ids: 画像ID一覧。
        limit: 上限数。

    Returns:
        制限後の一覧。
    """

    return image_ids[: min(limit, len(image_ids))]


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
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            num_workers=0,
            img_size=config.img_size,
            device=config.device,
            seed=config.seed,
            debug=config.debug,
            val_ratio=config.val_ratio,
            arch=config.arch,
            encoder=config.encoder,
            encoder_weights=config.encoder_weights,
            loss_name=config.loss_name,
            dice_weight=config.dice_weight,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            tversky_alpha=config.tversky_alpha,
            tversky_beta=config.tversky_beta,
            threshold=config.threshold,
            min_area=config.min_area,
            tta=config.tta,
            amp=config.amp,
            scheduler=config.scheduler,
            pin_memory=False,
            context_slices=config.context_slices,
            auto_threshold=config.auto_threshold,
            thresholds=config.thresholds,
            sampler=config.sampler,
            pos_boost=config.pos_boost,
            onecycle_pct_start=config.onecycle_pct_start,
            onecycle_div_factor=config.onecycle_div_factor,
            onecycle_final_div_factor=config.onecycle_final_div_factor,
        )

    data_dir = resolve_data_root(config.input_dir)
    num_channels = 3 * (config.context_slices * 2 + 1)
    train_transform = get_transforms(train=True, img_size=config.img_size, num_channels=num_channels)
    val_transform = get_transforms(train=False, img_size=config.img_size, num_channels=num_channels)

    if train_transform is None or val_transform is None:
        print("albumentations が見つからないため、基本変換のみを使用します。")

    train_csv = data_dir / "train" / "train.csv"
    train_meta = load_metadata(train_csv)
    image_dir = data_dir / "train" / "images"
    label_dir = data_dir / "train" / "labels"
    image_ids = list_image_ids(image_dir)
    train_ids, val_ids = split_by_crop_id(image_ids, train_meta, config.val_ratio, config.seed)

    if config.debug:
        train_ids = limit_ids(train_ids, 64)
        val_ids = limit_ids(val_ids, 32)

    train_dataset = SegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_ids=train_ids,
        transform=train_transform,
        img_size=config.img_size,
        metadata=train_meta,
        context_slices=config.context_slices,
    )
    val_dataset = SegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_ids=val_ids,
        transform=val_transform,
        img_size=config.img_size,
        metadata=train_meta,
        context_slices=config.context_slices,
    )

    sampler = None
    shuffle = True
    if config.sampler == "weighted":
        weights = compute_sample_weights(train_ids, label_dir, config.img_size, config.pos_boost)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = resolve_model(config, in_channels=num_channels).to(device)
    criterion = build_loss(config)
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    scheduler, step_per_batch = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    scaler = GradScaler(enabled=config.amp)

    best_dice = -1.0
    best_threshold = config.threshold
    current_threshold = config.threshold
    best_path = config.output_dir / "best_model.pth"
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")

        train_loss, train_dice = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            current_threshold,
            scaler,
            config.amp,
            scheduler,
            step_per_batch,
        )
        val_loss, val_dice, epoch_threshold = validate(
            model,
            val_loader,
            criterion,
            device,
            current_threshold,
            config.thresholds,
            config.auto_threshold,
            config.amp,
        )
        if config.auto_threshold:
            current_threshold = epoch_threshold
        if scheduler is not None and not step_per_batch:
            scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        if config.auto_threshold:
            print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, Th: {epoch_threshold:.3f}")
        else:
            print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            best_threshold = epoch_threshold
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (Dice: {best_dice:.4f})")

    print("\nGenerating predictions...")
    if best_path.exists():
        load_best_weights(model, best_path, device)
    if config.auto_threshold:
        print(f"Best threshold: {best_threshold:.3f}")

    test_image_dir = data_dir / "test" / "images"
    test_csv = data_dir / "test" / "test.csv"
    test_meta = load_metadata(test_csv)
    test_ids = list_image_ids(test_image_dir)
    if config.debug:
        test_ids = limit_ids(test_ids, 32)
    test_dataset = SegmentationDataset(
        image_dir=test_image_dir,
        label_dir=None,
        image_ids=test_ids,
        transform=val_transform,
        img_size=config.img_size,
        metadata=test_meta,
        context_slices=config.context_slices,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    predict_threshold = best_threshold if config.auto_threshold else config.threshold
    predictions = predict(
        model,
        test_loader,
        device,
        predict_threshold,
        config.min_area,
        config.tta,
        config.amp,
    )

    submission_path = config.output_dir / "submission.csv"
    save_submission(predictions, submission_path)

    print(f"\nSubmission saved to {submission_path}")
    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
