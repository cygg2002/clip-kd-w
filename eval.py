#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import open_clip
from open_clip import create_model_and_transforms

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
# ===================== Dataset：基于 CSV 读取图片 =====================

class CsvImageDataset(Dataset):
    def __init__(self, csv_path: str, preprocess):
        """
        CSV 格式示例:
        image_path,text
        /path/to/img.jpg,a photo of a Sylvicapra grimmia
        """
        df = pd.read_csv(csv_path)

        # 兼容列名里可能有空格的情况
        df.columns = [c.strip() for c in df.columns]

        if "image_path" not in df.columns or "text" not in df.columns:
            raise ValueError(f"CSV 必须包含 'image_path' 和 'text' 两列，当前列为: {df.columns.tolist()}")

        self.image_paths = df["image_path"].astype(str).tolist()
        # text 列本身就类似 "a photo of a Sylvicapra grimmia"
        self.texts = df["text"].astype(str).tolist()
        self.preprocess = preprocess

        # 所有不重复的 text 当一个类别
        self.class_texts = sorted(set(self.texts))          # C 个类别
        self.text_to_class = {t: i for i, t in enumerate(self.class_texts)}

        # 每张图的标签：它那一行的 text 在 class_texts 中的 index
        self.labels = np.array([self.text_to_class[t] for t in self.texts], dtype=np.int64)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = int(self.labels[idx])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.preprocess(img)

        return img, label


# ===================== 评估函数 =====================

@torch.no_grad()
def evaluate_csv_classification(
    model: torch.nn.Module,
    tokenizer,
    csv_path: str,
    preprocess,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
    topk=(1, 5),
):
    model.eval()

    # 1. 构建 Dataset / DataLoader
    dataset = CsvImageDataset(csv_path, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # 2. 编码所有类别文本，一次性算出 class_text_features
    class_texts = dataset.class_texts  # List[str]，长度 = C
    print(f"[INFO] 从 CSV 中发现 {len(class_texts)} 个不同的 text（按此作为类别）。")

    text_tokens = tokenizer(class_texts).to(device)  # [C, L]

    text_features = model.encode_text(text_tokens)   # [C, D]
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    num_classes = text_features.size(0)
    # topk 不能超过类别数，做个裁剪
    topk = tuple(k for k in topk if k <= num_classes)
    if len(topk) == 0:
        raise ValueError(f"topk 全都大于类别数 {num_classes}，请调小 topk。")

    # 3. 遍历所有图片，计算 image_features 和 logits
    total = 0
    correct_topk = {k: 0 for k in topk}

    # logit_scale：有则用，没有就当 1.0
    if hasattr(model, "logit_scale"):
        logit_scale = model.logit_scale.exp()
    else:
        logit_scale = torch.tensor(1.0, device=device)

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)  # [B]

        # [B, D]
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # [B, C]
        logits = logit_scale * (image_features @ text_features.t())

        max_k = max(topk)
        # pred: [B, max_k]，每行是 top-k 类别 id
        _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)

        # labels: [B] -> [B, 1]
        labels_view = labels.view(-1, 1).expand_as(pred)  # [B, max_k]
        correct = (pred == labels_view)                   # [B, max_k]

        for k in topk:
            # 前 k 里面只要有一个对的就算命中
            correct_k = correct[:, :k].any(dim=1).sum().item()
            correct_topk[k] += correct_k

        total += labels.size(0)

    metrics = {}
    for k in topk:
        acc = correct_topk[k] / total
        metrics[f"top{k}_acc"] = acc

    return metrics


# ===================== 主函数：加载模型 + checkpoint + 调用评估 =====================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIP model on CSV (image_path, text)")

    parser.add_argument(
        "--csv",
        type=str,
        default="/home/111_wcy/work/clip-kd/CLIP-KD-main/data/mammal_images/val.csv",
        help="val.csv 路径，包含 image_path 和 text 两列",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-B-32",
        help="open_clip 模型名，例如 'ViT-B-32'",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="open_clip 预训练标识（与你现在 create_model_and_transforms 一致），一般蒸馏用 None",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/111_wcy/work/clip-kd/distillation/CLIP-KD-main/logs/bioclip2_to_clipb32_fd_icl/checkpoints/epoch_12.pt",
        help="蒸馏后模型的 checkpoint 路径（.pt）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备，例如 'cuda' 或 'cpu'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="评估 batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="DataLoader num_workers",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 1. 创建模型和预处理（跟你现在训练/加载的方式保持一致）
    print(f"[INFO] 创建模型: {args.model_name}, pretrained={args.pretrained}")
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    model = model.to(device)

    # 2. 加载 checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    print(f"[INFO] 加载 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] load_state_dict done. missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print("  Missing keys (showing first 5):", missing[:5])
    if unexpected:
        print("  Unexpected keys (showing first 5):", unexpected[:5])

    model.eval()

    # 3. tokenizer（来自 open_clip）
    tokenizer = open_clip.get_tokenizer(args.model_name)

    # 4. 调用评估
    metrics = evaluate_csv_classification(
        model=model,
        tokenizer=tokenizer,
        csv_path=args.csv,
        preprocess=preprocess_val,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=(1, 5),
    )

    # 5. 打印结果
    print("\n========== Classification Results ==========")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("============================================")


if __name__ == "__main__":
    main()