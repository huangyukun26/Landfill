from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def compute_metrics(pred_dir: Path) -> dict[str, object]:
    conf = np.zeros((2, 2), dtype=np.int64)
    count = 0

    for gt_path in sorted(pred_dir.glob("*_gt.png")):
        base = gt_path.name[:-7]
        pred_path = pred_dir / f"{base}_pred.png"
        if not pred_path.exists():
            continue

        gt = (np.array(Image.open(gt_path)) > 127).astype(np.uint8)
        pred = (np.array(Image.open(pred_path)) > 127).astype(np.uint8)
        if gt.shape != pred.shape:
            raise ValueError(f"shape mismatch for {gt_path.name}: {gt.shape} vs {pred.shape}")

        idx = gt.reshape(-1) * 2 + pred.reshape(-1)
        conf += np.bincount(idx, minlength=4).reshape(2, 2)
        count += 1

    tp = conf[1, 1]
    tn = conf[0, 0]
    fp = conf[0, 1]
    fn = conf[1, 0]
    total = conf.sum()

    oa = (tp + tn) / total
    acc0 = tn / max(conf[0].sum(), 1)
    acc1 = tp / max(conf[1].sum(), 1)
    miou0 = tn / max(tn + fp + fn, 1)
    miou1 = tp / max(tp + fp + fn, 1)
    freq0 = conf[0].sum() / total
    freq1 = conf[1].sum() / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "confusion": conf.tolist(),
        "samples": count,
        "OA": oa,
        "mAcc": (acc0 + acc1) / 2,
        "mIoU": (miou0 + miou1) / 2,
        "fwIoU": freq0 * miou0 + freq1 * miou1,
        "IoU_non_landfill": miou0,
        "IoU_landfill": miou1,
        "precision_landfill": precision,
        "recall_landfill": recall,
        "f1_landfill": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    metrics = compute_metrics(args.pred_dir)
    lines = [f"{key}={value}" for key, value in metrics.items()]
    text = "\n".join(lines) + "\n"

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    print(text, end="")


if __name__ == "__main__":
    main()
