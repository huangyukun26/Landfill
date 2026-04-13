from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path: Path) -> np.ndarray:
    return (np.array(Image.open(path)) > 127).astype(np.uint8)


def per_image_rows(pred_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gt_path in sorted(pred_dir.glob("*_gt.png")):
        sample = gt_path.name[:-7]
        pred_path = pred_dir / f"{sample}_pred.png"
        img_path = pred_dir / f"{sample}_img.png"
        if not pred_path.exists():
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        tp = int(((gt == 1) & (pred == 1)).sum())
        tn = int(((gt == 0) & (pred == 0)).sum())
        fp = int(((gt == 0) & (pred == 1)).sum())
        fn = int(((gt == 1) & (pred == 0)).sum())
        gt_area = int(gt.sum())
        pred_area = int(pred.sum())
        union = tp + fp + fn

        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        iou = tp / union if union else 1.0

        rows.append(
            {
                "sample": sample,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "gt_area": gt_area,
                "pred_area": pred_area,
                "gt_ratio": gt_area / gt.size,
                "pred_ratio": pred_area / pred.size,
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "fp_ratio": fp / max((gt == 0).sum(), 1),
                "fn_ratio": fn / max(gt_area, 1),
                "img_file": img_path.name,
                "gt_file": gt_path.name,
                "pred_file": pred_path.name,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    if not rows:
        raise ValueError("no *_gt.png / *_pred.png pairs found")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]], top_k: int) -> None:
    rows_with_gt = [r for r in rows if int(r["gt_area"]) > 0]
    severe_miss = [r for r in rows_with_gt if float(r["recall"]) < 0.3]
    severe_fp = [r for r in rows_with_gt if float(r["precision"]) < 0.4]
    zero_recall = [r for r in rows_with_gt if float(r["recall"]) == 0.0]
    wrong_region = [r for r in zero_recall if int(r["pred_area"]) > 0]
    empty_pred = [r for r in zero_recall if int(r["pred_area"]) == 0]

    print(f"samples_with_gt={len(rows_with_gt)}")
    print(f"severe_miss={len(severe_miss)}")
    print(f"severe_fp={len(severe_fp)}")
    print(f"zero_recall={len(zero_recall)}")
    print(f"zero_recall_empty_pred={len(empty_pred)}")
    print(f"zero_recall_wrong_region={len(wrong_region)}")

    print("worst_recall")
    for r in sorted(rows_with_gt, key=lambda x: (float(x["recall"]), int(x["gt_area"])))[:top_k]:
        print(
            f'{r["sample"]} gt_area={r["gt_area"]} '
            f'precision={float(r["precision"]):.4f} recall={float(r["recall"]):.4f} '
            f'iou={float(r["iou"]):.4f} pred_area={r["pred_area"]}'
        )

    print("worst_precision")
    for r in sorted(rows_with_gt, key=lambda x: (float(x["precision"]), -int(x["pred_area"])))[:top_k]:
        print(
            f'{r["sample"]} gt_area={r["gt_area"]} '
            f'precision={float(r["precision"]):.4f} recall={float(r["recall"]):.4f} '
            f'iou={float(r["iou"]):.4f} pred_area={r["pred_area"]}'
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    rows = per_image_rows(args.pred_dir)
    write_csv(rows, args.output_csv)
    print_summary(rows, args.top_k)


if __name__ == "__main__":
    main()
