from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
import torch
from PIL import Image
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ESD_REPO = ROOT / "_external" / "ESD"
if str(ESD_REPO) not in sys.path:
    sys.path.insert(0, str(ESD_REPO))

from esd_quantizer import Quantizer  # noqa: E402


@dataclass
class RasterInfo:
    width: int
    height: int
    x0: float
    y0: float
    px: float
    py: float
    epsg: int


def parse_epsg(page: tifffile.TiffPage) -> int:
    geo = page.tags.get("GeoKeyDirectoryTag")
    if geo is None:
        raise ValueError("GeoKeyDirectoryTag missing")
    vals = tuple(geo.value)
    for i in range(4, len(vals), 4):
        key, _, _, value = vals[i : i + 4]
        if key == 3072:
            return int(value)
    raise ValueError("Projected EPSG code not found")


def raster_info(path: Path) -> RasterInfo:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        scale = page.tags["ModelPixelScaleTag"].value
        tie = page.tags["ModelTiepointTag"].value
        return RasterInfo(
            width=int(page.imagewidth),
            height=int(page.imagelength),
            x0=float(tie[3]),
            y0=float(tie[4]),
            px=float(scale[0]),
            py=float(scale[1]),
            epsg=parse_epsg(page),
        )


def bounds(info: RasterInfo) -> tuple[float, float, float, float]:
    minx = info.x0
    maxx = info.x0 + info.width * info.px
    maxy = info.y0
    miny = info.y0 - info.height * info.py
    return minx, miny, maxx, maxy


def sample_tile_window(img_info: RasterInfo, esd_info: RasterInfo) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = bounds(img_info)
    transformer = Transformer.from_crs(img_info.epsg, esd_info.epsg, always_xy=True)
    corners = [
        transformer.transform(minx, miny),
        transformer.transform(minx, maxy),
        transformer.transform(maxx, miny),
        transformer.transform(maxx, maxy),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    tx_min, tx_max = min(xs), max(xs)
    ty_min, ty_max = min(ys), max(ys)

    col0 = max(0, int(math.floor((tx_min - esd_info.x0) / esd_info.px)))
    col1 = min(esd_info.width, int(math.ceil((tx_max - esd_info.x0) / esd_info.px)))
    row0 = max(0, int(math.floor((esd_info.y0 - ty_max) / esd_info.py)))
    row1 = min(esd_info.height, int(math.ceil((esd_info.y0 - ty_min) / esd_info.py)))
    return row0, row1, col0, col1


def decode_esd_vectors(codes: np.ndarray, quantizer: Quantizer) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(codes[:12].astype(np.int32))
        vectors = quantizer.indices_to_codes(tensor).numpy()
    return vectors.astype(np.float32)  # months, h, w, channels


def resize_mask(mask_path: Path, width: int, height: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    arr = np.asarray(mask.resize((width, height), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return arr


def flatten_features(vectors: np.ndarray) -> np.ndarray:
    months, height, width, channels = vectors.shape
    return vectors.transpose(1, 2, 0, 3).reshape(height * width, months * channels)


def subset_rows(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_threshold_for_target_recall(y_true: np.ndarray, probs: np.ndarray, target_recall: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return 1.0
    order = np.argsort(-probs)
    y_sorted = y_true[order]
    cum_tp = np.cumsum(y_sorted)
    recall = cum_tp / positives
    idx = int(np.argmax(recall >= target_recall))
    return float(probs[order[idx]])


def evaluate_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (probs >= threshold).astype(np.uint8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    retained_fraction = float(y_pred.mean())

    image_hits = []
    image_retained = []
    for gid in np.unique(groups):
        mask = groups == gid
        image_y = y_true[mask]
        image_pred = y_pred[mask]
        image_hits.append(float(((image_y == 1) & (image_pred == 1)).any()))
        image_retained.append(float(image_pred.mean()))

    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "retained_fraction": retained_fraction,
        "image_hit_recall": float(np.mean(image_hits)),
        "image_retained_fraction_mean": float(np.mean(image_retained)),
    }


def evaluate_keep_fraction(
    y_true: np.ndarray,
    probs: np.ndarray,
    groups: np.ndarray,
    keep_fraction: float,
) -> dict[str, float]:
    keep_n = max(1, int(len(probs) * keep_fraction))
    order = np.argsort(-probs)
    y_pred = np.zeros_like(y_true, dtype=np.uint8)
    y_pred[order[:keep_n]] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    image_hits = []
    image_retained = []
    for gid in np.unique(groups):
        mask = groups == gid
        image_y = y_true[mask]
        image_pred = y_pred[mask]
        image_hits.append(float(((image_y == 1) & (image_pred == 1)).any()))
        image_retained.append(float(image_pred.mean()))

    return {
        "keep_fraction": keep_fraction,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "image_hit_recall": float(np.mean(image_hits)),
        "image_retained_fraction_mean": float(np.mean(image_retained)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-json", type=Path, required=True)
    parser.add_argument("--esd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--positive-threshold", type=float, default=0.10)
    parser.add_argument("--train-neg-ratio", type=float, default=5.0)
    args = parser.parse_args()

    quantizer = Quantizer()
    rows = subset_rows(args.subset_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    feature_rows = []

    for sample in rows:
        split = sample["split"]
        stem = sample["stem"]
        tile = sample["tile"]
        img_path = ROOT / "img" / split / f"{stem}.tiff"
        mask_path = ROOT / "ann" / split / f"{stem}.png"
        esd_path = args.esd_dir / f"SDC30_EBD_V001_{tile}_2024.tif"
        if not esd_path.exists():
            continue

        img_info = raster_info(img_path)
        esd_info = raster_info(esd_path)
        row0, row1, col0, col1 = sample_tile_window(img_info, esd_info)
        if row1 <= row0 or col1 <= col0:
            continue

        codes = tifffile.imread(esd_path)[:, row0:row1, col0:col1]
        vectors = decode_esd_vectors(codes, quantizer)
        mask_resized = resize_mask(mask_path, vectors.shape[2], vectors.shape[1])
        valid = ~np.all(codes[:12] == 0, axis=0)
        labels = (mask_resized >= args.positive_threshold).astype(np.uint8)

        features = flatten_features(vectors)
        labels_flat = labels.reshape(-1)
        valid_flat = valid.reshape(-1)

        feature_rows.append(
            {
                "split": split,
                "stem": stem,
                "tile": tile,
                "height": int(vectors.shape[1]),
                "width": int(vectors.shape[2]),
                "features": features[valid_flat],
                "labels": labels_flat[valid_flat],
            }
        )
        metadata_rows.append(
            {
                "split": split,
                "stem": stem,
                "tile": tile,
                "window": [row0, row1, col0, col1],
                "height": int(vectors.shape[1]),
                "width": int(vectors.shape[2]),
                "valid_cells": int(valid_flat.sum()),
                "positive_cells": int(labels_flat[valid_flat].sum()),
            }
        )

    (args.output_dir / "pilot_samples.json").write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")

    train_feats, train_labels = [], []
    val_feats, val_labels, val_groups = [], [], []
    test_feats, test_labels, test_groups = [], [], []

    for idx, item in enumerate(feature_rows):
        split = item["split"]
        x = item["features"]
        y = item["labels"]
        if split == "train":
            pos = x[y == 1]
            neg = x[y == 0]
            max_neg = int(len(pos) * args.train_neg_ratio) if len(pos) else len(neg)
            if len(neg) > max_neg > 0:
                rng = np.random.default_rng(42 + idx)
                neg = neg[rng.choice(len(neg), size=max_neg, replace=False)]
            train_feats.append(np.concatenate([pos, neg], axis=0))
            train_labels.append(np.concatenate([np.ones(len(pos), dtype=np.uint8), np.zeros(len(neg), dtype=np.uint8)]))
        elif split == "val":
            val_feats.append(x)
            val_labels.append(y)
            val_groups.append(np.full(len(y), idx, dtype=np.int32))
        elif split == "test":
            test_feats.append(x)
            test_labels.append(y)
            test_groups.append(np.full(len(y), idx, dtype=np.int32))

    X_train = np.concatenate(train_feats, axis=0).astype(np.float32)
    y_train = np.concatenate(train_labels, axis=0).astype(np.uint8)
    X_val = np.concatenate(val_feats, axis=0).astype(np.float32)
    y_val = np.concatenate(val_labels, axis=0).astype(np.uint8)
    g_val = np.concatenate(val_groups, axis=0)
    X_test = np.concatenate(test_feats, axis=0).astype(np.float32)
    y_test = np.concatenate(test_labels, axis=0).astype(np.uint8)
    g_test = np.concatenate(test_groups, axis=0)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X_train_s, y_train)
    val_probs = clf.predict_proba(X_val_s)[:, 1]
    test_probs = clf.predict_proba(X_test_s)[:, 1]

    report = {
        "train_cells": int(len(y_train)),
        "train_positive_fraction": float(y_train.mean()),
        "val_cells": int(len(y_val)),
        "val_positive_fraction": float(y_val.mean()),
        "test_cells": int(len(y_test)),
        "test_positive_fraction": float(y_test.mean()),
        "val_auc": float(roc_auc_score(y_val, val_probs)),
        "val_ap": float(average_precision_score(y_val, val_probs)),
        "test_auc": float(roc_auc_score(y_test, test_probs)),
        "test_ap": float(average_precision_score(y_test, test_probs)),
    }

    thresholds = {}
    for target in (0.90, 0.95):
        t = choose_threshold_for_target_recall(y_val, val_probs, target)
        thresholds[f"val_recall_target_{int(target * 100)}"] = {
            "val": evaluate_threshold(y_val, val_probs, g_val, t),
            "test": evaluate_threshold(y_test, test_probs, g_test, t),
        }

    report["thresholds"] = thresholds
    report["keep_fraction_metrics"] = {
        f"top_{int(frac * 100)}pct": evaluate_keep_fraction(y_test, test_probs, g_test, frac)
        for frac in (0.10, 0.20, 0.30, 0.40, 0.50)
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (args.output_dir / "report.txt").open("w", encoding="utf-8") as f:
        for key in ("train_cells", "train_positive_fraction", "val_cells", "val_positive_fraction", "test_cells", "test_positive_fraction", "val_auc", "val_ap", "test_auc", "test_ap"):
            f.write(f"{key}={report[key]}\n")
        for name, payload in thresholds.items():
            f.write(f"[{name}]\n")
            for split in ("val", "test"):
                for metric_name, metric_value in payload[split].items():
                    f.write(f"{split}_{metric_name}={metric_value}\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
