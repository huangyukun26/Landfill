from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from pyproj import Transformer
from sklearn.ensemble import RandomForestClassifier
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


def aggregate_window_features(codes_window: np.ndarray, quantizer: Quantizer) -> tuple[np.ndarray, float]:
    valid = ~np.all(codes_window[:12] == 0, axis=0)
    valid_fraction = float(valid.mean())
    vectors = decode_esd_vectors(codes_window, quantizer)  # months, h, w, ch
    flat = vectors[ :, valid, : ] if valid.any() else None
    if flat is None or flat.size == 0:
        feat = np.zeros(145, dtype=np.float32)
        feat[-1] = valid_fraction
        return feat, valid_fraction
    # vectors[:, valid, :] -> months, n, channels
    flat = flat.reshape(12, -1, 6)
    mean = flat.mean(axis=1).reshape(-1)
    std = flat.std(axis=1).reshape(-1)
    feat = np.concatenate([mean, std, np.array([valid_fraction], dtype=np.float32)]).astype(np.float32)
    return feat, valid_fraction


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ar1, ac0, ac1 = a
    br0, br1, bc0, bc1 = b
    return not (ar1 <= br0 or br1 <= ar0 or ac1 <= bc0 or bc1 <= ac0)


def sample_negative_windows(
    tile_shape: tuple[int, int],
    positive_windows: list[tuple[int, int, int, int]],
    window_size: tuple[int, int],
    need: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    height, width = tile_shape
    wh, ww = window_size
    results: list[tuple[int, int, int, int]] = []
    attempts = 0
    max_attempts = max(200, need * 200)
    while len(results) < need and attempts < max_attempts:
        attempts += 1
        row0 = rng.randint(0, max(0, height - wh))
        col0 = rng.randint(0, max(0, width - ww))
        win = (row0, row0 + wh, col0, col0 + ww)
        if any(overlaps(win, p) for p in positive_windows):
            continue
        if any(overlaps(win, r) for r in results):
            continue
        results.append(win)
    return results


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


def evaluate_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (probs >= threshold).astype(np.uint8)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    neg_mask = y_true == 0
    negative_rejection = float((y_pred[neg_mask] == 0).mean()) if neg_mask.any() else 0.0
    retained_fraction = float(y_pred.mean())
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_rejection": negative_rejection,
        "retained_fraction": retained_fraction,
        "speedup_approx": float(1.0 / max(retained_fraction, 1e-9)),
    }


def evaluate_keep_fraction(y_true: np.ndarray, probs: np.ndarray, keep_fraction: float) -> dict[str, float]:
    keep_n = max(1, int(len(probs) * keep_fraction))
    order = np.argsort(-probs)
    y_pred = np.zeros_like(y_true, dtype=np.uint8)
    y_pred[order[:keep_n]] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    neg_mask = y_true == 0
    negative_rejection = float((y_pred[neg_mask] == 0).mean()) if neg_mask.any() else 0.0
    return {
        "keep_fraction": keep_fraction,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "negative_rejection": negative_rejection,
        "speedup_approx": float(1.0 / keep_fraction),
    }


def fit_and_eval(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, object]:
    if name == "logreg":
        scaler = StandardScaler()
        X_train_in = scaler.fit_transform(X_train)
        X_val_in = scaler.transform(X_val)
        X_test_in = scaler.transform(X_test)
        model = LogisticRegression(
            max_iter=400,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=1,
            random_state=42,
        )
    elif name == "rf":
        scaler = None
        X_train_in, X_val_in, X_test_in = X_train, X_val, X_test
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(name)

    model.fit(X_train_in, y_train)
    val_probs = model.predict_proba(X_val_in)[:, 1]
    test_probs = model.predict_proba(X_test_in)[:, 1]

    result: dict[str, object] = {
        "val_auc": float(roc_auc_score(y_val, val_probs)),
        "val_ap": float(average_precision_score(y_val, val_probs)),
        "test_auc": float(roc_auc_score(y_test, test_probs)),
        "test_ap": float(average_precision_score(y_test, test_probs)),
        "thresholds": {},
        "keep_fraction_metrics": {},
    }
    for target in (0.90, 0.95):
        threshold = choose_threshold_for_target_recall(y_val, val_probs, target)
        result["thresholds"][f"val_recall_target_{int(target * 100)}"] = {
            "val": evaluate_threshold(y_val, val_probs, threshold),
            "test": evaluate_threshold(y_test, test_probs, threshold),
        }
    for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        result["keep_fraction_metrics"][f"top_{int(frac * 100)}pct"] = evaluate_keep_fraction(y_test, test_probs, frac)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-json", type=Path, required=True)
    parser.add_argument("--esd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--neg-per-positive", type=int, default=10)
    parser.add_argument("--min-valid-fraction", type=float, default=0.5)
    args = parser.parse_args()

    rng = random.Random(42)
    quantizer = Quantizer()
    rows = json.loads(args.subset_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_tile: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_tile.setdefault(row["tile"], []).append(row)

    window_rows: list[dict[str, object]] = []

    for tile, samples in by_tile.items():
        esd_path = args.esd_dir / f"SDC30_EBD_V001_{tile}_2024.tif"
        if not esd_path.exists():
            continue

        esd_info = raster_info(esd_path)
        codes_full = tifffile.imread(esd_path)
        tile_h, tile_w = codes_full.shape[1], codes_full.shape[2]

        positive_windows: list[tuple[int, int, int, int]] = []
        sample_windows: list[tuple[dict[str, str], tuple[int, int, int, int]]] = []

        for sample in samples:
            img_path = ROOT / "img" / sample["split"] / f"{sample['stem']}.tiff"
            img_info = raster_info(img_path)
            win = sample_tile_window(img_info, esd_info)
            positive_windows.append(win)
            sample_windows.append((sample, win))

        for sample, win in sample_windows:
            row0, row1, col0, col1 = win
            codes_window = codes_full[:, row0:row1, col0:col1]
            feat, valid_fraction = aggregate_window_features(codes_window, quantizer)
            window_rows.append(
                {
                    "split": sample["split"],
                    "tile": tile,
                    "stem": sample["stem"],
                    "label": 1,
                    "window": [row0, row1, col0, col1],
                    "valid_fraction": valid_fraction,
                    "features": feat.tolist(),
                }
            )

            neg_windows = sample_negative_windows(
                (tile_h, tile_w),
                positive_windows,
                (row1 - row0, col1 - col0),
                args.neg_per_positive,
                rng,
            )
            for idx, neg_win in enumerate(neg_windows):
                nr0, nr1, nc0, nc1 = neg_win
                neg_codes = codes_full[:, nr0:nr1, nc0:nc1]
                neg_feat, neg_valid_fraction = aggregate_window_features(neg_codes, quantizer)
                if neg_valid_fraction < args.min_valid_fraction:
                    continue
                window_rows.append(
                    {
                        "split": sample["split"],
                        "tile": tile,
                        "stem": f"{sample['stem']}__neg{idx}",
                        "label": 0,
                        "window": [nr0, nr1, nc0, nc1],
                        "valid_fraction": neg_valid_fraction,
                        "features": neg_feat.tolist(),
                    }
                )

    metadata_path = args.output_dir / "window_metadata.json"
    metadata_path.write_text(json.dumps(window_rows, indent=2), encoding="utf-8")

    X_train = np.array([r["features"] for r in window_rows if r["split"] == "train"], dtype=np.float32)
    y_train = np.array([r["label"] for r in window_rows if r["split"] == "train"], dtype=np.uint8)
    X_val = np.array([r["features"] for r in window_rows if r["split"] == "val"], dtype=np.float32)
    y_val = np.array([r["label"] for r in window_rows if r["split"] == "val"], dtype=np.uint8)
    X_test = np.array([r["features"] for r in window_rows if r["split"] == "test"], dtype=np.float32)
    y_test = np.array([r["label"] for r in window_rows if r["split"] == "test"], dtype=np.uint8)

    report: dict[str, object] = {
        "train_windows": int(len(y_train)),
        "train_positive_fraction": float(y_train.mean()),
        "val_windows": int(len(y_val)),
        "val_positive_fraction": float(y_val.mean()),
        "test_windows": int(len(y_test)),
        "test_positive_fraction": float(y_test.mean()),
        "models": {},
    }

    for model_name in ("logreg", "rf"):
        report["models"][model_name] = fit_and_eval(model_name, X_train, y_train, X_val, y_val, X_test, y_test)

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "report.txt").open("w", encoding="utf-8") as f:
        for key in ("train_windows", "train_positive_fraction", "val_windows", "val_positive_fraction", "test_windows", "test_positive_fraction"):
            f.write(f"{key}={report[key]}\n")
        for model_name, model_report in report["models"].items():
            f.write(f"[{model_name}]\n")
            for metric in ("val_auc", "val_ap", "test_auc", "test_ap"):
                f.write(f"{metric}={model_report[metric]}\n")
            for section_name, payload in model_report["thresholds"].items():
                f.write(f"[{model_name}.{section_name}]\n")
                for split in ("val", "test"):
                    for metric_name, metric_value in payload[split].items():
                        f.write(f"{split}_{metric_name}={metric_value}\n")
            for section_name, payload in model_report["keep_fraction_metrics"].items():
                f.write(f"[{model_name}.{section_name}]\n")
                for metric_name, metric_value in payload.items():
                    f.write(f"{metric_name}={metric_value}\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
