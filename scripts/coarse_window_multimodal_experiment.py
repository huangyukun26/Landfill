from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
from PIL import Image
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class RasterInfo:
    width: int
    height: int
    x0: float
    y0: float
    px: float
    py: float
    epsg: int


class Quantizer(torch.nn.Module):
    def __init__(self, levels: list[int] | None = None):
        super().__init__()
        levels = levels or [8, 8, 8, 5, 5, 5]
        _levels = torch.tensor(levels, dtype=torch.int32)
        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        self.register_buffer("_basis", _basis, persistent=False)

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def indices_to_level_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.unsqueeze(-1)
        return (indices // self._basis) % self._levels

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self._scale_and_shift_inverse(self.indices_to_level_indices(indices))


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


def chip_pixel_window_to_bounds(info: RasterInfo, row0: int, row1: int, col0: int, col1: int) -> tuple[float, float, float, float]:
    minx = info.x0 + col0 * info.px
    maxx = info.x0 + col1 * info.px
    maxy = info.y0 - row0 * info.py
    miny = info.y0 - row1 * info.py
    return minx, miny, maxx, maxy


def map_chip_window_to_esd(
    img_info: RasterInfo,
    esd_info: RasterInfo,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
) -> tuple[int, int, int, int]:
    minx, miny, maxx, maxy = chip_pixel_window_to_bounds(img_info, row0, row1, col0, col1)
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

    e_col0 = max(0, int(math.floor((tx_min - esd_info.x0) / esd_info.px)))
    e_col1 = min(esd_info.width, int(math.ceil((tx_max - esd_info.x0) / esd_info.px)))
    e_row0 = max(0, int(math.floor((esd_info.y0 - ty_max) / esd_info.py)))
    e_row1 = min(esd_info.height, int(math.ceil((esd_info.y0 - ty_min) / esd_info.py)))
    return e_row0, e_row1, e_col0, e_col1


def decode_esd_vectors(codes: np.ndarray, quantizer: Quantizer) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(codes[:12].astype(np.int32))
        vectors = quantizer.indices_to_codes(tensor).numpy()
    return vectors.astype(np.float32)  # months, h, w, channels


def aggregate_esd_window(codes_window: np.ndarray, quantizer: Quantizer) -> np.ndarray:
    valid = ~np.all(codes_window[:12] == 0, axis=0)
    valid_fraction = float(valid.mean())
    if valid_fraction == 0.0:
        feat = np.zeros(145, dtype=np.float32)
        feat[-1] = 0.0
        return feat
    vectors = decode_esd_vectors(codes_window, quantizer)
    flat = vectors[:, valid, :].reshape(12, -1, 6)
    mean = flat.mean(axis=1).reshape(-1)
    std = flat.std(axis=1).reshape(-1)
    return np.concatenate([mean, std, np.array([valid_fraction], dtype=np.float32)]).astype(np.float32)


def crop_rgb_feature(image_rgb: np.ndarray, row0: int, row1: int, col0: int, col1: int, size: int) -> np.ndarray:
    crop = image_rgb[row0:row1, col0:col1]
    pil = Image.fromarray(crop)
    resized = np.asarray(pil.resize((size, size), resample=Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    stats = np.concatenate([resized.mean(axis=(0, 1)), resized.std(axis=(0, 1))], axis=0)
    return np.concatenate([resized.reshape(-1), stats], axis=0).astype(np.float32)


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


def sample_negative_windows(
    mask_bin: np.ndarray,
    window_px: int,
    need: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    height, width = mask_bin.shape
    results: list[tuple[int, int, int, int]] = []
    attempts = 0
    max_attempts = need * 500
    while len(results) < need and attempts < max_attempts:
        attempts += 1
        row0 = rng.randint(0, max(0, height - window_px))
        col0 = rng.randint(0, max(0, width - window_px))
        row1 = row0 + window_px
        col1 = col0 + window_px
        if mask_bin[row0:row1, col0:col1].sum() != 0:
            continue
        win = (row0, row1, col0, col1)
        if any(not (row1 <= r0 or rr1 <= row0 or col1 <= c0 or cc1 <= col0) for r0, rr1, c0, cc1 in results):
            continue
        results.append(win)
    return results


def center_window(mask_bin: np.ndarray, window_px: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask_bin > 0)
    cy = int(round(float(ys.mean())))
    cx = int(round(float(xs.mean())))
    h, w = mask_bin.shape
    row0 = min(max(0, cy - window_px // 2), h - window_px)
    col0 = min(max(0, cx - window_px // 2), w - window_px)
    return row0, row0 + window_px, col0, col0 + window_px


def fit_and_eval(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs", n_jobs=1, random_state=42)
    clf.fit(X_train_s, y_train)
    val_probs = clf.predict_proba(X_val_s)[:, 1]
    test_probs = clf.predict_proba(X_test_s)[:, 1]

    report: dict[str, object] = {
        "val_auc": float(roc_auc_score(y_val, val_probs)),
        "val_ap": float(average_precision_score(y_val, val_probs)),
        "test_auc": float(roc_auc_score(y_test, test_probs)),
        "test_ap": float(average_precision_score(y_test, test_probs)),
        "thresholds": {},
        "keep_fraction_metrics": {},
    }
    for target in (0.90, 0.95):
        threshold = choose_threshold_for_target_recall(y_val, val_probs, target)
        report["thresholds"][f"val_recall_target_{int(target * 100)}"] = {
            "val": evaluate_threshold(y_val, val_probs, threshold),
            "test": evaluate_threshold(y_test, test_probs, threshold),
        }
    for frac in (0.10, 0.20, 0.30, 0.40, 0.50):
        report["keep_fraction_metrics"][f"top_{int(frac * 100)}pct"] = evaluate_keep_fraction(y_test, test_probs, frac)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--subset-json", type=Path, required=True)
    parser.add_argument("--esd-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-px", type=int, default=512)
    parser.add_argument("--rgb-size", type=int, default=16)
    parser.add_argument("--neg-per-positive", type=int, default=5)
    args = parser.parse_args()

    rng = random.Random(42)
    quantizer = Quantizer()
    subset = json.loads(args.subset_json.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sample in subset:
        split = sample["split"]
        stem = sample["stem"]
        tile = sample["tile"]
        img_path = args.project_root / "img" / split / f"{stem}.tiff"
        mask_path = args.project_root / "ann" / split / f"{stem}.png"
        esd_path = args.esd_dir / f"SDC30_EBD_V001_{tile}_2024.tif"
        if not esd_path.exists():
            continue

        img_info = raster_info(img_path)
        esd_info = raster_info(esd_path)
        image_rgb = np.asarray(Image.open(img_path).convert("RGB"))
        mask_bin = (np.asarray(Image.open(mask_path).convert("L")) > 127).astype(np.uint8)
        pos_win = center_window(mask_bin, args.window_px)
        neg_wins = sample_negative_windows(mask_bin, args.window_px, args.neg_per_positive, rng)

        codes_full = tifffile.imread(esd_path)

        def append_row(label: int, win: tuple[int, int, int, int], sample_name: str) -> None:
            row0, row1, col0, col1 = win
            rgb_feat = crop_rgb_feature(image_rgb, row0, row1, col0, col1, args.rgb_size)
            e_row0, e_row1, e_col0, e_col1 = map_chip_window_to_esd(img_info, esd_info, row0, row1, col0, col1)
            esd_feat = aggregate_esd_window(codes_full[:, e_row0:e_row1, e_col0:e_col1], quantizer)
            rows.append(
                {
                    "split": split,
                    "sample": sample_name,
                    "label": label,
                    "rgb_feat": rgb_feat.tolist(),
                    "esd_feat": esd_feat.tolist(),
                }
            )

        append_row(1, pos_win, stem)
        for idx, win in enumerate(neg_wins):
            append_row(0, win, f"{stem}__neg{idx}")

    (args.output_dir / "window_rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def mat(split: str, key: str) -> np.ndarray:
        return np.array([r[key] for r in rows if r["split"] == split], dtype=np.float32)

    def lab(split: str) -> np.ndarray:
        return np.array([r["label"] for r in rows if r["split"] == split], dtype=np.uint8)

    y_train = lab("train")
    y_val = lab("val")
    y_test = lab("test")
    X_train_rgb = mat("train", "rgb_feat")
    X_val_rgb = mat("val", "rgb_feat")
    X_test_rgb = mat("test", "rgb_feat")
    X_train_esd = mat("train", "esd_feat")
    X_val_esd = mat("val", "esd_feat")
    X_test_esd = mat("test", "esd_feat")

    report = {
        "train_windows": int(len(y_train)),
        "val_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "positive_fraction": {
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
        "models": {
            "rgb_only": fit_and_eval(X_train_rgb, y_train, X_val_rgb, y_val, X_test_rgb, y_test),
            "esd_only": fit_and_eval(X_train_esd, y_train, X_val_esd, y_val, X_test_esd, y_test),
            "rgb_plus_esd": fit_and_eval(
                np.concatenate([X_train_rgb, X_train_esd], axis=1),
                y_train,
                np.concatenate([X_val_rgb, X_val_esd], axis=1),
                y_val,
                np.concatenate([X_test_rgb, X_test_esd], axis=1),
                y_test,
            ),
        },
    }

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.output_dir / "report.txt").open("w", encoding="utf-8") as f:
        for key in ("train_windows", "val_windows", "test_windows"):
            f.write(f"{key}={report[key]}\n")
        for split, frac in report["positive_fraction"].items():
            f.write(f"positive_fraction_{split}={frac}\n")
        for model_name, payload in report["models"].items():
            f.write(f"[{model_name}]\n")
            for metric_name in ("val_auc", "val_ap", "test_auc", "test_ap"):
                f.write(f"{metric_name}={payload[metric_name]}\n")
            for section_name, section_payload in payload["thresholds"].items():
                f.write(f"[{model_name}.{section_name}]\n")
                for split in ("val", "test"):
                    for metric_name, metric_value in section_payload[split].items():
                        f.write(f"{split}_{metric_name}={metric_value}\n")
            for section_name, section_payload in payload["keep_fraction_metrics"].items():
                f.write(f"[{model_name}.{section_name}]\n")
                for metric_name, metric_value in section_payload.items():
                    f.write(f"{metric_name}={metric_value}\n")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
