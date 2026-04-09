from __future__ import annotations

import argparse
import ast
import csv
import math
from collections import Counter
from pathlib import Path

import mgrs

R = 6378137.0


def mercator_to_lonlat(x: float, y: float) -> tuple[float, float]:
    lon = x / R * 180 / math.pi
    lat = (2 * math.atan(math.exp(y / R)) - math.pi / 2) * 180 / math.pi
    return lon, lat


def center_lonlat(row: dict[str, str]) -> tuple[float, float]:
    if row.get("center_lon") and row.get("center_lat"):
        return float(row["center_lon"]), float(row["center_lat"])

    tie = ast.literal_eval(row["tiepoint"])
    scale = ast.literal_eval(row["pixel_scale"])
    ulx = float(tie[3])
    uly = float(tie[4])
    px = float(scale[0])
    py = float(scale[1])
    width = float(row["width"])
    height = float(row["height"])
    cx = ulx + px * width / 2.0
    cy = uly - py * height / 2.0
    return mercator_to_lonlat(cx, cy)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether local landfill samples are covered by the ESD tile list.")
    parser.add_argument("--metadata", type=Path, default=Path("metadata/image_metadata.csv"))
    parser.add_argument("--tiles", type=Path, default=Path("esd_tiles_needed.txt"))
    parser.add_argument("--output", type=Path, default=Path("metadata/esd_tile_usage.csv"))
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.metadata, encoding="utf-8")))
    needed_tiles = {line.strip() for line in args.tiles.read_text(encoding="utf-8").splitlines() if line.strip()}
    converter = mgrs.MGRS()

    tile_counts: Counter[str] = Counter()
    split_tile_counts: Counter[tuple[str, str]] = Counter()

    for row in rows:
        lon, lat = center_lonlat(row)
        tile = converter.toMGRS(lat, lon, MGRSPrecision=0)
        tile_counts[tile] += 1
        split_tile_counts[(row["split"], tile)] += 1

    used_tiles = set(tile_counts)
    missing_tiles = sorted(used_tiles - needed_tiles)
    extra_tiles = sorted(needed_tiles - used_tiles)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tile", "sample_count", "train_count", "val_count", "test_count", "in_needed_list"])
        for tile in sorted(tile_counts):
            writer.writerow([
                tile,
                tile_counts[tile],
                split_tile_counts.get(("train", tile), 0),
                split_tile_counts.get(("val", tile), 0),
                split_tile_counts.get(("test", tile), 0),
                "yes" if tile in needed_tiles else "no",
            ])

    covered_samples = sum(count for tile, count in tile_counts.items() if tile in needed_tiles)

    print(f"dataset_samples={len(rows)}")
    print(f"dataset_unique_tiles={len(used_tiles)}")
    print(f"needed_tiles={len(needed_tiles)}")
    print(f"covered_samples={covered_samples}")
    print(f"coverage_ratio={covered_samples / len(rows):.6f}")
    print(f"used_tiles_missing_from_txt={len(missing_tiles)}")
    print(f"txt_tiles_not_used_by_dataset={len(extra_tiles)}")
    print(f"wrote_tile_usage={output_path}")

    if missing_tiles:
        print("missing_tiles_sample=", missing_tiles[:20])
    if extra_tiles:
        print("extra_tiles_sample=", extra_tiles[:20])

    return 0 if not missing_tiles and not extra_tiles else 1


if __name__ == "__main__":
    raise SystemExit(main())
