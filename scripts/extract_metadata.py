from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any
from PIL import Image, TiffTags


def get_tags(image: Image.Image) -> dict[str, Any]:
    return {TiffTags.TAGS.get(k, k): v for k, v in getattr(image, "tag_v2", {}).items()}


def infer_filename_pattern(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) == 4:
        return "id_country_lon_lat"
    if len(parts) == 5:
        return "id_minx_maxx_miny_maxy"
    if len(parts) == 6:
        return "id_country_minx_maxx_miny_maxy"
    return f"other_{len(parts)}"


def extract_record(path: Path) -> dict[str, Any]:
    split = path.parent.name
    stem = path.stem
    parts = stem.split("_")
    image = Image.open(path)
    tags = get_tags(image)

    record = {
        "split": split,
        "filename": path.name,
        "stem": stem,
        "pattern": infer_filename_pattern(stem),
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "crs": tags.get("GeoAsciiParamsTag", ""),
        "pixel_scale": tags.get("ModelPixelScaleTag", ""),
        "tiepoint": tags.get("ModelTiepointTag", ""),
        "datetime": tags.get("DateTime", ""),
    }

    if len(parts) == 4:
        record.update({
            "sample_id": parts[0],
            "country": parts[1],
            "center_lon": parts[2],
            "center_lat": parts[3],
            "minx": "",
            "maxx": "",
            "miny": "",
            "maxy": "",
        })
    elif len(parts) == 5:
        record.update({
            "sample_id": parts[0],
            "country": "",
            "center_lon": "",
            "center_lat": "",
            "minx": parts[1],
            "maxx": parts[2],
            "miny": parts[3],
            "maxy": parts[4],
        })
    elif len(parts) == 6:
        record.update({
            "sample_id": parts[0],
            "country": parts[1],
            "center_lon": "",
            "center_lat": "",
            "minx": parts[2],
            "maxx": parts[3],
            "miny": parts[4],
            "maxy": parts[5],
        })
    else:
        record.update({
            "sample_id": parts[0] if parts else "",
            "country": "",
            "center_lon": "",
            "center_lat": "",
            "minx": "",
            "maxx": "",
            "miny": "",
            "maxy": "",
        })

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Export landfill image metadata from local TIFF files.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("metadata/image_metadata.csv"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_path = args.output if args.output.is_absolute() else project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted((project_root / "img").rglob("*.tif*")):
        rows.append(extract_record(path))

    fieldnames = [
        "split", "filename", "stem", "pattern", "sample_id", "country",
        "center_lon", "center_lat", "minx", "maxx", "miny", "maxy",
        "width", "height", "mode", "crs", "pixel_scale", "tiepoint", "datetime",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
