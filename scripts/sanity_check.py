from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPLITS = ("train", "val", "test")


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stem_without_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".tiff"):
        return name[:-5]
    if lower.endswith(".tif"):
        return name[:-4]
    if lower.endswith(".png"):
        return name[:-4]
    return Path(name).stem


def check_split(project_root: Path, split: str) -> list[str]:
    issues: list[str] = []
    img_dir = project_root / "img" / split
    ann_dir = project_root / "ann" / split
    list_path = project_root / "ImageSets" / f"{split}.txt"

    images = sorted(p.name for p in img_dir.glob("*.tif*")) if img_dir.exists() else []
    masks = sorted(p.name for p in ann_dir.glob("*.png")) if ann_dir.exists() else []
    listed = read_lines(list_path)

    print(f"[{split}] images={len(images)} masks={len(masks)} list_entries={len(listed)}")

    if img_dir.exists() and ann_dir.exists() and len(images) != len(masks):
        issues.append(f"{split}: image/mask count mismatch ({len(images)} vs {len(masks)})")

    if list_path.exists() and images and len(listed) != len(images):
        issues.append(f"{split}: split file count mismatch ({len(listed)} vs {len(images)})")

    image_stems = {stem_without_suffix(name) for name in images}
    mask_stems = {stem_without_suffix(name) for name in masks}
    listed_stems = {stem_without_suffix(name) for name in listed}

    missing_masks = sorted(image_stems - mask_stems)[:5]
    missing_images = sorted(mask_stems - image_stems)[:5]
    missing_from_list = sorted(image_stems - listed_stems)[:5]
    missing_from_dir = sorted(listed_stems - image_stems)[:5]

    if missing_masks:
        issues.append(f"{split}: missing masks for samples {missing_masks}")
    if missing_images:
        issues.append(f"{split}: missing images for samples {missing_images}")
    if list_path.exists() and missing_from_list:
        issues.append(f"{split}: files not covered by split list {missing_from_list}")
    if list_path.exists() and missing_from_dir:
        issues.append(f"{split}: split list entries missing on disk {missing_from_dir}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local landfill project layout.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Path to the project root.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    print(f"Project root: {project_root}")

    required = [
        project_root / "code" / "train.py",
        project_root / "code" / "test_landfill.py",
        project_root / "download_esd.py",
        project_root / "docs" / "project_scope.md",
        project_root / "docs" / "data_inventory.md",
    ]

    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        print("Missing required project files:")
        for item in missing_required:
            print(f"  - {item}")
        return 1

    dataset_available = all((project_root / part).exists() for part in ("img", "ann", "ImageSets"))
    if not dataset_available:
        print("Dataset directories are not present. Repository scaffold is valid; data checks skipped.")
        return 0

    issues: list[str] = []
    for split in SPLITS:
        issues.extend(check_split(project_root, split))

    paper_subset = project_root / "ImageSets" / "test_paper_five_images.txt"
    if paper_subset.exists():
        lines = read_lines(paper_subset)
        print(f"[paper_subset] entries={len(lines)}")

    if issues:
        print("\nSanity check failed:")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("\nSanity check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
