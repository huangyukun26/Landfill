# Landfill

Landfill semantic segmentation research workspace based on a SAM + LoRA adaptation pipeline, with an additional research track for evaluating ESD (Embedded Seamless Data) as a semantic prior for landfill detection.

This repository is organized as a lightweight, reproducible project scaffold around the current local assets:

- local landfill image and mask dataset
- legacy training / inference code under `code/`
- ESD download helper
- research documentation that defines the current task scope

The raw dataset is intentionally **not versioned** in Git.

## Current Scope

The immediate research question is:

1. Can the existing annotated remote-sensing imagery support a reliable landfill segmentation baseline?
2. Can ESD high-dimensional Earth embeddings help landfill prediction?
3. If yes, is ESD more useful for:
   - classification / candidate screening
   - coarse localization
   - feature fusion with high-resolution imagery

The current conclusion is:

- `image-only` is the correct baseline for semantic segmentation
- `ESD-only` is more suitable for classification or coarse localization
- `image + ESD` is the most promising route for improving robustness

See [docs/project_scope.md](./docs/project_scope.md) and [docs/data_inventory.md](./docs/data_inventory.md).

## Repository Layout

```text
.
|-- README.md
|-- requirements.txt
|-- download_esd.py
|-- esd_tiles_needed.txt
|-- docs/
|   |-- project_scope.md
|   `-- data_inventory.md
|-- scripts/
|   `-- sanity_check.py
`-- code/
    |-- train.py
    |-- test_landfill.py
    |-- trainer.py
    |-- datasets/
    |-- segment_anything/
    `-- ...
```

## Local Data Snapshot

The local dataset currently contains:

- `train`: 957 image / mask pairs
- `val`: 206 image / mask pairs
- `test`: 206 image / mask pairs
- image format: `2000 x 2000` RGBA TIFF
- mask format: `2000 x 2000` PNG, binary landfill annotation

The dataset split files exist locally but are not committed.

## Quick Start

### 1. Create an environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the local sanity check

```bash
python scripts/sanity_check.py --project-root .
```

### 3. Train locally after preparing checkpoints and data

Example:

```bash
python code/train.py ^
  --root_path G:\GIS\LANDFILL ^
  --list_dir G:\GIS\LANDFILL\ImageSets ^
  --output G:\GIS\LANDFILL\outputs
```

### 4. Test locally

```bash
python code/test_landfill.py ^
  --volume_path G:\GIS\LANDFILL ^
  --list_dir G:\GIS\LANDFILL\ImageSets ^
  --output_dir G:\GIS\LANDFILL\outputs\eval
```

## Notes

- The code under `code/` is adapted from `SAMed` and still contains historical experimental branches unrelated to landfill.
- Raw datasets, checkpoints, and generated outputs are ignored by Git.
- `download_esd.py` has been sanitized to require credentials from CLI arguments or environment variables instead of hard-coded secrets.

## Next Engineering Steps

1. Establish a reproducible `image-only` segmentation baseline.
2. Align landfill samples with ESD tile / year metadata.
3. Run `ESD-only` classification as a feasibility test.
4. Evaluate `image + ESD` fusion.
