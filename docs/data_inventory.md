# Data Inventory

This summary is derived from the current local workspace at `G:\GIS\LANDFILL`.

## Dataset Layout

Expected local structure:

```text
G:\GIS\LANDFILL
|-- ann/
|   |-- train/
|   |-- val/
|   `-- test/
|-- img/
|   |-- train/
|   |-- val/
|   `-- test/
`-- ImageSets/
    |-- train.txt
    |-- val.txt
    `-- test.txt
```

## Current Counts

| Split | Images (`.tiff`) | Masks (`.png`) | Split File Lines |
|------|-------------------:|---------------:|-----------------:|
| train | 957 | 957 | 957 |
| val   | 206 | 206 | 206 |
| test  | 206 | 206 | 206 |

Additional local file:

- `ImageSets/test_paper_five_images.txt`: 5 samples used for paper figures or qualitative examples

## Sample Naming Pattern

Example image:

- `1000_US_-92.5387_39.7341.tiff`

Example mask:

- `1000_US_-92.5387_39.7341.png`

This indicates:

- a stable sample id prefix
- country code
- longitude / latitude embedded in the filename

That naming pattern is important because it may support downstream alignment with ESD tiles.

## Image Characteristics

Observed local samples:

- image size: `2000 x 2000`
- image mode: `RGBA`
- mask size: `2000 x 2000`
- mask mode: `L` or `RGB`
- annotation semantics: binary landfill / non-landfill

## Engineering Implications

1. The dataset is already large enough for a first segmentation baseline.
2. Filenames carry geographic hints, but year information still needs to be confirmed for ESD alignment.
3. Because images are RGBA, any training pipeline that forces `RGB` should be reviewed to ensure the alpha band is not carrying useful validity information.
4. ESD alignment is blocked unless each sample can be matched to both:
   - acquisition year
   - geographic tile or spatial extent
