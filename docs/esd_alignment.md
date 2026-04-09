# ESD Alignment

This note summarizes how the current landfill dataset should be aligned with ESD downloads.

## Key Result

The existing `esd_tiles_needed.txt` exactly matches the spatial footprint of the current landfill dataset.

Verified against local metadata:

- dataset samples: `1369`
- dataset unique MGRS tiles: `860`
- tiles in `esd_tiles_needed.txt`: `860`
- covered samples: `1369 / 1369`
- spatial coverage ratio: `1.0`
- used tiles missing from txt: `0`
- txt tiles not used by dataset: `0`

## What This Means

You do **not** need to download the full ESD archive.

Spatially, the current TXT file is already the correct subset definition for the landfill samples.

## Important Limitation

The dataset still lacks acquisition year metadata.

That means:

- spatial alignment is solved
- temporal alignment is not solved yet

So the correct download strategy is:

1. keep `esd_tiles_needed.txt` as the spatial filter
2. obtain the sample year mapping from the mentor or data provider
3. only download the required tile subset for the relevant year or years

## Download Strategy

### If all samples belong to one year

Download only:

- the tiles in `esd_tiles_needed.txt`
- for that single year

### If samples span multiple years

Download only:

- the tiles in `esd_tiles_needed.txt`
- for the specific years actually used by the sample set

Do **not** download 2000-2024 blindly.

## Practical Recommendation

Before downloading any large ESD subset, ask for one of the following:

1. sample acquisition year per image
2. original imagery metadata table
3. a mapping from sample id to year

Without year, the download scope cannot be minimized correctly in the time dimension.

## Reproducible Check

Run:

```bash
python scripts/check_esd_coverage.py
```

This will export:

- `metadata/esd_tile_usage.csv`

That file lists each required ESD tile and how many landfill samples fall into it.
