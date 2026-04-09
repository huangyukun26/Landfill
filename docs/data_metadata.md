# Data Metadata

This document summarizes the landfill dataset metadata that can be confirmed directly from the current local files in `G:\GIS\LANDFILL`.

## What Can Be Confirmed Now

### 1. File Naming Patterns

The image filenames are not fully uniform. There are three observed patterns across `1369` TIFF files:

| Pattern | Count | Example | Interpretation |
|---|---:|---|---|
| `id_country_lon_lat.tiff` | 992 | `1000_US_-92.5387_39.7341.tiff` | center longitude / latitude in WGS84 |
| `id_minx_maxx_miny_maxy.tiff` | 154 | `1536_-11811645.996306727_-11807645.996306727_4254369.837366666_4258369.837366666.tiff` | bounding box in projected meters |
| `id_country_minx_maxx_miny_maxy.tiff` | 223 | `114_US_-13544758.554120153_-13540758.554120153_4542169.146012883_4546169.146012883.tiff` | bounding box in projected meters with a country token |

This means filename parsing must support mixed coordinate encodings.

### 2. Coordinate System / Projection

All checked TIFF samples are GeoTIFFs and carry spatial metadata.

Observed GeoTIFF tags:

- `GeoAsciiParamsTag = WGS 84 / Pseudo-Mercator|WGS 84|`
- `GeoKeyDirectoryTag` resolves to `EPSG:3857`
- `ModelPixelScaleTag` is approximately `(2.0, 2.0, 0.0)`

Dataset-wide summary from all local TIFF files:

- projection: `EPSG:3857` / `WGS 84 / Pseudo-Mercator`
- pixel size: almost uniformly `2m x 2m`
- raster size: `2000 x 2000`
- nominal spatial extent per tile: about `4000m x 4000m`

### 3. Coordinate Semantics in Filenames

For the `id_country_lon_lat.tiff` pattern, the filename coordinates are the **center point in WGS84**.

This was verified against GeoTIFF tiepoint metadata. Example:

- filename: `1000_US_-92.5387_39.7341.tiff`
- GeoTIFF upper-left tiepoint and pixel size imply a raster center at:
  - longitude: `-92.5387`
  - latitude: `39.7341`

For the `bbox` filename patterns, the four numeric values match the projected raster bounds in `EPSG:3857`.

### 4. Split Counts

The local dataset currently contains:

| Split | Images | Masks | Split List Entries |
|---|---:|---:|---:|
| train | 957 | 957 | 957 |
| val | 206 | 206 | 206 |
| test | 206 | 206 | 206 |

### 5. Image / Label Characteristics

- image format: RGBA TIFF
- image size: `2000 x 2000`
- mask format: PNG
- mask size: `2000 x 2000`
- labels are binary landfill / non-landfill masks

## What Cannot Be Confirmed Yet

### 1. Acquisition Year

The acquisition year is **not present** in the current filename patterns.

Also, no TIFF `DateTime` metadata was found in the local imagery.

This means the year needed for ESD alignment cannot currently be recovered from local raster files alone.

### 2. Original Sensor / Source Metadata

The local TIFF metadata confirms georeferencing, but not the upstream sensor or acquisition timestamp needed for a precise temporal match to ESD.

## Impact on ESD Research

For ESD alignment, the dataset is already strong on:

- spatial metadata
- tile georeferencing
- coordinate extraction

But it is currently blocked on one critical field:

- **year**

Without year, you cannot confidently map each landfill sample to the correct annual ESD embedding.

## Recommended Next Action

Ask the mentor or data provider for one of the following:

1. the acquisition year for each sample
2. the original image source table
3. a metadata spreadsheet linking sample ids to year / sensor / region

Once the year is available, the dataset can be aligned to ESD tiles in a defensible way.
