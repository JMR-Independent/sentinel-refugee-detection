# NDBI Spectral Gap Predicts Refugee Camp Detectability from Sentinel-2

Code and data for the paper:

> **Robles Leyton, J. M.** (2026). NDBI Spectral Gap Predicts Refugee Camp Detectability from Sentinel-2: Environmental Limits at 10 m Resolution. *IEEE Geoscience and Remote Sensing Letters* (submitted).

## Key Finding

The NDBI spectral gap between camp and background explains **83% of cross-country detection variance** (*r* = 0.912, permutation *p* = 0.005). Camps built with concrete or metal on rural land are detectable; camps of mud and thatch on spectrally similar soil are invisible at 10 m resolution.

| Country      | Camps | NDBI Gap | LOCO AUC (Spectral) | LOCO AUC (Texture) |
|:-------------|------:|---------:|--------------------:|-------------------:|
| Turkey       |     6 |   +0.129 |               0.773 |              0.629 |
| Uganda       |     6 |   +0.055 |               0.654 |              0.634 |
| Chad         |    19 |   +0.046 |               0.652 |              0.536 |
| Ethiopia     |    26 |   +0.023 |               0.678 |              0.611 |
| Yemen        |     6 |   +0.020 |               0.658 |              0.584 |
| Syria        |    28 |   -0.001 |               0.531 |              0.426 |
| South Sudan  |    10 |   -0.077 |               0.238 |            **0.542** |

## Repository Structure

```
src/                    Core Python modules (data loading, model, training, utils)
notebooks/              Jupyter notebooks (00-06) for the full pipeline
configs/                YAML configuration
download_full_dataset.py   Sentinel-2 tile download via Planetary Computer
data/
  labels/               Camp coordinates (UNHCR, HDX, OpenStreetMap)
  analysis/             Derived features, LOCO results, texture analysis
  sentinel2/            Raw .npy tiles (not in repo; 647 MB, see below)
paper/
  robles_leyton_grsl_2026.tex       Main manuscript (two-column)
  robles_leyton_grsl_2026_review.tex  Review format (single-column)
  cover_letter.tex                  Cover letter
  regenerate_figure.py              Script to reproduce Figure 1
  roble1.{png,pdf}                  Figure 1
tests/                  Unit tests
```

## Reproducing Results

### 1. Environment

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ with numpy, scikit-learn, matplotlib, scipy, rasterio, and planetary-computer.

### 2. Download Sentinel-2 tiles

```bash
python download_full_dataset.py
```

Downloads 1,689 tiles (128 x 128 px, 9 channels) from Microsoft Planetary Computer. Requires ~650 MB disk space and a Planetary Computer API key.

### 3. Run the analysis pipeline

Run the notebooks in order:

| Notebook | Description |
|:---------|:------------|
| `00_visual_validation.ipynb` | Visual inspection of downloaded tiles |
| `01_download_labels.ipynb` | Compile camp coordinates from UNHCR + OSM |
| `02_download_sentinel2.ipynb` | Download and preprocess Sentinel-2 imagery |
| `03_prepare_tiles.ipynb` | Tile extraction, normalization, train/test split |
| `04_train_model.ipynb` | Train CNN baseline (ResNet-18; GPU recommended) |
| `05_generalization.ipynb` | Cross-country generalization analysis |
| `06_validation.ipynb` | LOCO evaluation, NDBI gap analysis, statistical tests |

### 4. Regenerate Figure 1

```bash
python paper/regenerate_figure.py
```

Produces `paper/roble1.{png,pdf,tiff}` from the derived features in `data/analysis/`.

## Data Sources

- **Sentinel-2 L2A imagery:** [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/)
- **Camp locations:** [UNHCR Data Portal](https://data.unhcr.org/), [HDX](https://data.humdata.org/), [OpenStreetMap](https://www.openstreetmap.org/)

## Responsible Use

This research characterizes the **physical limits** of satellite-based camp detection. It must not be used for surveillance, targeting, or any activity that could harm vulnerable populations.

## Citation

```bibtex
@article{RoblesLeyton2026,
  author  = {Robles Leyton, Jose Miguel},
  title   = {{NDBI} Spectral Gap Predicts Refugee Camp Detectability
             from {Sentinel-2}: Environmental Limits at 10\,m Resolution},
  journal = {IEEE Geosci. Remote Sens. Lett.},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT
