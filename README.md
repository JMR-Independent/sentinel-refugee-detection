# Sentinel-2 Refugee Camp Detection

A scalable, fully open pipeline to identify candidate informal settlements from Sentinel-2 imagery.

## Method

- **Input:** Sentinel-2 L2A multispectral imagery (6 bands: RGB + NIR + SWIR)
- **Model:** ResNet-18 binary classifier (camp vs non-camp)
- **Training data:** Known camps from UNHCR + OpenStreetMap (Syria, South Sudan)
- **Evaluation:** Generalization to unseen countries (Chad, Ethiopia, Yemen)

## Project Structure

```
notebooks/          Jupyter notebooks (01-06) for the full pipeline
src/                Python modules (data, model, training, utilities)
configs/            YAML configuration files
data/               Downloaded imagery and labels (not in git)
paper/              Manuscript draft
```

## Quick Start

```bash
pip install -r requirements.txt
```

Then run the notebooks in order:
1. `01_download_labels.ipynb` - Get camp coordinates from UNHCR + OSM
2. `02_download_sentinel2.ipynb` - Download Sentinel-2 tiles (run on Colab)
3. `03_prepare_tiles.ipynb` - Tile, normalize, split dataset
4. `04_train_model.ipynb` - Train CNN classifier (run on Colab)
5. `05_generalization.ipynb` - Test on unseen countries
6. `06_validation.ipynb` - Validation analysis and figures

## Data Sources

- [Sentinel-2](https://planetarycomputer.microsoft.com/) via Microsoft Planetary Computer
- [UNHCR Data](https://data.unhcr.org/) for camp locations
- [OpenStreetMap](https://www.openstreetmap.org/) via Overpass API

## Responsible Use

This project identifies candidate informal settlements from satellite imagery. It must not be
used for surveillance, targeting, or any activity that could harm vulnerable populations.
All detections require human review and local contextual verification.

## License

MIT
