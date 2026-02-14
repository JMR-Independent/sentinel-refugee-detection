"""Retry failed downloads with rate limiting."""
import csv, json, os, sys, time, traceback
from collections import defaultdict
import numpy as np

sys.path.insert(0, '.')
from src.utils import compute_indices, save_tile, load_config

DATA_DIR = 'data/sentinel2'
LABELS_FILE = 'data/labels/full_dataset_all_locations.csv'

config = load_config()

def log(msg):
    print(msg, flush=True)

def download_one(lat, lon, tile_id, country, label, config):
    """Download one tile with full error reporting."""
    import planetary_computer as pc
    from pystac_client import Client
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    
    out_path = f"{DATA_DIR}/{tile_id}.npy"
    if os.path.exists(out_path):
        return True
    
    # Search
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    delta = 0.05
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{config['date_range'][0]}/{config['date_range'][1]}",
        query={"eo:cloud_cover": {"lt": config["cloud_threshold"]}},
        max_items=3,
    )
    items = list(search.items())
    if not items:
        return False
    
    item = items[0]
    tile_size = config['tile_size_download']
    resolution = config['resolution']
    half_extent = (tile_size * resolution) / 2
    
    tile_bands = []
    meta = {}
    for band in config['bands']:
        href = item.assets[band].href
        with rasterio.open(href) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            window = from_bounds(
                x - half_extent, y - half_extent,
                x + half_extent, y + half_extent,
                src.transform,
            )
            data = src.read(1, window=window,
                          out_shape=(tile_size, tile_size),
                          resampling=rasterio.enums.Resampling.bilinear).astype(np.float32)
            tile_bands.append(data)
            if not meta:
                meta = {"crs": str(src.crs),
                       "datetime": item.datetime.isoformat() if item.datetime else None,
                       "scene_id": item.id}
    
    raw_tile = np.stack(tile_bands, axis=0)
    index_tile = compute_indices(raw_tile)
    meta.update({"tile_id": tile_id, "country": country, "label": label,
                "cloud_pct": item.properties.get('eo:cloud_cover', -1)})
    save_tile(index_tile, out_path, meta)
    return True

# Load missing tiles
locations = []
with open(LABELS_FILE) as f:
    for row in csv.DictReader(f):
        if not os.path.exists(f"{DATA_DIR}/{row['tile_id']}.npy"):
            locations.append(row)

log(f"Retrying {len(locations)} missing tiles...")

success = 0
failed = 0
errors = {}

for i, loc in enumerate(locations):
    tile_id = loc['tile_id']
    try:
        ok = download_one(float(loc['lat']), float(loc['lon']), 
                         tile_id, loc['country'], loc['label'], config)
        if ok:
            success += 1
            log(f"[{i+1:4d}/{len(locations)}] OK   {tile_id:40s} {loc['country']}")
        else:
            failed += 1
            log(f"[{i+1:4d}/{len(locations)}] FAIL {tile_id:40s} {loc['country']} (no scenes)")
    except Exception as e:
        failed += 1
        err_type = type(e).__name__
        errors[err_type] = errors.get(err_type, 0) + 1
        log(f"[{i+1:4d}/{len(locations)}] ERR  {tile_id:40s} {loc['country']} {err_type}: {str(e)[:80]}")
    
    # Rate limiting: 0.5s between tiles, 3s every 20 tiles
    time.sleep(0.5)
    if (i + 1) % 20 == 0:
        time.sleep(3)

log(f"\nRETRY COMPLETE: {success} OK, {failed} failed")
if errors:
    log(f"Error types:")
    for k, v in sorted(errors.items(), key=lambda x: -x[1]):
        log(f"  {k}: {v}")

total_now = len([f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
log(f"Total tiles on disk: {total_now}")
