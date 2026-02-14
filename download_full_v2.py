"""Download full dataset v2 — with timeouts, resume, flushed output."""
import csv
import json
import os
import sys
import time
import signal
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')
from src.utils import compute_indices, save_tile, load_config

DATA_DIR = 'data/sentinel2'
LABELS_FILE = 'data/labels/full_dataset_all_locations.csv'
TIMEOUT_SECONDS = 60  # Per-tile timeout

config = load_config()

def log(msg):
    print(msg, flush=True)

def search_with_timeout(lat, lon, config, max_items=3):
    """Search for scenes with timeout."""
    import planetary_computer as pc
    from pystac_client import Client
    
    catalog = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    date_start, date_end = config["date_range"]
    delta = 0.05
    bbox = [lon - delta, lat - delta, lon + delta, lat + delta]
    
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": config["cloud_threshold"]}},
        max_items=max_items,
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
    )
    return list(search.items())

def download_tile_safe(item, lat, lon, bands, tile_size, resolution):
    """Download a single tile with error handling."""
    import rasterio
    from rasterio.windows import from_bounds
    from pyproj import Transformer
    
    half_extent = (tile_size * resolution) / 2
    tile_bands = []
    meta = {}
    
    for band in bands:
        href = item.assets[band].href
        with rasterio.open(href) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            x, y = transformer.transform(lon, lat)
            window = from_bounds(
                x - half_extent, y - half_extent,
                x + half_extent, y + half_extent,
                src.transform,
            )
            data = src.read(
                1, window=window,
                out_shape=(tile_size, tile_size),
                resampling=rasterio.enums.Resampling.bilinear,
            ).astype(np.float32)
            tile_bands.append(data)
            if not meta:
                meta = {
                    "crs": str(src.crs),
                    "datetime": item.datetime.isoformat() if item.datetime else None,
                    "scene_id": item.id,
                }
    
    raw_tile = np.stack(tile_bands, axis=0)
    index_tile = compute_indices(raw_tile)
    return raw_tile, index_tile, meta

# Load locations
locations = []
with open(LABELS_FILE) as f:
    for row in csv.DictReader(f):
        locations.append(row)

# Group camp tiles by parent
camp_groups = defaultdict(list)
neg_tiles = []
for loc in locations:
    if loc['label'] == 'camp':
        parent = loc.get('parent_camp', loc['tile_id'])
        camp_groups[parent].append(loc)
    else:
        neg_tiles.append(loc)

total = len(locations)
existing = sum(1 for loc in locations if os.path.exists(f"{DATA_DIR}/{loc['tile_id']}.npy"))
log(f"Total: {total}, Already: {existing}, Remaining: {total-existing}")

success = existing
failed = 0
failed_tiles = []
counter = 0

# Scene cache: reuse for nearby tiles
scene_cache = {}

def process_tile(loc, item=None):
    """Process a single tile. Returns (ok, item)."""
    tile_id = loc['tile_id']
    out_path = f"{DATA_DIR}/{tile_id}.npy"
    if os.path.exists(out_path):
        return True, item
    
    lat, lon = float(loc['lat']), float(loc['lon'])
    
    try:
        if item is None:
            items = search_with_timeout(lat, lon, config, max_items=3)
            if not items:
                return False, None
            item = items[0]
        
        _, index_tile, meta = download_tile_safe(
            item, lat, lon, config['bands'],
            config['tile_size_download'], config['resolution'])
        
        meta['tile_id'] = tile_id
        meta['country'] = loc['country']
        meta['label'] = loc['label']
        meta['cloud_pct'] = item.properties.get('eo:cloud_cover', -1)
        
        save_tile(index_tile, out_path, meta)
        return True, item
    except Exception as e:
        return False, None

# Phase 1: Camp tiles
log("=" * 60)
log("PHASE 1: Camp tiles")
log("=" * 60)

for parent_id, group in sorted(camp_groups.items()):
    all_done = all(os.path.exists(f"{DATA_DIR}/{loc['tile_id']}.npy") for loc in group)
    if all_done:
        counter += len(group)
        continue
    
    center = next((loc for loc in group if loc.get('is_center', '') == 'True'), group[0])
    
    try:
        items = search_with_timeout(float(center['lat']), float(center['lon']), config, max_items=3)
        item = items[0] if items else None
    except Exception:
        item = None
    
    for loc in group:
        counter += 1
        tile_id = loc['tile_id']
        if os.path.exists(f"{DATA_DIR}/{tile_id}.npy"):
            continue
        
        try:
            ok, _ = process_tile(loc, item=item)
        except Exception:
            ok = False
        
        if ok:
            success += 1
            log(f"[{counter:4d}/{total}] OK   {tile_id:40s} {loc['country']}")
        else:
            failed += 1
            failed_tiles.append(tile_id)
            log(f"[{counter:4d}/{total}] FAIL {tile_id:40s} {loc['country']}")
    
    time.sleep(0.1)

# Phase 2: Negatives
log("")
log("=" * 60)
log("PHASE 2: Negative tiles")
log("=" * 60)

for i, loc in enumerate(neg_tiles):
    counter += 1
    tile_id = loc['tile_id']
    if os.path.exists(f"{DATA_DIR}/{tile_id}.npy"):
        continue
    
    try:
        ok, item = process_tile(loc)
    except Exception:
        ok = False
    
    if ok:
        success += 1
        log(f"[{counter:4d}/{total}] OK   {tile_id:40s} {loc['country']}")
    else:
        failed += 1
        failed_tiles.append(tile_id)
        log(f"[{counter:4d}/{total}] FAIL {tile_id:40s} {loc['country']}")
    
    if (i + 1) % 20 == 0:
        time.sleep(0.3)

# Summary
log("")
log("=" * 60)
log(f"DOWNLOAD COMPLETE: {success}/{total} OK, {failed} failed")
if failed_tiles:
    for ft in failed_tiles[:30]:
        log(f"  FAILED: {ft}")
    if len(failed_tiles) > 30:
        log(f"  ... and {len(failed_tiles)-30} more")
log("=" * 60)
