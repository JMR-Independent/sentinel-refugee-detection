"""Download full dataset from Microsoft Planetary Computer.

Groups camp tiles by parent to reuse scene searches.
Skips already-downloaded tiles. Resumes gracefully.
"""
import csv
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

sys.path.insert(0, '.')
from src.utils import search_sentinel2, download_tile, compute_indices, save_tile, load_config

DATA_DIR = 'data/sentinel2'
LABELS_FILE = 'data/labels/full_dataset_all_locations.csv'
os.makedirs(DATA_DIR, exist_ok=True)

config = load_config()

# Load all locations
locations = []
with open(LABELS_FILE) as f:
    for row in csv.DictReader(f):
        locations.append(row)

# Separate into camp tiles (grouped by parent) and negatives
camp_groups = defaultdict(list)
neg_tiles = []
for loc in locations:
    if loc['label'] == 'camp':
        parent = loc.get('parent_camp', loc['tile_id'])
        camp_groups[parent].append(loc)
    else:
        neg_tiles.append(loc)

total = len(locations)
already = sum(1 for loc in locations if os.path.exists(f"{DATA_DIR}/{loc['tile_id']}.npy"))
print(f"Total tiles: {total}")
print(f"Already downloaded: {already}")
print(f"To download: {total - already}")
print(f"Camp groups: {len(camp_groups)} (with {sum(len(v) for v in camp_groups.values())} tiles)")
print(f"Negative tiles: {len(neg_tiles)}")
print()

success = already
failed = 0
failed_tiles = []

def download_single(loc, item=None):
    """Download a single tile. Returns (success, item_used)."""
    tile_id = loc['tile_id']
    out_path = f"{DATA_DIR}/{tile_id}.npy"
    
    if os.path.exists(out_path):
        return True, item
    
    lat = float(loc['lat'])
    lon = float(loc['lon'])
    country = loc['country']
    
    try:
        # Search for scene if not provided
        if item is None:
            items = search_sentinel2(lat, lon, config, max_items=3)
            if not items:
                raise ValueError("No Sentinel-2 scenes found")
            item = items[0]
        
        # Download
        raw_tile, index_tile, meta = download_tile(
            item, lat, lon, config['bands'],
            tile_size=config['tile_size_download'],
            resolution=config['resolution']
        )
        
        # Add metadata
        meta['tile_id'] = tile_id
        meta['country'] = country
        meta['label'] = loc['label']
        meta['cloud_pct'] = item.properties.get('eo:cloud_cover', -1)
        
        # Save index tile (R,G,B,NDVI,NDBI,SWIR_ratio)
        save_tile(index_tile, out_path, meta)
        
        return True, item
        
    except Exception as e:
        return False, None

# ============================================================
# Phase 1: Camp tiles (grouped by parent, share scenes)
# ============================================================
print("=" * 60)
print("PHASE 1: Camp tiles (grouped by parent camp)")
print("=" * 60)

counter = 0
for parent_id, group in sorted(camp_groups.items()):
    # Check if all already downloaded
    all_done = all(os.path.exists(f"{DATA_DIR}/{loc['tile_id']}.npy") for loc in group)
    if all_done:
        counter += len(group)
        continue
    
    # Search scene once for the center tile
    center = next((loc for loc in group if loc.get('is_center', '') == 'True'), group[0])
    lat = float(center['lat'])
    lon = float(center['lon'])
    
    try:
        items = search_sentinel2(lat, lon, config, max_items=3)
        if not items:
            for loc in group:
                if not os.path.exists(f"{DATA_DIR}/{loc['tile_id']}.npy"):
                    failed += 1
                    failed_tiles.append(f"{loc['tile_id']}: No scenes found")
            counter += len(group)
            continue
        item = items[0]
    except Exception as e:
        for loc in group:
            if not os.path.exists(f"{DATA_DIR}/{loc['tile_id']}.npy"):
                failed += 1
                failed_tiles.append(f"{loc['tile_id']}: Scene search failed: {e}")
        counter += len(group)
        continue
    
    cloud = item.properties.get('eo:cloud_cover', -1)
    
    for loc in group:
        counter += 1
        tile_id = loc['tile_id']
        
        if os.path.exists(f"{DATA_DIR}/{tile_id}.npy"):
            continue
        
        ok, _ = download_single(loc, item=item)
        if ok:
            success += 1
            status = "OK"
        else:
            failed += 1
            failed_tiles.append(f"{tile_id}: Download failed")
            status = "FAIL"
        
        print(f"[{counter:4d}/{total}] {status} {tile_id:45s} {loc['country']:15s} cloud={cloud:.1f}%")
    
    time.sleep(0.2)  # Rate limiting between camps

# ============================================================
# Phase 2: Negative tiles (each needs own scene search)
# ============================================================
print()
print("=" * 60)
print("PHASE 2: Negative tiles")
print("=" * 60)

for i, loc in enumerate(neg_tiles):
    counter += 1
    tile_id = loc['tile_id']
    
    if os.path.exists(f"{DATA_DIR}/{tile_id}.npy"):
        continue
    
    ok, item = download_single(loc)
    if ok:
        success += 1
        cloud = item.properties.get('eo:cloud_cover', -1) if item else -1
        status = "OK"
    else:
        failed += 1
        failed_tiles.append(f"{tile_id}: Download failed")
        cloud = -1
        status = "FAIL"
    
    print(f"[{counter:4d}/{total}] {status} {tile_id:45s} {loc['country']:15s} cloud={cloud:.1f}%")
    
    if (i + 1) % 10 == 0:
        time.sleep(0.5)  # Rate limiting

# ============================================================
# Summary
# ============================================================
print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print(f"  Success: {success}/{total}")
print(f"  Failed:  {failed}")
if failed_tiles:
    for ft in failed_tiles:
        print(f"    {ft}")
print("=" * 60)
