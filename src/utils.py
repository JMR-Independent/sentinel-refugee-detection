"""Utility functions for data download and geospatial operations."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml


def load_config(config_path="configs/default.yaml"):
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# OpenStreetMap / Overpass API
# ---------------------------------------------------------------------------

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Bounding boxes [south, west, north, east] for each country
COUNTRY_BOUNDS = {
    "syria": (32.31, 35.72, 37.32, 42.38),
    "south_sudan": (3.49, 23.44, 12.24, 35.95),
    "chad": (7.44, 13.47, 23.45, 24.00),
    "ethiopia": (3.40, 32.99, 14.89, 47.99),
    "yemen": (12.11, 41.81, 19.00, 54.68),
}

# Known cities for urban negative samples (lat, lon)
URBAN_CENTERS = {
    "syria": [(33.51, 36.29), (36.20, 37.15)],          # Damascus, Aleppo
    "south_sudan": [(4.85, 31.61), (6.20, 31.58)],      # Juba, Malakal
    "chad": [(12.11, 15.04), (9.14, 18.39)],            # N'Djamena, Sarh
    "ethiopia": [(9.02, 38.75), (7.68, 36.83)],         # Addis Ababa, Jimma
    "yemen": [(15.35, 44.21), (12.78, 45.04)],          # Sana'a, Aden
}

# Known informal settlement areas (NOT refugee camps) — hard negatives
# These are dense, unplanned neighborhoods that might look like camps
INFORMAL_AREAS = {
    "syria": [(33.48, 36.32), (36.18, 37.12)],          # Damascus/Aleppo outskirts
    "south_sudan": [(4.83, 31.59), (4.87, 31.63)],      # Juba informal areas
    "chad": [(12.08, 15.06)],                            # N'Djamena periphery
    "ethiopia": [(8.98, 38.72), (9.05, 38.80)],         # Addis Ababa outskirts
    "yemen": [(15.33, 44.19), (12.80, 45.06)],          # Sana'a/Aden edges
}


def query_osm_camps(country, bounds=None):
    """Query OpenStreetMap for refugee camp locations via Overpass API.

    Parameters
    ----------
    country : str
        Country name (key in COUNTRY_BOUNDS).
    bounds : tuple, optional
        (south, west, north, east) override.

    Returns
    -------
    list of dict
        Each dict has 'lat', 'lon', 'name', 'source'.
    """
    if bounds is None:
        bounds = COUNTRY_BOUNDS[country]
    s, w, n, e = bounds

    query = f"""
    [out:json][timeout:120];
    (
      node["place"="camp"]({s},{w},{n},{e});
      node["refugee"="yes"]({s},{w},{n},{e});
      node["camp:type"="refugee"]({s},{w},{n},{e});
      node["social_facility"="refugee_camp"]({s},{w},{n},{e});
      node["amenity"="refugee_camp"]({s},{w},{n},{e});
      node["landuse"="refugee_camp"]({s},{w},{n},{e});
      way["place"="camp"]({s},{w},{n},{e});
      way["refugee"="yes"]({s},{w},{n},{e});
      way["camp:type"="refugee"]({s},{w},{n},{e});
      way["social_facility"="refugee_camp"]({s},{w},{n},{e});
      way["amenity"="refugee_camp"]({s},{w},{n},{e});
      way["landuse"="refugee_camp"]({s},{w},{n},{e});
      relation["refugee"="yes"]({s},{w},{n},{e});
      relation["boundary"="refugee_camp"]({s},{w},{n},{e});
    );
    out center;
    """

    resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    camps = []
    for element in data.get("elements", []):
        if element["type"] == "node":
            lat, lon = element["lat"], element["lon"]
        elif "center" in element:
            lat, lon = element["center"]["lat"], element["center"]["lon"]
        else:
            continue

        name = element.get("tags", {}).get("name", "unnamed")
        camps.append({
            "lat": lat,
            "lon": lon,
            "name": name,
            "country": country,
            "source": "osm",
        })

    return camps


def query_osm_camps_all(countries=None, delay=2.0):
    """Query OSM camps for multiple countries with rate limiting.

    Parameters
    ----------
    countries : list of str, optional
        Country names. Defaults to all in COUNTRY_BOUNDS.
    delay : float
        Seconds to wait between API calls.

    Returns
    -------
    pd.DataFrame
        Columns: lat, lon, name, country, source.
    """
    if countries is None:
        countries = list(COUNTRY_BOUNDS.keys())

    all_camps = []
    for country in countries:
        print(f"Querying OSM for {country}...")
        camps = query_osm_camps(country)
        print(f"  Found {len(camps)} features")
        all_camps.extend(camps)
        time.sleep(delay)

    return pd.DataFrame(all_camps)


# ---------------------------------------------------------------------------
# UNHCR data
# ---------------------------------------------------------------------------

def load_unhcr_camps(csv_path):
    """Load UNHCR camp locations from a downloaded CSV file.

    The CSV should have at minimum: latitude, longitude, name, country columns.
    Download from https://data.unhcr.org/

    Parameters
    ----------
    csv_path : str or Path
        Path to the UNHCR CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: lat, lon, name, country, source.
    """
    df = pd.read_csv(csv_path)

    lat_col = next(c for c in df.columns if c.lower() in ("latitude", "lat", "y"))
    lon_col = next(c for c in df.columns if c.lower() in ("longitude", "lon", "long", "x"))
    name_col = next((c for c in df.columns if c.lower() in ("name", "site_name", "location_name")), None)
    country_col = next((c for c in df.columns if c.lower() in ("country", "country_name")), None)

    result = pd.DataFrame({
        "lat": df[lat_col].astype(float),
        "lon": df[lon_col].astype(float),
        "name": df[name_col] if name_col else "unnamed",
        "country": df[country_col].str.lower() if country_col else "unknown",
        "source": "unhcr",
    })

    return result.dropna(subset=["lat", "lon"])


# ---------------------------------------------------------------------------
# Merge and deduplicate
# ---------------------------------------------------------------------------

def merge_and_deduplicate(dfs, buffer_km=1.0):
    """Merge multiple camp DataFrames and remove near-duplicates.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        Each with columns: lat, lon, name, country, source.
    buffer_km : float
        Minimum distance in km between unique camps.

    Returns
    -------
    pd.DataFrame
        Deduplicated camp locations.
    """
    merged = pd.concat(dfs, ignore_index=True)
    if len(merged) == 0:
        return merged

    keep = np.ones(len(merged), dtype=bool)
    coords = merged[["lat", "lon"]].values

    for i in range(len(coords)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(coords)):
            if not keep[j]:
                continue
            dist = _haversine_km(coords[i, 0], coords[i, 1],
                                 coords[j, 0], coords[j, 1])
            if dist < buffer_km:
                keep[j] = False

    result = merged[keep].reset_index(drop=True)
    print(f"Deduplicated: {len(merged)} -> {len(result)} locations "
          f"(buffer={buffer_km} km)")
    return result


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers between two points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Grid tiling: multiple tiles per camp location
# ---------------------------------------------------------------------------

def generate_grid_tiles(center_lat, center_lon, grid_size=3,
                        tile_size=128, resolution=10):
    """Generate a grid of tile centers around a camp location.

    For a 3x3 grid with 128px tiles at 10m, each tile covers 1.28km.
    The grid covers ~3.84km x 3.84km centered on the camp.

    Parameters
    ----------
    center_lat, center_lon : float
        Camp center coordinates.
    grid_size : int
        Grid dimension (3 = 3x3 = 9 tiles).
    tile_size : int
        Tile size in pixels.
    resolution : int
        Meters per pixel.

    Returns
    -------
    list of dict
        Each dict has 'lat', 'lon', 'grid_row', 'grid_col'.
    """
    tile_extent_m = tile_size * resolution  # meters per tile
    half_grid = grid_size // 2

    # Approximate meters -> degrees
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

    tiles = []
    for row in range(-half_grid, half_grid + 1):
        for col in range(-half_grid, half_grid + 1):
            offset_lat = (row * tile_extent_m) / m_per_deg_lat
            offset_lon = (col * tile_extent_m) / m_per_deg_lon
            tiles.append({
                "lat": center_lat + offset_lat,
                "lon": center_lon + offset_lon,
                "grid_row": row,
                "grid_col": col,
            })

    return tiles


def expand_camps_to_grid(camps_df, config):
    """Expand each camp location into a grid of tiles with distance-based labels.

    Since we typically have POINTS (not polygons), labels are assigned by
    Manhattan distance from the center tile:
    - Center (0,0): "camp" — high confidence, point is here
    - Adjacent (dist=1): "camp" — within ~1 tile of point, likely partial coverage
    - Corner (dist=2): EXCLUDED by default (include_corners=false)
      If included, labeled "camp_context" with reduced loss weight.

    Why exclude corners by default:
    - Tile covers 1.28km, buffer is 500m
    - Corner center is ~1.81km from camp point
    - At that distance, <10% of typical camp falls in corner
    - Less noise > more data

    Parameters
    ----------
    camps_df : pd.DataFrame
        Camp locations with lat, lon, tile_id, country.
    config : dict
        Configuration with grid settings.

    Returns
    -------
    pd.DataFrame
        Expanded locations with one row per tile.
    """
    grid_size = config["grid"]["size"]
    tile_size = config["tile_size_download"]
    resolution = config["resolution"]
    include_corners = config["grid"].get("include_corners", False)

    rows = []
    for _, camp in camps_df.iterrows():
        grid_tiles = generate_grid_tiles(
            camp["lat"], camp["lon"],
            grid_size=grid_size,
            tile_size=tile_size,
            resolution=resolution,
        )
        for tile in grid_tiles:
            manhattan_dist = abs(tile["grid_row"]) + abs(tile["grid_col"])
            is_center = (manhattan_dist == 0)

            # Skip corners if configured
            if manhattan_dist >= 2 and not include_corners:
                continue

            # Label based on distance from camp center point
            if manhattan_dist <= 1:
                label = "camp"            # center + direct neighbors
            else:
                label = "camp_context"    # corners — reduced weight in training

            rows.append({
                "lat": tile["lat"],
                "lon": tile["lon"],
                "name": camp.get("name", "unnamed"),
                "country": camp["country"],
                "source": camp.get("source", "unknown"),
                "label": label,
                "tile_id": (f"{camp['tile_id']}_r{tile['grid_row']:+d}"
                            f"c{tile['grid_col']:+d}"),
                "parent_camp": camp["tile_id"],
                "is_center": is_center,
                "grid_row": tile["grid_row"],
                "grid_col": tile["grid_col"],
                "manhattan_dist": manhattan_dist,
            })

    result = pd.DataFrame(rows)
    n_camp = (result["label"] == "camp").sum()
    n_context = (result["label"] == "camp_context").sum()
    tiles_per = len(result) // max(len(camps_df), 1)
    print(f"Expanded {len(camps_df)} camps -> {len(result)} grid tiles "
          f"({tiles_per} per camp, corners={'included' if include_corners else 'excluded'})")
    print(f"  Labels: {n_camp} camp (center+adjacent)"
          + (f", {n_context} camp_context (corners)" if n_context > 0 else ""))
    return result


# ---------------------------------------------------------------------------
# Categorized negative sample generation
# ---------------------------------------------------------------------------

def generate_negatives(camps_df, config, seed=42):
    """Generate categorized negative samples: rural, urban, barren.

    Parameters
    ----------
    camps_df : pd.DataFrame
        Positive camp locations with lat, lon, country.
    config : dict
        Configuration with negative_ratio, negative_buffer_km, negative_categories.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Negative samples with 'neg_category' column.
    """
    rng = np.random.default_rng(seed)
    n_per_camp = config["negative_ratio"]
    min_dist = config["negative_buffer_km"]
    categories = config["negative_categories"]

    all_negatives = []

    for country in camps_df["country"].unique():
        if country not in COUNTRY_BOUNDS:
            print(f"  Skipping {country} (no bounding box)")
            continue

        s, w, n, e = COUNTRY_BOUNDS[country]
        country_camps = camps_df[camps_df["country"] == country]
        camp_coords = country_camps[["lat", "lon"]].values

        for cat in categories:
            n_needed = len(country_camps) * n_per_camp

            if cat == "informal" and country in INFORMAL_AREAS:
                # Hard negatives: informal settlements that are NOT camps
                negs = _sample_near_locations(
                    INFORMAL_AREAS[country], n_needed, camp_coords, min_dist, rng,
                    radius_deg=0.05)  # ~5km radius
            elif cat == "urban" and country in URBAN_CENTERS:
                # Sample near known cities
                negs = _sample_near_cities(
                    country, n_needed, camp_coords, min_dist, rng)
            elif cat == "barren":
                # Sample in arid/low-NDVI zones (edges of bounding box)
                negs = _sample_barren(
                    s, w, n, e, n_needed, camp_coords, min_dist, rng)
            else:
                # Rural: random within country bounds
                negs = _sample_random(
                    s, w, n, e, n_needed, camp_coords, min_dist, rng)

            for i, neg in enumerate(negs):
                all_negatives.append({
                    "lat": neg[0],
                    "lon": neg[1],
                    "name": f"neg_{cat}_{country}_{i:04d}",
                    "country": country,
                    "source": "negative",
                    "label": "non-camp",
                    "neg_category": cat,
                })

        print(f"  {country}: generated {len([x for x in all_negatives if x['country'] == country])} negatives across {len(categories)} categories")

    return pd.DataFrame(all_negatives)


def _sample_random(s, w, n, e, n_needed, camp_coords, min_dist, rng):
    """Sample random points within bounds, far from camps."""
    points = []
    attempts = 0
    while len(points) < n_needed and attempts < n_needed * 50:
        lat = rng.uniform(s, n)
        lon = rng.uniform(w, e)
        if _min_dist_to_camps(lat, lon, camp_coords) >= min_dist:
            points.append((lat, lon))
        attempts += 1
    return points


def _sample_near_cities(country, n_needed, camp_coords, min_dist, rng):
    """Sample points near known urban centers."""
    cities = URBAN_CENTERS.get(country, [])
    if not cities:
        return []

    points = []
    attempts = 0
    while len(points) < n_needed and attempts < n_needed * 50:
        city = cities[rng.integers(len(cities))]
        # Random offset within ~20km of city center
        lat = city[0] + rng.uniform(-0.18, 0.18)
        lon = city[1] + rng.uniform(-0.18, 0.18)
        if _min_dist_to_camps(lat, lon, camp_coords) >= min_dist:
            points.append((lat, lon))
        attempts += 1
    return points


def _sample_near_locations(locations, n_needed, camp_coords, min_dist, rng,
                           radius_deg=0.05):
    """Sample points near specific locations (for hard negatives)."""
    if not locations:
        return []

    points = []
    attempts = 0
    while len(points) < n_needed and attempts < n_needed * 50:
        loc = locations[rng.integers(len(locations))]
        lat = loc[0] + rng.uniform(-radius_deg, radius_deg)
        lon = loc[1] + rng.uniform(-radius_deg, radius_deg)
        if _min_dist_to_camps(lat, lon, camp_coords) >= min_dist:
            points.append((lat, lon))
        attempts += 1
    return points


def _sample_barren(s, w, n, e, n_needed, camp_coords, min_dist, rng):
    """Sample points in likely barren/desert areas (top/bottom of bbox)."""
    points = []
    attempts = 0
    # Bias toward edges of bounding box (more likely desert/barren)
    while len(points) < n_needed and attempts < n_needed * 50:
        if rng.random() > 0.5:
            lat = rng.uniform(s, s + (n - s) * 0.3)  # southern edge
        else:
            lat = rng.uniform(n - (n - s) * 0.3, n)   # northern edge
        lon = rng.uniform(w, e)
        if _min_dist_to_camps(lat, lon, camp_coords) >= min_dist:
            points.append((lat, lon))
        attempts += 1
    return points


def _min_dist_to_camps(lat, lon, camp_coords):
    """Minimum haversine distance from a point to any camp."""
    if len(camp_coords) == 0:
        return float("inf")
    dists = [_haversine_km(lat, lon, c[0], c[1]) for c in camp_coords]
    return min(dists)


# ---------------------------------------------------------------------------
# Derived spectral indices
# ---------------------------------------------------------------------------

def compute_indices(raw_tile):
    """Compute derived spectral indices from raw 6-band Sentinel-2 tile.

    Input bands order: B02(Blue), B03(Green), B04(Red), B08(NIR), B11(SWIR1), B12(SWIR2)
    Output: 6-channel array [R, G, B, NDVI, NDBI, SWIR_ratio]

    Parameters
    ----------
    raw_tile : np.ndarray
        Shape (6, H, W) with raw Sentinel-2 bands.

    Returns
    -------
    np.ndarray
        Shape (6, H, W) with RGB + derived indices.
    """
    blue, green, red, nir, swir1, swir2 = [
        raw_tile[i].astype(np.float32) for i in range(6)
    ]

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = _safe_ratio(nir - red, nir + red)

    # NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)  -- Built-up index
    ndbi = _safe_ratio(swir1 - nir, swir1 + nir)

    # SWIR ratio = SWIR2 / SWIR1
    swir_ratio = _safe_ratio(swir2, swir1)

    return np.stack([red, green, blue, ndvi, ndbi, swir_ratio], axis=0)


def _safe_ratio(numerator, denominator):
    """Division avoiding NaN/inf."""
    denom = denominator.copy()
    denom[denom == 0] = 1e-10
    return numerator / denom


# ---------------------------------------------------------------------------
# Sentinel-2 download via Microsoft Planetary Computer
# ---------------------------------------------------------------------------

def search_sentinel2(lat, lon, config, max_items=5):
    """Search for Sentinel-2 scenes covering a location.

    Parameters
    ----------
    lat, lon : float
        Center coordinates.
    config : dict
        Configuration with 'cloud_threshold' and 'date_range'.
    max_items : int
        Maximum number of scenes to return.

    Returns
    -------
    list
        STAC items sorted by cloud cover (ascending).
    """
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


def download_tile(item, lat, lon, bands, tile_size=128, resolution=10):
    """Download and crop a Sentinel-2 tile centered on a location.

    Downloads raw bands, then computes derived indices (NDVI, NDBI, SWIR_ratio).

    Parameters
    ----------
    item : pystac.Item
        STAC item from search_sentinel2.
    lat, lon : float
        Center coordinates.
    bands : list of str
        Raw band names (e.g., ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']).
    tile_size : int
        Output tile size in pixels (before any resizing).
    resolution : int
        Target resolution in meters.

    Returns
    -------
    np.ndarray
        Raw tile: shape (n_bands, tile_size, tile_size), dtype float32.
    np.ndarray
        Index tile: shape (6, tile_size, tile_size) with RGB + indices.
    dict
        Metadata with 'crs', 'datetime', 'scene_id'.
    """
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
                1,
                window=window,
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


def save_tile(tile_array, output_path, meta=None):
    """Save a multi-band tile as .npy with optional metadata.

    Parameters
    ----------
    tile_array : np.ndarray
        Shape (n_bands, H, W).
    output_path : str or Path
        Output .npy file path.
    meta : dict, optional
        Metadata saved as JSON sidecar.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, tile_array)

    if meta:
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# GeoJSON export
# ---------------------------------------------------------------------------

def df_to_geojson(df, output_path):
    """Save a DataFrame with lat/lon columns as GeoJSON."""
    features = []
    for _, row in df.iterrows():
        props = {}
        for k, v in row.items():
            if k in ("lat", "lon"):
                continue
            # Convert numpy types to Python types for JSON
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = float(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            props[k] = v

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["lon"]), float(row["lat"])],
            },
            "properties": props,
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)

    print(f"Saved {len(features)} features to {output_path}")


def print_dataset_summary(camp_tiles_df, negatives_df, config):
    """Print exact dataset counts. No estimation — hard numbers only.

    Parameters
    ----------
    camp_tiles_df : pd.DataFrame
        Camp tiles (output of expand_camps_to_grid).
    negatives_df : pd.DataFrame
        Negative samples (output of generate_negatives).
    config : dict
        Configuration.
    """
    n_cats = len(config["negative_categories"])
    n_ratio = config["negative_ratio"]

    print("=" * 60)
    print("DATASET SUMMARY (exact counts)")
    print("=" * 60)

    # Positives
    n_unique_camps = camp_tiles_df["parent_camp"].nunique() if "parent_camp" in camp_tiles_df else len(camp_tiles_df)
    n_camp = (camp_tiles_df["label"] == "camp").sum()
    n_context = (camp_tiles_df["label"] == "camp_context").sum()
    print(f"\nPositives:")
    print(f"  Unique camps:        {n_unique_camps}")
    print(f"  Camp tiles:          {n_camp} (center + adjacent)")
    if n_context > 0:
        print(f"  Context tiles:       {n_context} (corners, weight 0.5)")
    print(f"  Total positive tiles: {len(camp_tiles_df)}")

    # Per country
    print(f"\n  By country:")
    for country in sorted(camp_tiles_df["country"].unique()):
        n = (camp_tiles_df["country"] == country).sum()
        print(f"    {country}: {n} tiles")

    # Negatives
    print(f"\nNegatives:")
    print(f"  Total: {len(negatives_df)}")
    if "neg_category" in negatives_df:
        for cat in config["negative_categories"]:
            n = (negatives_df["neg_category"] == cat).sum()
            print(f"    {cat}: {n}")

    print(f"\n  By country:")
    for country in sorted(negatives_df["country"].unique()):
        n = (negatives_df["country"] == country).sum()
        print(f"    {country}: {n}")

    # Ratio
    total_pos = len(camp_tiles_df)
    total_neg = len(negatives_df)
    ratio = total_neg / max(total_pos, 1)
    print(f"\nBalance:")
    print(f"  Positive:Negative = 1:{ratio:.1f}")
    print(f"  Total tiles: {total_pos + total_neg}")

    # Expected formula check
    expected_neg = n_unique_camps * n_ratio * n_cats
    print(f"\n  Expected negatives (formula): {n_unique_camps} camps * "
          f"{n_ratio}/cat * {n_cats} cats = {expected_neg}")
    if abs(total_neg - expected_neg) > expected_neg * 0.1:
        print(f"  WARNING: actual ({total_neg}) differs from expected ({expected_neg})")
    print("=" * 60)
