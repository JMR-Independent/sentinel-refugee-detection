"""Dataset classes and data preparation utilities."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CampTileDataset(Dataset):
    """PyTorch Dataset for camp/non-camp classification tiles.

    Expects a manifest CSV with columns: tile_id, path, label, country, split.
    Tiles are stored as .npy files of shape (n_channels, H, W).

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest CSV.
    split : str
        One of 'train', 'val', 'test'.
    transform : callable, optional
        Augmentation function applied to the numpy array.
    normalize : bool
        Whether to apply percentile normalization.
    norm_stats : dict, optional
        Pre-computed stats with 'low' and 'high' arrays per channel.
    model_size : int
        Resize tiles to this size (default 64).
    """

    LABEL_MAP = {"camp": 1, "non-camp": 0, "negative": 0}

    def __init__(self, manifest_path, split, transform=None,
                 normalize=True, norm_stats=None, model_size=64):
        self.manifest = pd.read_csv(manifest_path)
        self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.normalize = normalize
        self.norm_stats = norm_stats
        self.model_size = model_size

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        tile = np.load(row["path"]).astype(np.float32)  # (C, H, W)
        label = self.LABEL_MAP.get(row["label"], 0)

        if self.normalize and self.norm_stats is not None:
            tile = normalize_tile(tile, self.norm_stats)

        # Resize from download size (128) to model size (64) if needed
        if tile.shape[1] != self.model_size or tile.shape[2] != self.model_size:
            tile = resize_tile(tile, self.model_size)

        if self.transform is not None:
            tile = self.transform(tile)

        return torch.from_numpy(tile.copy()), torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def resize_tile(tile, target_size):
    """Resize a tile using area averaging (downscale) or bilinear (upscale).

    Parameters
    ----------
    tile : np.ndarray
        Shape (C, H, W).
    target_size : int
        Target spatial dimension.

    Returns
    -------
    np.ndarray
        Shape (C, target_size, target_size).
    """
    c, h, w = tile.shape
    if h == target_size and w == target_size:
        return tile

    # Simple block averaging for 2x downscale (128->64)
    if h == target_size * 2 and w == target_size * 2:
        return (tile[:, 0::2, 0::2] + tile[:, 1::2, 0::2] +
                tile[:, 0::2, 1::2] + tile[:, 1::2, 1::2]) / 4.0

    # General case: use torch interpolate
    t = torch.from_numpy(tile).unsqueeze(0)  # (1, C, H, W)
    t = torch.nn.functional.interpolate(
        t, size=(target_size, target_size), mode="bilinear", align_corners=False
    )
    return t.squeeze(0).numpy()


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_norm_stats(manifest_path, split="train", low_pct=2, high_pct=98):
    """Compute per-channel percentile statistics from training tiles.

    Parameters
    ----------
    manifest_path : str or Path
        Path to manifest CSV.
    split : str
        Split to compute stats from (usually 'train').
    low_pct, high_pct : float
        Percentiles for clipping.

    Returns
    -------
    dict
        'low': array of shape (n_channels,), 'high': array of shape (n_channels,).
    """
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"] == split]

    all_values = []
    for _, row in manifest.iterrows():
        tile = np.load(row["path"]).astype(np.float32)
        all_values.append(tile.reshape(tile.shape[0], -1))

    all_values = np.concatenate(all_values, axis=1)

    low = np.percentile(all_values, low_pct, axis=1)
    high = np.percentile(all_values, high_pct, axis=1)

    return {"low": low, "high": high}


def normalize_tile(tile, stats):
    """Apply percentile normalization to a tile.

    Parameters
    ----------
    tile : np.ndarray
        Shape (C, H, W).
    stats : dict
        With 'low' and 'high' arrays of shape (C,).

    Returns
    -------
    np.ndarray
        Normalized to [0, 1].
    """
    low = stats["low"][:, None, None]
    high = stats["high"][:, None, None]

    denom = high - low
    denom[denom == 0] = 1.0

    tile = (tile - low) / denom
    return np.clip(tile, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Augmentations (for satellite imagery)
# ---------------------------------------------------------------------------

class SatelliteAugmentation:
    """Random augmentations suitable for overhead satellite imagery.

    Parameters
    ----------
    rotation : bool
        Random 90-degree rotations.
    flip_h : bool
        Random horizontal flip.
    flip_v : bool
        Random vertical flip.
    brightness : float
        Maximum brightness jitter (fraction).
    """

    def __init__(self, rotation=True, flip_h=True, flip_v=True, brightness=0.1):
        self.rotation = rotation
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.brightness = brightness

    def __call__(self, tile):
        """Apply augmentations to a tile of shape (C, H, W)."""
        if self.rotation:
            k = np.random.randint(0, 4)
            tile = np.rot90(tile, k, axes=(1, 2))

        if self.flip_h and np.random.random() > 0.5:
            tile = tile[:, :, ::-1]

        if self.flip_v and np.random.random() > 0.5:
            tile = tile[:, ::-1, :]

        if self.brightness > 0:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            tile = tile * factor
            tile = np.clip(tile, 0.0, 1.0)

        return tile


# ---------------------------------------------------------------------------
# Manifest creation with scene-level splitting
# ---------------------------------------------------------------------------

def create_manifest(tiles_dir, labels_df, output_path, train_countries,
                    test_countries, val_fraction=0.2, by_scene=True, seed=42):
    """Create a manifest CSV with scene-aware train/val/test split.

    Scene-level splitting ensures tiles from the same Sentinel-2 scene
    never appear in both train and val, preventing data leakage.

    Parameters
    ----------
    tiles_dir : str or Path
        Directory containing .npy tile files.
    labels_df : pd.DataFrame
        With columns: tile_id, label, country, (optionally scene_id).
    output_path : str or Path
        Where to save the manifest CSV.
    train_countries : list of str
        Countries for train/val split.
    test_countries : list of str
        Countries for test split.
    val_fraction : float
        Fraction of training scenes to use for validation.
    by_scene : bool
        If True, split by scene_id to prevent leakage.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        The manifest with columns: tile_id, path, label, country, split, scene_id.
    """
    tiles_dir = Path(tiles_dir)
    rng = np.random.default_rng(seed)

    rows = []
    for _, row in labels_df.iterrows():
        tile_path = tiles_dir / f"{row['tile_id']}.npy"
        if not tile_path.exists():
            continue

        country = str(row["country"]).lower()
        if country in [c.lower() for c in test_countries]:
            split = "test"
        elif country in [c.lower() for c in train_countries]:
            split = "train"
        else:
            continue

        rows.append({
            "tile_id": row["tile_id"],
            "path": str(tile_path),
            "label": row["label"],
            "country": country,
            "split": split,
            "scene_id": row.get("scene_id", "unknown"),
            "neg_category": row.get("neg_category", ""),
            "parent_camp": row.get("parent_camp", ""),
        })

    manifest = pd.DataFrame(rows)

    # Split train into train/val
    train_mask = manifest["split"] == "train"

    if by_scene and "scene_id" in manifest.columns:
        # Scene-level split: all tiles from same scene go to same split
        train_scenes = manifest[train_mask]["scene_id"].unique()
        rng.shuffle(train_scenes)
        n_val_scenes = max(1, int(len(train_scenes) * val_fraction))
        val_scenes = set(train_scenes[:n_val_scenes])

        val_mask = train_mask & manifest["scene_id"].isin(val_scenes)
        manifest.loc[val_mask, "split"] = "val"
        print(f"Scene-level split: {len(train_scenes)} scenes -> "
              f"{len(train_scenes) - n_val_scenes} train, {n_val_scenes} val")
    else:
        # Fallback: random row-level split
        train_indices = manifest[train_mask].index.values
        rng.shuffle(train_indices)
        n_val = int(len(train_indices) * val_fraction)
        manifest.loc[train_indices[:n_val], "split"] = "val"

    manifest.to_csv(output_path, index=False)
    print(f"Manifest saved to {output_path}")
    for split in ["train", "val", "test"]:
        subset = manifest[manifest["split"] == split]
        print(f"  {split}: {len(subset)} tiles "
              f"({subset['label'].value_counts().to_dict()})")

    return manifest
