"""Model definitions for camp detection."""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def create_camp_classifier(config):
    """Create a ResNet-18 binary classifier for 6-band satellite tiles.

    Modifies a pretrained ResNet-18:
    - First conv layer: 3 channels -> 6 channels (duplicated RGB weights)
    - Final FC layer: 512 -> 1 (binary, sigmoid output)

    Parameters
    ----------
    config : dict
        Model configuration with keys: backbone, pretrained, in_channels.

    Returns
    -------
    nn.Module
        The modified ResNet-18 model.
    """
    in_channels = config["model"]["in_channels"]
    pretrained = config["model"]["pretrained"]

    # Load pretrained ResNet-18
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # Modify first conv layer for 6 input channels
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels, 64,
        kernel_size=7, stride=2, padding=3, bias=False,
    )

    if pretrained:
        # Initialize: copy RGB weights, duplicate for extra bands
        with torch.no_grad():
            # First 3 channels get the original RGB weights
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Extra channels get averaged RGB weights, scaled down
            rgb_mean = old_conv.weight.mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i:i+1, :, :] = rgb_mean * 0.5

    model.conv1 = new_conv

    # Replace final FC: 512 -> 1 (binary classification)
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model


def freeze_backbone(model):
    """Freeze all layers except the final FC and layer4.

    Parameters
    ----------
    model : nn.Module
        ResNet-18 model.
    """
    for name, param in model.named_parameters():
        if "fc" not in name and "layer4" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters.

    Parameters
    ----------
    model : nn.Module
        ResNet-18 model.
    """
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def extract_band_features(tiles, labels):
    """Extract simple statistical features from tiles for baseline models.

    For each tile, computes per-band: mean, std, median (18 features for 6 bands).

    Parameters
    ----------
    tiles : list of np.ndarray
        Each of shape (C, H, W).
    labels : list of int
        Binary labels.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, n_features).
    np.ndarray
        Label array.
    """
    features = []
    for tile in tiles:
        band_features = []
        for c in range(tile.shape[0]):
            band = tile[c]
            band_features.extend([
                band.mean(),
                band.std(),
                np.median(band),
            ])
        features.append(band_features)

    return np.array(features), np.array(labels)


def compute_ndvi(tile, nir_idx=3, red_idx=2):
    """Compute NDVI from a multi-band tile.

    NDVI = (NIR - Red) / (NIR + Red)

    Parameters
    ----------
    tile : np.ndarray
        Shape (C, H, W).
    nir_idx : int
        Index of NIR band (default 3 = B08).
    red_idx : int
        Index of Red band (default 2 = B04).

    Returns
    -------
    np.ndarray
        NDVI image of shape (H, W), values in [-1, 1].
    """
    nir = tile[nir_idx].astype(np.float32)
    red = tile[red_idx].astype(np.float32)
    denom = nir + red
    denom[denom == 0] = 1e-10
    return (nir - red) / denom


def ndvi_threshold_classifier(tiles, threshold=-0.1):
    """Simple NDVI-based classifier.

    Camps tend to have low NDVI (not vegetation). Predicts 'camp' if
    mean NDVI is below threshold.

    Parameters
    ----------
    tiles : list of np.ndarray
        Each of shape (C, H, W).
    threshold : float
        NDVI threshold below which a tile is classified as camp.

    Returns
    -------
    np.ndarray
        Binary predictions.
    np.ndarray
        Mean NDVI values (for ROC analysis).
    """
    predictions = []
    ndvi_means = []
    for tile in tiles:
        ndvi = compute_ndvi(tile)
        mean_ndvi = ndvi.mean()
        ndvi_means.append(mean_ndvi)
        predictions.append(1 if mean_ndvi < threshold else 0)

    return np.array(predictions), np.array(ndvi_means)
