import numpy as np

from src.utils import compute_indices, generate_grid_tiles


def test_compute_indices_simple():
    raw = np.zeros((6, 2, 2), dtype=np.float32)
    # Order: B02, B03, B04, B08, B11, B12
    raw[2] = 1.0  # Red
    raw[3] = 3.0  # NIR
    raw[4] = 2.0  # SWIR1
    raw[5] = 4.0  # SWIR2

    out = compute_indices(raw)
    assert out.shape == (6, 2, 2)

    ndvi = out[3]
    ndbi = out[4]
    swir_ratio = out[5]

    assert np.allclose(ndvi, 0.5)
    assert np.allclose(ndbi, -0.2)
    assert np.allclose(swir_ratio, 2.0)


def test_generate_grid_tiles():
    tiles = generate_grid_tiles(0.0, 0.0, grid_size=3, tile_size=128, resolution=10)
    assert len(tiles) == 9
    rows = {t['grid_row'] for t in tiles}
    cols = {t['grid_col'] for t in tiles}
    assert rows == {-1, 0, 1}
    assert cols == {-1, 0, 1}
    center = [t for t in tiles if t['grid_row'] == 0 and t['grid_col'] == 0]
    assert len(center) == 1
