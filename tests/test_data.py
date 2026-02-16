import numpy as np
import pandas as pd

from src.data import compute_norm_stats, create_manifest


def test_compute_norm_stats_sampling(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    # Two tiles with known ranges
    t1 = np.ones((6, 4, 4), dtype=np.float32)
    t2 = np.ones((6, 4, 4), dtype=np.float32) * 10

    np.save(tiles_dir / "tile1.npy", t1)
    np.save(tiles_dir / "tile2.npy", t2)

    manifest = pd.DataFrame({
        "tile_id": ["tile1", "tile2"],
        "path": [str(tiles_dir / "tile1.npy"), str(tiles_dir / "tile2.npy")],
        "label": ["camp", "non-camp"],
        "country": ["syria", "syria"],
        "split": ["train", "train"],
    })
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    stats = compute_norm_stats(
        manifest_path=manifest_path,
        split="train",
        low_pct=2,
        high_pct=98,
        max_tiles=1,
        sample_pixels=4,
        seed=0,
    )

    assert stats["low"].shape == (6,)
    assert stats["high"].shape == (6,)
    assert np.all(stats["high"] >= stats["low"])


def test_create_manifest_unknown_scene(tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    np.save(tiles_dir / "t1.npy", np.zeros((6, 4, 4), dtype=np.float32))
    np.save(tiles_dir / "t2.npy", np.zeros((6, 4, 4), dtype=np.float32))

    labels = pd.DataFrame({
        "tile_id": ["t1", "t2"],
        "label": ["camp", "non-camp"],
        "country": ["syria", "syria"],
        # scene_id omitted to simulate unknown
    })

    out = create_manifest(
        tiles_dir=tiles_dir,
        labels_df=labels,
        output_path=tmp_path / "manifest.csv",
        train_countries=["syria"],
        test_countries=[],
        val_fraction=0.5,
        by_scene=True,
        seed=0,
    )

    assert set(out["split"]).issubset({"train", "val"})
    assert (out["split"] == "val").sum() == 1
    assert (out["split"] == "train").sum() == 1
