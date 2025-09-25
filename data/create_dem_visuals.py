import os
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from data_collection.slope_data_collector import SlopeDataCollector


def save_3d_and_contours(lat: float, lon: float, out_dir: str = "temp_dem", half_side_m: int = 200) -> None:
    Path(out_dir).mkdir(exist_ok=True, parents=True)

    collector = SlopeDataCollector()

    # Build interactive 3D figure data (but we will save static PNG via matplotlib style)
    fig3d = collector.plot_terrain_3d(lat=lat, lon=lon, output_dir=out_dir, half_side_m=half_side_m, save_plots=True)

    # Also generate a 2D contour map from the same DEM window
    # Reuse the PNG saved by plot_terrain_3d inputs to reconstruct arrays
    # Instead, load a fresh cropped DEM and create contours directly.
    # Download a small DEM window and reuse the crop logic
    region = collector._offset(lat, lon, metres=max(2_000, half_side_m * 4))
    dem_path = Path(out_dir) / f"dem_contour_{lat:.5f}_{lon:.5f}.tif"
    if not collector._download_elevation_data(region, str(dem_path), lat, lon):
        raise RuntimeError("Failed to download DEM for contour map")

    try:
        # Read DEM with GDAL if available, else fallbacks handled in collector helper
        dem = None
        gt = None
        try:
            from osgeo import gdal  # type: ignore
            ds = gdal.Open(str(dem_path))
            dem = ds.ReadAsArray()
            gt = ds.GetGeoTransform()
        except Exception:
            # Fallback to PIL/approx geotransform
            from PIL import Image
            img = Image.open(str(dem_path))
            dem = np.array(img)
            px_m = 10.0
            deg_per_m_lat = 1.0 / 110540.0
            deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(lat)) + 1e-9)
            xres = px_m * deg_per_m_lon
            yres = -px_m * deg_per_m_lat
            minx = lon - (dem.shape[1] * xres) / 2.0
            maxy = lat - (dem.shape[0] * yres) / 2.0
            gt = (minx, xres, 0.0, maxy, 0.0, yres)

        h, w = dem.shape
        ys, xs = collector._crop_window(gt, w, h, lat, lon, half_side_m=half_side_m)
        dem_c = dem[ys, xs]

        # 2D contour plot
        fig, ax = plt.subplots(figsize=(8, 6))
        levels = 15
        cs = ax.contour(dem_c, levels=levels, cmap='terrain')
        ax.clabel(cs, inline=True, fontsize=8)
        ax.set_title(f"2D Contour Map – lat {lat:.5f}, lon {lon:.5f}")
        ax.set_xlabel('pixel x')
        ax.set_ylabel('pixel y')
        out_png = Path(out_dir) / f"contour_{lat:.5f}_{lon:.5f}.png"
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
        plt.close(fig)
    finally:
        try:
            if dem_path.exists():
                dem_path.unlink()
        except Exception:
            pass


def main():
    # Default to application default location if not provided
    lat = 32.82426656
    lon = -117.23500108
    if len(sys.argv) >= 3:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    out_dir = "temp_dem"
    save_3d_and_contours(lat, lon, out_dir=out_dir, half_side_m=200)
    print(f"Saved 3D and contour PNGs in {out_dir}")


if __name__ == "__main__":
    main()


