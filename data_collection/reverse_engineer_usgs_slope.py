"""
Reverse-engineer 'Slope From USGS Elevation Data' from Corrected_Input_Data.csv.

For two rows (default: 0 and 1), download/crop DEM, try multiple slope
computations, and print candidates against CSV values to identify the
likely methodology (e.g., center Horn slope, local 3x3/5x5 max, richdem).

Run (in landslide env):
  python data_collection/reverse_engineer_usgs_slope.py
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.transform import rowcol
    from rasterio.warp import calculate_default_transform, reproject, Resampling
except Exception:
    raise SystemExit("rasterio is required. Activate 'landslide' env and conda install rasterio.")

try:
    import richdem as rd
    RICHDEM_OK = True
except Exception:
    RICHDEM_OK = False
    warnings.warn("richdem not available; will skip richdem variant.")

try:
    import leafmap
    LEAFMAP_OK = True
except Exception:
    LEAFMAP_OK = False
    warnings.warn("leafmap not available; provide local GeoTIFF tiles.")


@dataclass
class SlopeCandidates:
    horn_center: Optional[float]
    horn_local3x3_max: Optional[float]
    horn_local5x5_max: Optional[float]
    richdem_center: Optional[float]
    richdem_no_gt_center: Optional[float]
    richdem_no_gt_local3x3_max: Optional[float]


def latlon_to_utm_epsg(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    south = lat < 0
    return f"EPSG:{32700 + zone}" if south else f"EPSG:{32600 + zone}"


def offset_bbox(lat: float, lon: float, metres: float = 10000) -> List[float]:
    R = 6_378_137.0
    d_lat = metres / R
    d_lon = metres / (R * math.cos(math.radians(lat)))
    return [
        lon - math.degrees(d_lon),
        lat - math.degrees(d_lat),
        lon + math.degrees(d_lon),
        lat + math.degrees(d_lat),
    ]


def download_dem(lat: float, lon: float, metres: int = 20000) -> Optional[str]:
    if not LEAFMAP_OK:
        return None
    region = offset_bbox(lat, lon, metres=metres)
    try:
        paths = leafmap.download_ned(region, out_dir="data")
        if paths:
            return paths[0]
    except Exception as e:
        warnings.warn(f"leafmap download failed: {e}")
    return None


def crop_src(tif_path: str, lat: float, lon: float, half_px: int = 512) -> Tuple[np.ndarray, rasterio.Affine, dict]:
    with rasterio.open(tif_path) as src:
        r, c = rowcol(src.transform, lon, lat)
        H, W = src.height, src.width
        r0, r1 = max(0, r - half_px), min(H, r + half_px)
        c0, c1 = max(0, c - half_px), min(W, c + half_px)
        assert r1 > r0 and c1 > c0, "Empty crop"
        win = Window.from_slices((r0, r1), (c0, c1))
        z = src.read(1, window=win).astype(np.float32)
        transform = src.window_transform(win)
        meta = {"src_r": r, "src_c": c, "r0": r0, "c0": c0}
    return z, transform, meta


def reproject_to_utm(z: np.ndarray, src_transform, src_crs, lat: float, lon: float) -> Tuple[np.ndarray, rasterio.Affine]:
    dst_crs = latlon_to_utm_epsg(lat, lon)
    # 10 m target resolution
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, z.shape[1], z.shape[0],
        *rasterio.transform.array_bounds(z.shape[0], z.shape[1], src_transform),
        resolution=10,
    )
    out = np.empty((dst_height, dst_width), dtype=np.float32)
    reproject(
        source=z,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        num_threads=2,
    )
    return out, dst_transform


def horn_slope_deg(z_m: np.ndarray, cell: float = 10.0) -> np.ndarray:
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32) / (8.0 * cell)
    Ky = np.array([[ 1,  2,  1],
                   [ 0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32) / (8.0 * cell)
    Z = z_m
    Gx = (Kx[0,0]*Z[:-2,:-2] + Kx[0,1]*Z[:-2,1:-1] + Kx[0,2]*Z[:-2,2:] +
          Kx[1,0]*Z[1:-1,:-2] + Kx[1,1]*Z[1:-1,1:-1] + Kx[1,2]*Z[1:-1,2:] +
          Kx[2,0]*Z[2:,:-2] + Kx[2,1]*Z[2:,1:-1] + Kx[2,2]*Z[2:,2:])
    Gy = (Ky[0,0]*Z[:-2,:-2] + Ky[0,1]*Z[:-2,1:-1] + Ky[0,2]*Z[:-2,2:] +
          Ky[1,0]*Z[1:-1,:-2] + Ky[1,1]*Z[1:-1,1:-1] + Ky[1,2]*Z[1:-1,2:] +
          Ky[2,0]*Z[2:,:-2] + Ky[2,1]*Z[2:,1:-1] + Ky[2,2]*Z[2:,2:])
    slope = np.rad2deg(np.arctan(np.sqrt(Gx*Gx + Gy*Gy)))
    return slope


def candidates_for_point(tif_path: str, lat: float, lon: float) -> SlopeCandidates:
    with rasterio.open(tif_path) as src:
        src_crs = src.crs
    z, t_src, meta = crop_src(tif_path, lat, lon, half_px=512)

    z_m, t_m = reproject_to_utm(z, t_src, src_crs, lat, lon)
    horn = horn_slope_deg(z_m, cell=10.0)

    # map target to reprojected grid indices
    inv = ~t_m
    cc = int(inv * (lon, lat))[0]
    rr = int(inv * (lon, lat))[1]
    rr_c = min(max(rr - 1, 0), horn.shape[0] - 1)
    cc_c = min(max(cc - 1, 0), horn.shape[1] - 1)

    center = float(horn[rr_c, cc_c])
    local3 = float(np.nanmax(horn[max(rr_c-1,0):rr_c+2, max(cc_c-1,0):cc_c+2]))
    local5 = float(np.nanmax(horn[max(rr_c-2,0):rr_c+3, max(cc_c-2,0):cc_c+3]))

    rd_center = None
    rd_no_gt_center = None
    rd_no_gt_local3 = None
    if RICHDEM_OK:
        dem_rd = rd.rdarray(z, no_data=-9999.0)
        dem_rd.geotransform = (t_src.c, t_src.a, 0.0, t_src.f, 0.0, t_src.e)
        rd_slope = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
        cr = min(max(meta["src_r"] - meta["r0"], 0), rd_slope.shape[0] - 1)
        cc0 = min(max(meta["src_c"] - meta["c0"], 0), rd_slope.shape[1] - 1)
        rd_center = float(rd_slope[cr, cc0])

        # Also compute richdem WITHOUT geotransform to mimic legacy behavior
        dem_rd2 = rd.rdarray(z, no_data=-9999.0)
        rd_slope2 = rd.TerrainAttribute(dem_rd2, attrib='slope_degrees')
        rd_no_gt_center = float(rd_slope2[cr, cc0])
        rd_no_gt_local3 = float(np.nanmax(rd_slope2[max(cr-1,0):cr+2, max(cc0-1,0):cc0+2]))

    return SlopeCandidates(center, local3, local5, rd_center, rd_no_gt_center, rd_no_gt_local3)


def find_dem_or_download(lat: float, lon: float) -> Optional[str]:
    # Try an existing local file first if present (user may have tiles)
    # Otherwise attempt download via leafmap
    return download_dem(lat, lon, metres=25_000)


def main() -> None:
    df = pd.read_csv('data/Corrected_Input_Data.csv')
    rows = [0, 1]
    for idx in rows:
        row = df.iloc[idx]
        lat = float(row['Latitude'])
        lon = float(row['Longitude'])
        csv_val = float(row['Slope From USGS Elevation Data'])
        tif = find_dem_or_download(lat, lon)
        if not tif:
            print(f"Row {idx}: DEM not available (lat={lat}, lon={lon}). Skipping.")
            continue
        try:
            cand = candidates_for_point(tif, lat, lon)
            print(f"\nRow {idx} lat={lat:.6f}, lon={lon:.6f}")
            print(f"CSV slope: {csv_val:.6f}")
            print(f"Horn center: {cand.horn_center}")
            print(f"Horn local 3x3 max: {cand.horn_local3x3_max}")
            print(f"Horn local 5x5 max: {cand.horn_local5x5_max}")
            print(f"RichDEM center: {cand.richdem_center}")
        except Exception as e:
            print(f"Row {idx}: failed to compute candidates: {e}")


if __name__ == '__main__':
    main()


