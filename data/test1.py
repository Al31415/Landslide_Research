import os
import leafmap
import rasterio

from PIL import Image
import numpy as np
import richdem as rd
from rasterio.transform import rowcol

data_lat = 20.644348
data_long = -156.098826

import math

def offset(lat, lon):
    R = 6378137
    dn = 100
    de = 100
    dLat = dn / R
    dLon = de / (R * math.cos(math.pi * lat / 180))
    latO = lat + dLat * 180 / math.pi
    lonO = lon + dLon * 180 / math.pi
    region = [lon, lat, lonO, latO]
    return region

region = offset(data_lat, data_long)
dem_url = leafmap.download_ned(region, return_url=True)[0]

dem_filename = os.path.join("data", os.path.basename(dem_url))
leafmap.download_file(dem_url, dem_filename)

im = Image.open(dem_filename)
imarray = np.array(im)


with rasterio.open(dem_filename) as ds:
    transform = ds.transform
    r, c = rowcol(transform, data_long, data_lat)
    ycoord = int(r)
    xcoord = int(c)

beau = rd.rdarray(imarray, no_data=-9999)
dem_slope = rd.TerrainAttribute(beau, attrib='slope_degrees')
print(f"Slope value at coordinates ({ycoord}, {xcoord}): {dem_slope[ycoord][xcoord]}")
