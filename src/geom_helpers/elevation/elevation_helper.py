import numpy as np
import matplotlib.pyplot as plt
from geom_helpers.tiles.xyz_tiles import num2deg

from basic_helpers.types_base import ElevArr, BBox

def get_elevation_data(fpath: str) -> ElevArr:
    data = plt.imread(fpath)
    if np.max(data) < 2:
        # Mapzen tiles => raw values btw 0 and 1 => scale by 255 to be in range 0 - 255, then multiply with RGB channel values, sum and subtract constant
        elevations = np.round(np.sum(data * 255 * np.array([256, 1, 1/256]), axis=2) - 32768)
    else:
        # Mapterhorn tiles => raw values btw 0 and 255 => no scaling, multiply with RGB channel values, sum and subtract constant
        elevations = np.round(np.sum(data * np.array([256, 1, 1/256]), axis=2) - 32768)
    return elevations.astype(int)

def get_ref_full_bbox(xs: list[int], ys: list[int], z: int) -> BBox:
    max_lat, min_lon = num2deg(xs[0], ys[0], z)
    min_lat, max_lon = num2deg(max(xs)+1, max(ys)+1, z)
    
    return min_lat, min_lon, max_lat, max_lon
