import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from geom_helpers.tiles.terrain_heatmap import export_raw_img, save_img_as_tile, preprocess_elev_arr
from geom_helpers.tiles.xyz_tiles import check_path, write_file, download_remote_file, TILES_DIRECTORY, ELEV_DATA_PATH

from basic_helpers.types_base import ElevArr


KERNEL = 2

def create_terrain_tile_from_mapzen(z: int, x: int, y: int, ax: Axes) -> bool:
    url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png'
    aws_path = os.path.join(ELEV_DATA_PATH, "terrarium", str(z), str(x), f"{y}.png")
    #print("aws_path", aws_path, os.path.exists(os.path.join(ELEV_DATA_PATH, "terrarium", z, str(x))))
    tiles_path = os.path.join(TILES_DIRECTORY, str(z), str(x), f"{y}.png")
    #print("tiles_path", tiles_path, os.path.exists(os.path.join(TILES_DIRECTORY, z, str(x))))

    aws_path_exists = check_path(z, x, os.path.join(ELEV_DATA_PATH, "terrarium"))
    if not aws_path_exists:
        return aws_path_exists
    tiles_path_exists = check_path(z, x, TILES_DIRECTORY)
    if not tiles_path_exists:
        return tiles_path_exists

    if not os.path.exists(aws_path):
        #print("   download terrain tile and save", z, x, y)
        content = download_remote_file(url)
        if content:
            is_terrain_tile = write_file(content, aws_path)
            if not is_terrain_tile:
                return False

    #print("   open terrain tile and compute elevations", z, x, y)
    if int(z) > 6:
        elevations = np.clip(preprocess_elev_arr(get_elevation_data(aws_path)), a_min=-10, a_max=10000)
    else:
        elevations = preprocess_elev_arr(get_elevation_data(aws_path))
    #print("   process elevations and save heatmap", z, x, y)
    export_raw_img(elevations, z, x, y, ax, cmap='gist_earth', vmin=-500, vmax=1350, cbar=False, kernel=KERNEL)
    #print("   load heatmap and saved map tile", z, x, y)
    save_img_as_tile(tiles_path, z, x, y, KERNEL)

    return True

def get_elevation_data_old(fpath: str) -> ElevArr:
    data = plt.imread(fpath)
    elevations = np.round(np.sum(data * 255 * np.array([256, 1, 1/256]), axis=2) - 32768)
    return elevations.astype(int)

def get_elevation_data(fpath: str) -> ElevArr:
    data = plt.imread(fpath)
    # Ensure we only use RGB, even if the file has an Alpha channel (RGBA)
    if data.shape[-1] == 4:
        data = data[:, :, :3]
        
    elevations = np.round(np.sum(data * 255 * np.array([256, 1, 1/256]), axis=2) - 32768)
    return elevations.astype(int)