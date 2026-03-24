import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
import numpy as np

from PIL import Image

from basic_helpers.types_base import ElevArr

# ElevArr = npt.NDArray[np.floating[Any] | np.integer[Any]]

def preprocess_elev_arr(arr: ElevArr, lim: int = 1000, fct: float = 0.85) -> ElevArr:
    arr = arr.copy()
    diff_arr = np.zeros_like(arr)
    mask = arr > lim
    diff_arr[mask] = (arr[mask] - 1000) * fct
    arr -= diff_arr
    return arr

def export_raw_img(elev_arr: ElevArr, zoom: int, x: int, y: int, ax: Axes, kernel: int, 
                   cmap: str = 'gist_earth', vmin: int = -500, vmax: int = 1800, 
                   cbar: bool = False) -> None:
    ax.clear()
    ax = sns.heatmap(elev_arr, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax, cbar=cbar)
    ax.set_axis_off()
    plt.savefig(f'output{kernel}{zoom}-{x}-{y}.png', bbox_inches='tight', pad_inches=0.0)

def save_img_as_tile(tile_path: str, zoom: int, x: int, y: int, kernel: int) -> None:
    img = imageio.v2.imread(f'output{kernel}{zoom}-{x}-{y}.png')
    mask = img[:,:,3] > 0
    img2 = img[mask.sum(axis=1)>0, :, :][:,mask.sum(axis=0)>0,:]

    img_obj = Image.fromarray(img2)
    img_obj = img_obj.resize((256,256))

    img_obj.save(tile_path)
    img_obj.close()
    os.remove(f'output{kernel}{zoom}-{x}-{y}.png')

