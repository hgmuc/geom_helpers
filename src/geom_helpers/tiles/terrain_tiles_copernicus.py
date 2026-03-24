import os
import math
import numpy as np
# import numpy.typing as npt
from shapely.geometry import Point #, Polygon
from matplotlib.axes import Axes
from math import floor, ceil
from scipy import interpolate
from itertools import product
import rasterio   # switch from GDAL to rasterio - better with pip and cloud servers
#from osgeo import gdal
#from concurrent.futures import ThreadPoolExecutor

from typing import cast

from basic_helpers.file_helper import do_unpickle
from geom_helpers.tiles.terrain_heatmap import export_raw_img, save_img_as_tile, preprocess_elev_arr
from geom_helpers.tiles.xyz_tiles import (
    num2deg, download_remote_file, write_file, make_folder,
    TILES_DIRECTORY, ELEV_DATA_PATH, REF_BBOX, HIGH_RES_BBOX)

from basic_helpers.types_base import Url, CoordVal, BBox, ElevArr


SrtmLatLonMap = dict[tuple[CoordVal, CoordVal, int], str | None]
SrtmDataDict = dict[str, ElevArr]


def init_existing_srtm_dict() -> SrtmLatLonMap:
    EXISTING_SRTM = {(int(f[1:3]), int(f[4:7]), 90): os.path.join("C:/SRTM2", f) for f in os.listdir("C:/SRTM2")}
    for res in [30, 90]:
        for f in os.listdir(os.path.join(ELEV_DATA_PATH, "copernicus", str(res))):
            key1 = int(f[23:25]) * (-1 if f[22] == 'S' else 1)
            key2 = int(f[30:33]) * (-1 if f[29] == 'W' else 1)
            #print(key1, key2, res, "\t", os.path.join(ELEV_DATA_PATH, "copernicus", str(res), f))
            EXISTING_SRTM[(key1, key2, res)] = os.path.join(ELEV_DATA_PATH, "copernicus", str(res), f)

    return cast(SrtmLatLonMap, EXISTING_SRTM)

EXISTING_SRTM = init_existing_srtm_dict()
SRTM_DATA_DICT = do_unpickle("C:/01_AnacondaProjects/osmium/SRTM_DATA_DICT.pkl")

def get_aws_copernius_file_url(res: int = 90, lat: str = 'N48', lon: str = 'E012') -> str:
    res_arcsec = res//3
    base_url = get_aws_copernicus_base_url(res)
    file_path_comp = f'Copernicus_DSM_COG_{res_arcsec}_{lat}_00_{lon}_00_DEM'
    return cast(str, base_url + file_path_comp + "/" + file_path_comp + ".tif")

def get_aws_copernicus_base_url(res: int = 90) -> Url:
    return f'https://copernicus-dem-{res}m.s3.amazonaws.com/'

def create_terrain_tile_from_copernicus(z: int,x: int,y: int, ax: Axes, res: int = 90) -> bool:
    lat, lon = num2deg(x,y,z)
    tiles_bbox = get_tiles_bbox(x, y, z)
    #tile_poly = Polygon([tiles_bbox[:2], [tiles_bbox[2], tiles_bbox[1]], 
    #                     tiles_bbox[2:], [tiles_bbox[0], tiles_bbox[3]]])

    if not Point(lon,lat).intersects(REF_BBOX):   # type: ignore
    #if not tile_poly.intersects(REF_BBOX):
        print("A == False", z, x, y, res, "|", tiles_bbox)
        return False

    if res == 30 and not Point(lon,lat).intersects(HIGH_RES_BBOX):  # type: ignore
    #if res == 30 and not tile_poly.intersects(HIGH_RES_BBOX):
        print("B == False", z, x, y, res, "|", tiles_bbox)
        return False    

    srtm_files = get_srtm_file_names(tiles_bbox, res)
    
    arr_slices: dict[int, ElevArr | None] = {}
    dims = []
    #with ThreadPoolExecutor(4) as executor:
    for i, key in enumerate(sorted(srtm_files)):
        fpath = srtm_files[key] if key in srtm_files else None
        lat, lon, res = key
        if fpath is None and fpath not in SRTM_DATA_DICT:
            _ = handle_copernicus_download(res, lat, lon)
            SRTM_DATA_DICT[fpath] = np.empty((1,1))
        print(" - load    ", z, x, y, " - ", i, fpath)
        elev_arr = get_elevation_data(int(lat), int(lon), res)
        SRTM_DATA_DICT[fpath] = elev_arr
        dims.append(elev_arr.shape[0])
        arr_slices[i] = get_slice(elev_arr, tiles_bbox, i, len(srtm_files), SRTM_DATA_DICT, dims, res)

        if arr_slices[i] is None:
            return False
            
    if len(arr_slices) == 1:
        elev_arr = arr_slices[0]
    elif len(arr_slices) > 1:
        elev_arr = compose_elev_arr_slices(arr_slices)

    if len(arr_slices) == 1:
        elev_arr = arr_slices[0]
    elif len(arr_slices) > 1:
        elev_arr = compose_elev_arr_slices(arr_slices)
    
    if elev_arr:
        elev_arr = np.clip(preprocess_elev_arr(elev_arr, 1000, 0.85), a_min=-10, a_max=10000)
        export_raw_img(elev_arr, z, x, y, ax, cmap='gist_earth', vmin=-500, vmax=1350, cbar=False, kernel=2)
        make_folder(x, z, TILES_DIRECTORY)
        save_img_as_tile(os.path.join(TILES_DIRECTORY, str(z), str(x), f"{y}.png"), z, x, y, kernel=2)

        return True
    else:
        return False

def get_elevation_data(lat: CoordVal, lon: CoordVal, res: int, fpath: str = "") -> ElevArr:
    fpath = cast(str, EXISTING_SRTM[(lat, lon, res)]) if (lat, lon, res) in EXISTING_SRTM else ""
    #print(fpath, lat, lon, res)
    if fpath and isinstance(fpath, str):
        fpath = fpath.replace("/", "\\")
        
        if fpath in SRTM_DATA_DICT:
            elev_arr = SRTM_DATA_DICT[fpath]
            #print("SRTM ", elev_arr.shape, "\t", fpath.split("/")[-1])
        else:
            #elev_arr = gdal.Open(fpath).ReadAsArray()
            with rasterio.open(fpath) as src:
                elev_arr = src.read(1) # Reads the first band as a Numpy array

            #print("COPER" if fpath.endswith(".tif") else "SRTM ", elev_arr.shape, "\t", fpath.split("/")[-1])
            
    else:
        elev_arr = np.ones((10,10), dtype="int") * -10
        #print("EMPTY", elev_arr.shape)
    
    return elev_arr

def get_slice(elev_arr: ElevArr, bbox: BBox, pos_idx: int, n_quads: int, 
              SRTM_DATA_DICT: SrtmDataDict, dims: list[int], res) -> ElevArr | None:
    DIM, W_ARR = elev_arr.shape
    
    if DIM == 10:
        elev_arr, DIM, W_ARR, dims = stretch_empty_arr(elev_arr, res, dims)
    
    if W_ARR < DIM:
        elev_arr, DIM, W_ARR, dims = stretch_arr(elev_arr, DIM, W_ARR, dims)
    
    if res == 30 and DIM < 3600:
        elev_arr, DIM, W_ARR, dims = increase_arr_size(elev_arr, DIM, dims)
        
    if pos_idx > 0 and DIM != dims[0]:
        elev_arr, DIM, W_ARR, dims = align_arr_size(elev_arr, DIM, dims)
        
    lat1, lon1, lat2, lon2 = bbox
    
    if lat1 < 0:
        lat1 = np.abs(floor(lat1) - lat1 + ceil(lat1))
    if lat2 < 0:
        lat2 = np.abs(floor(lat2) - lat2 + ceil(lat2))
        
    if lon1 < 0:
        lon1 = np.abs(floor(lon1) - lon1 + ceil(lon1))
        
    if lon2 <= 0:
        lon2 -= 1e-6
        lon2 = np.abs(floor(lon2) - lon2 + ceil(lon2))        

    y1 = int(floor((1 - (lat1 - int(lat1))) * DIM))
    y2 = int(ceil((1 - (lat2 - int(lat2))) * DIM))
    x1 = int(floor((lon1 - int(lon1)) * DIM))
    x2 = int(ceil((lon2 - int(lon2)) * DIM))
    
    if n_quads == 1:
        return elev_arr[y2:y1+1, x1:x2+1]
        
    elif n_quads == 2 and int(lat1) == int(lat2):
        if pos_idx == 0:
            return elev_arr[y2:y1+1, x1:DIM+1]
        elif pos_idx == 1:
            return elev_arr[y2:y1+1, :x2]
        
    elif n_quads == 2 and int(lon1) == int(lon2):
        if pos_idx == 0:
            return elev_arr[:y1+1, x1:x2+1]
        elif pos_idx == 1:
            return elev_arr[y2:, x1:x2+1]
        
    elif n_quads == 4:
        if pos_idx == 0:
            return elev_arr[:y1+1, x1:DIM]
        elif pos_idx == 1:
            return elev_arr[:y1+1, :x2+1]
        elif pos_idx == 2:
            return elev_arr[y2:, x1:DIM]
        elif pos_idx == 3:
            return elev_arr[y2:, :x2+1]

    #else:
    return None

def stretch_empty_arr(elev_arr: ElevArr, res: int, dims: list[int]) -> tuple[ElevArr, int, int, list[int]]:
    fct = 120
    if res == 30:
        fct *= 3
    elev_arr = np.repeat(np.repeat(elev_arr, fct, axis=1), fct, axis=0)
    DIM, W_ARR = elev_arr.shape
    print("STRETCH EMPTY", DIM, W_ARR)
    dims[-1] = DIM
    return elev_arr, DIM, W_ARR, dims

def stretch_arr(elev_arr: ElevArr, DIM: int, W_ARR: int, dims: list[int]) -> tuple[ElevArr, int, int, list[int]]:
    x = np.linspace(0, DIM-1, W_ARR)
    y = np.array(range(DIM))
    f = interpolate.interp2d(x, y, elev_arr, kind='linear')
    
    xnew = np.linspace(0, DIM-1, DIM)
    znew = f(xnew, y)    # type: ignore

    DIM, W_ARR = elev_arr.shape
    dims[-1] = DIM
    return znew, DIM, W_ARR, dims

def increase_arr_size(elev_arr: ElevArr, DIM: int, dims: list[int]) -> tuple[ElevArr, int, int, list[int]]:
    new_axis_len = 3600 if DIM == 1200 else 3601
    elev_arr = np.repeat(np.repeat(elev_arr, 3, axis=1), 3, axis=0)[:new_axis_len, :new_axis_len]
    print("RES = 30  -> INCREASED Array", elev_arr.shape)
    DIM, W_ARR = elev_arr.shape
    dims[-1] = DIM
    return elev_arr, DIM, W_ARR, dims

def align_arr_size(elev_arr, DIM, dims) -> tuple[ElevArr, int, int, list[int]]:
    print("SIZE MISMATCH  -> Aligning Array", DIM, dims)
    if DIM > dims[0]:
        elev_arr = elev_arr[:dims[0], :dims[0]]
    else:
        elev_arr = np.pad(elev_arr, ((0,1), (0,1)), 'edge')

    DIM, W_ARR = elev_arr.shape
    dims[-1] = DIM
    return elev_arr, DIM, W_ARR, dims

def get_slice2(elev_arr: ElevArr, bbox: BBox, pos_idx: int, n_quads: int, SRTM_DATA_DICT: SrtmDataDict) -> ElevArr | None:
    DIM, W_ARR = elev_arr.shape
    
    if DIM == 10:
        elev_arr = np.repeat(np.repeat(elev_arr, 120, axis=1), 120, axis=0)
        DIM, W_ARR = elev_arr.shape
    
    if W_ARR < DIM:
        fct = math.ceil(DIM / W_ARR)
        new_width = W_ARR * fct
        elev_arr = np.repeat(elev_arr, fct, axis=1)[:, (new_width-DIM)//2:-(new_width-DIM)//2]
        
    ADDITIONAL_RC = DIM % 100
    lat1, lon1, lat2, lon2 = bbox
    
    if lat1 < 0:
        lat1 = np.abs(floor(lat1) - lat1 + ceil(lat1))
    if lat2 < 0:
        lat2 = np.abs(floor(lat2) - lat2 + ceil(lat2))
        
    if lon1 < 0:
        lon1 = np.abs(floor(lon1) - lon1 + ceil(lon1))
        
    if lon2 <= 0:
        lon2 -= 1e-6
        lon2 = np.abs(floor(lon2) - lon2 + ceil(lon2))
        

    y1 = int(floor((1 - (lat1 - int(lat1))) * DIM))
    y2 = int(ceil((1 - (lat2 - int(lat2))) * DIM))
    x1 = int(floor((lon1 - int(lon1)) * DIM))
    x2 = int(ceil((lon2 - int(lon2)) * DIM))
    
    if n_quads == 1:
        return elev_arr[y2:y1+1, x1:x2+1]
        
    elif n_quads == 2 and int(lat1) == int(lat2):
        if pos_idx == 0:
            return elev_arr[y2:y1+1-ADDITIONAL_RC, x1:DIM+1-ADDITIONAL_RC]
        elif pos_idx == 1:
            return elev_arr[y2:y1+1-ADDITIONAL_RC, :x2]
        
    elif n_quads == 2 and int(lon1) == int(lon2):
        if pos_idx == 0:
            return elev_arr[ADDITIONAL_RC:y1+1, x1:x2+1-ADDITIONAL_RC]
        elif pos_idx == 1:
            return elev_arr[y2:, x1:x2+1-ADDITIONAL_RC]

    elif n_quads == 4:
        if pos_idx == 0:
            return elev_arr[ADDITIONAL_RC:y1+1+ADDITIONAL_RC, x1:DIM-ADDITIONAL_RC]
        elif pos_idx == 1:
            return elev_arr[ADDITIONAL_RC:y1+1+ADDITIONAL_RC, :x2+1]
        elif pos_idx == 2:
            return elev_arr[y2:, x1:DIM-ADDITIONAL_RC]
        elif pos_idx == 3:
            return elev_arr[y2:, :x2+1]
        
    #else:
    return None

def compose_elev_arr_slices(arr_slices: dict[int, ElevArr | None]) -> ElevArr | None:
    W = H = 0
    w0 = h1 = -1
    spreading_east = True
    if len(arr_slices) == 2 and arr_slices[0] and arr_slices[1]:
        if arr_slices[0].shape[0] == arr_slices[1].shape[0]:
            H = arr_slices[0].shape[0]
            W = arr_slices[0].shape[1] + arr_slices[1].shape[1]
        else:
            H = arr_slices[0].shape[0] + arr_slices[1].shape[0]
            W = arr_slices[0].shape[1]
            spreading_east = False
            
        arr = np.zeros((H, W), dtype=int)
        if spreading_east:
            w0 = arr_slices[0].shape[1]
            arr[:, :w0] = arr_slices[0]
            arr[:, w0:] = arr_slices[1]
        else:
            h1 = arr_slices[1].shape[0]
            arr[h1:, :] = arr_slices[0]
            arr[:h1, :] = arr_slices[1]
            
    elif arr_slices[0] and arr_slices[1] and arr_slices[2] and arr_slices[3]:
        try:
            H = arr_slices[0].shape[0] + arr_slices[2].shape[0]
            W = arr_slices[0].shape[1] + arr_slices[1].shape[1]
            arr = np.zeros((H, W), dtype=int)

            w0 = arr_slices[0].shape[1]
            h1 = arr_slices[2].shape[0]
            arr[h1:,:w0] = arr_slices[0]
            arr[h1:,w0:] = arr_slices[1]
            arr[:h1,:w0] = arr_slices[2]
            arr[:h1,w0:] = arr_slices[3]
        except Exception as e:
            print("arr", [a.shape for a in arr_slices.values() if a], h1, w0, e)
    
    return None

def handle_copernicus_download(res: int, lat: CoordVal, lon: CoordVal) -> bool:
    latstr, lonstr = get_lonlat_str(lat, lon)
    print(" - start downloading", lat, lon)
    dwnld_url = get_aws_copernius_file_url(res, latstr, lonstr)
    content = download_remote_file(dwnld_url)

    if content is None:
        EXISTING_SRTM[(int(latstr[1:]), int(lonstr[1:]), res)] = None
        return False
        
    fpath = get_aws_copernicus_fpath(dwnld_url, res)
    is_new_dwnld = write_file(content, fpath)
    if is_new_dwnld:
        EXISTING_SRTM[(int(latstr[1:]), int(lonstr[1:]), res)] = fpath
    
    return is_new_dwnld

def get_aws_copernicus_fpath(dwnld_url: Url, res: int = 90) -> str:
    fname = dwnld_url.split("/")[-1]
    fpath = os.path.join(ELEV_DATA_PATH, "copernicus", str(res), fname)
    return fpath

def get_tiles_bbox(x: int, y: int, z: int) -> BBox:
    lat1, lon1 = num2deg(x, y, z)
    lat2, lon2 = num2deg(x+1, y+1, z)
    return lat2, lon1, lat1, lon2

def get_srtm_file_names(bbox: BBox, res: int) -> SrtmLatLonMap:
    lat_range = list(range(int(bbox[0]) - (0 if bbox[0] >= 0 else 1), 
                           int(bbox[2]) + (1 if bbox[2] > 0 else 0)))
    lon_range = list(range(int(bbox[1]) - (0 if bbox[1] >= 0 else 1), 
                           int(bbox[3]) + (1 if bbox[3] > 0 else 0)))
    
    keys = [(int(lat), int(lon), res) for lat, lon in product(lat_range, lon_range, repeat=1)]

    return {key: EXISTING_SRTM[key] if key in EXISTING_SRTM else None for key in keys}

def get_lonlat_str(lat: CoordVal, lon: CoordVal) -> tuple[str, str]:
    latstr = f"{'N' if lat > 0 else 'S'}{int(lat):02d}" 
    if lon < 0:
        lonstr = f"{'E' if lon >= 0 else 'W'}{int(np.abs(lon-1)):03d}" 
    else:
        lonstr = f"{'E' if lon >= 0 else 'W'}{int(np.abs(lon)):03d}" 

    print("lat, lon:", lat, lon, " || ", latstr, lonstr)
    return latstr, lonstr