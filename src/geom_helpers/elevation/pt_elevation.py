import os
import numpy as np
from scipy import interpolate
from numba import njit
from typing import Literal

from basic_helpers.file_helper import do_pickle
#from geom_helpers.tiles.xyz_tiles import deg2num, num2deg

from basic_helpers.types_base import CoordVal, ElevArr, ElevInt, BBox

SrtmDataDict = dict[str, ElevArr]

def get_srtm_dict_path() -> str:
    return os.path.join("C:/01_AnacondaProjects/bikesite/SRTM_DATA_DICT.pkl")

def get_srtm_files_directory() -> str:
    return os.path.join("C:/SRTM2")

def get_file_name(lat: CoordVal, lon: CoordVal) -> str | None:
    """
    Returns filename such as N27E086.hgt, concatenated
    with HGTDIR where these 'hgt' files are kept
    """
    srtm_files_dir = get_srtm_files_directory()

    if lat >= 0:
        ns = 'N'
    elif lat < 0:
        ns = 'S'
        lat = np.abs(lat) + 1

    if lon >= 0:
        ew = 'E'
    elif lon < 0:
        ew = 'W'
        lon = np.abs(lon) + 1

    hgt_file = "{}{:02d}{}{:03d}.hgt".format(ns, int(lat), ew, int(lon))
    hgt_file_path = os.path.join(srtm_files_dir, hgt_file)
    if os.path.isfile(hgt_file_path):
        return hgt_file_path.replace("/", "\\")
    else:
        return None

def get_elevation_arr(lat: CoordVal, lon: CoordVal, DIMS: int = 1201) -> ElevArr | None:
    '''
    Liefert den Inhalt der HGT-Datei zu Lat-Lon als Array (1201x1201) zurück
    
    Input: lat, lon (floats)
    '''
    hgt_file = get_file_name(lat, lon)
    if hgt_file is None:
        return None
    with open(hgt_file, 'rb') as hgt_data:
        # Each data is 16bit signed integer(i2) - big endian(>)
        elevations = np.fromfile(hgt_data, np.dtype('>i2'), DIMS*DIMS).reshape((DIMS, DIMS))
        hgt_data.close()
        
    return elevations

def fill_elevation_voids(ev_arr: ElevArr) -> ElevArr | None:
    if ev_arr is None:
        return None
    
    res_ev_arr = np.zeros_like(ev_arr)

    missing_coords = np.argwhere(ev_arr<0)
    lat_rows = set([coords[0] for coords in missing_coords])
    lon_cols = set([coords[1] for coords in missing_coords])

    res_ev_arr = 4 * ev_arr

    for i in lat_rows:   
        missing_pos = [coords[1] for coords in missing_coords if coords[0] == i]
        res_ev_arr[i, missing_pos] = 0
        x = list(np.argwhere(ev_arr[i,:]>=0).reshape(-1,))
        y = ev_arr[i, x]
        xnew = np.arange(0,1201,1)

        for k in ['nearest', 'slinear']:
            f = interpolate.interp1d(x, y, kind=k, fill_value="extrapolate")   # type: ignore
            ynew = f(xnew)   # use interpolation function returned by `interp1d`
            res_ev_arr[i, missing_pos] += np.round(ynew[missing_pos]).astype(int)

    for i in lon_cols:   
        missing_pos = [coords[0] for coords in missing_coords if coords[1] == i]
        x = list(np.argwhere(ev_arr[:, i]>=0).reshape(-1,))
        y = ev_arr[x, i]
        xnew = np.arange(0,1201,1)

        for k in ['nearest', 'slinear']:
            f = interpolate.interp1d(x, y, kind=k, fill_value="extrapolate")  # type: ignore
            ynew = f(xnew)   # use interpolation function returned by `interp1d`       
            res_ev_arr[missing_pos, i] += np.round(ynew[missing_pos]).astype(int)

    res_ev_arr = np.round((res_ev_arr / 4)).astype(int)

    return res_ev_arr

def get_srtm_elevations_from_bbox(bbox: BBox, srtm_data_dict: SrtmDataDict | None = None, 
                                  elevations: ElevArr | None = None, DIMS: int = 1201) -> ElevArr | None:
    '''
    Ermittelt die Höheangabe für einen definierten Bereich
    - Wenn das HGT File noch nicht geladen wurde, wird es automatisch nachgeladen und Lücken gefüllt
    - Interpoliert die Höhenangabe für Koordinaten, die nicht genau auf die Rasterlinien fallen
    
    Input
    - bbox: Tuple von (min_lat, min_lon, max_lat, max_lon - vier Floats)
    - elevations: optional ein bereits geladenes Array mit SRTM-Höhenangaben
    
    Output:
    - Array mit Höhe in M
    '''
    min_lat, min_lon, max_lat, max_lon = bbox
    
    write_srtm_data_dict = False if srtm_data_dict is None else True

    if elevations is None:
        has_updates = False
        for (lat, lon) in [(min_lat, min_lon), (max_lat, max_lon)]:
            fname = get_file_name(lat, lon)
            if isinstance(srtm_data_dict, dict) and fname in srtm_data_dict:
                elevations = srtm_data_dict[fname]
                if elevations is None:
                    print("get_srtm_elevations_from_bbox - elevations None A", type(elevations), 
                        lat, lon, fname)
            elif fname is not None:
                print("load and fill voids", fname, lat, lon)
                elevations = get_elevation_arr(lat, lon)
                if elevations is not None:
                    elevations = fill_elevation_voids(elevations)
                    if elevations is not None and isinstance(srtm_data_dict, dict):
                        srtm_data_dict[fname] = elevations.copy()
                        has_updates = True
                else:
                    print("get_srtm_elevations_from_bbox - elevations None B", type(elevations), 
                          lat, lon, fname)
                    return None
            else:
                return None

        if write_srtm_data_dict and has_updates:
            do_pickle(srtm_data_dict, get_srtm_dict_path())

    min_lat_row = int(int(int(round(min_lat*1e6)) - int(min_lat)*1e6) * (DIMS - 1) / 1e6)
    min_lon_col = int(int(int(round(min_lon*1e6)) - int(min_lon)*1e6) * (DIMS - 1) / 1e6)

    max_lat_row = int(int(int(round(max_lat*1e6)) - int(max_lat)*1e6) * (DIMS - 1) / 1e6)
    max_lon_col = int(int(int(round(max_lon*1e6)) - int(max_lon)*1e6) * (DIMS - 1) / 1e6)

    assert elevations is not None

    return elevations[max(DIMS - 2 - max_lat_row, 0):(DIMS - min_lat_row), min_lon_col:max_lon_col+2]


def get_srtm_elevation(lat: CoordVal, lon: CoordVal, srtm_data_dict: SrtmDataDict | None = None, 
                       elevations: ElevArr | None = None, DIMS: int | Literal['auto'] = 1201) -> ElevInt | None:
    '''
    Ermittelt die Höheangabe zu einer Lat-Lon-Koordinate
    - Wenn das HGT File noch nicht geladen wurde, wird es automatisch nachgeladen und Lücken gefüllt
    - Interpoliert die Höhenangabe für Koordinaten, die nicht genau auf die Rasterlinien fallen
    
    Input
    - lat, lon: floats
    - elevations: optional ein bereits geladenes Array mit SRTM-Höhenangaben
    
    Output:
    - Höhe in M
    '''
    if elevations is not None and DIMS == 'auto':
        YDIMS, XDIMS = elevations.shape
    elif isinstance(DIMS, int):
        YDIMS = XDIMS = DIMS
    else:
        YDIMS = XDIMS = -1

    has_updates = False
    write_srtm_data_dict = False if srtm_data_dict is None else True

    if elevations is None:
        fname = get_file_name(lat, lon)
        if isinstance(srtm_data_dict, dict) and fname in srtm_data_dict:
            elevations = srtm_data_dict[fname]
            if elevations is None:
                print("get_srtm_elevations - elevations None A", type(elevations), 
                      lat, lon, fname)
        elif fname is not None:
            print("load and fill voids", fname, lat, lon)
            elevations = get_elevation_arr(lat, lon)
            if elevations is not None:
                elevations = fill_elevation_voids(elevations)
                if elevations is not None and isinstance(srtm_data_dict, dict):
                    srtm_data_dict[fname] = elevations.copy()
                    has_updates = True
            else:
                print("get_srtm_elevations - elevations None B", type(elevations), 
                      lat, lon, fname)
                return None
        else:
            return None
        
        if write_srtm_data_dict and has_updates:
            do_pickle(srtm_data_dict, get_srtm_dict_path())
    
    lat_row = int(int(int(round(lat*1e6)) - int(lat)*1e6) * (YDIMS - 1) / 1e6)
    lat_perc = ((lat - int(lat)) - lat_row / (YDIMS - 1)) * (YDIMS - 1)
    
    if lon < 0:
        lon = np.abs(lon)
        lon_col = int(int(int(round(lon*1e6)) - int(lon)*1e6) * (XDIMS - 1) / 1e6)
        lon_perc = ((lon - int(lon)) - lon_col / (XDIMS - 1)) * (XDIMS - 1)
        lon_col = (XDIMS - 1) - lon_col
        diff_arr = np.array([[lat_perc, lat_perc], [1-lat_perc, 1-lat_perc]])
        diff_arr *= np.array([[lon_perc, 1-lon_perc], [lon_perc, 1-lon_perc]])
    else:
        lon_col = int(int(int(round(lon*1e6)) - int(lon)*1e6) * (XDIMS - 1) / 1e6)
        lon_perc = ((lon - int(lon)) - lon_col / (XDIMS - 1)) * (XDIMS - 1)
        diff_arr = np.array([[lat_perc, lat_perc], [1-lat_perc, 1-lat_perc]])
        diff_arr *= np.array([[1-lon_perc, lon_perc], [1-lon_perc, lon_perc]])

    assert elevations is not None
    if XDIMS == -1:
        YDIMS, XDIMS = elevations.shape
    
    raw_elevations = elevations[max(YDIMS - 2 - lat_row, 0):(YDIMS - lat_row), min(lon_col, XDIMS-2):lon_col+2]

    return int(round(np.sum(raw_elevations*diff_arr)))

def get_srtm_elevation_numba(lat: CoordVal, lon: CoordVal, srtm_data_dict: SrtmDataDict | None = None, 
                             elevations: ElevArr | None = None, DIMS: int | Literal['auto'] = 1201) -> ElevInt | None:
    '''
    Ermittelt die Höheangabe zu einer Lat-Lon-Koordinate
    - Wenn das HGT File noch nicht geladen wurde, wird es automatisch nachgeladen und Lücken gefüllt
    - Interpoliert die Höhenangabe für Koordinaten, die nicht genau auf die Rasterlinien fallen
    
    Input
    - lat, lon: floats
    - elevations: optional ein bereits geladenes Array mit SRTM-Höhenangaben
    
    Output:
    - Höhe in M
    '''
    if elevations is not None and DIMS == 'auto':
        YDIMS, XDIMS = elevations.shape
    elif isinstance(DIMS, int):
        YDIMS = XDIMS = DIMS
    else:
        YDIMS = XDIMS = -1

    write_srtm_data_dict = False if srtm_data_dict is None else True

    if elevations is None:
        fname = get_file_name(lat, lon)
        if isinstance(srtm_data_dict, dict) and fname in srtm_data_dict:
            elevations = srtm_data_dict[fname]
            if elevations is None:
                print("get_srtm_elevations - elevations None A", type(elevations), 
                      lat, lon, fname)
        elif fname is not None:
            print("load and fill voids", fname, lat, lon)
            elevations = get_elevation_arr(lat, lon)
            if elevations is not None:
                elevations = fill_elevation_voids(elevations)
                if elevations is not None and isinstance(srtm_data_dict, dict):
                    srtm_data_dict[fname] = elevations.copy()
                    has_updates = True
            else:
                print("get_srtm_elevations - elevations None B", type(elevations), 
                      lat, lon, fname)
                return None
        else:
            return None

        if write_srtm_data_dict and has_updates:
            do_pickle(srtm_data_dict, get_srtm_dict_path())

    assert elevations is not None
    if XDIMS == -1:
        YDIMS, XDIMS = elevations.shape

    if lon < 0:
        return compute_srtm_elevation_numba_w(lat, lon, elevations, XDIMS, YDIMS)
    else:
        return compute_srtm_elevation_numba_e(lat, lon, elevations, XDIMS, YDIMS)

@njit(fastmath=True)
def compute_srtm_elevation_numba_e(lat: CoordVal, lon: CoordVal, elevations: ElevArr, XDIMS: int, YDIMS: int):
    '''
    Ermittelt die Höheangabe zu einer Lat-Lon-Koordinate
    - Interpoliert die Höhenangabe für Koordinaten, die nicht genau auf die Rasterlinien fallen
    
    Input
    - lat, lon: floats
    - elevations: ein bereits geladenes Array mit SRTM-Höhenangaben
    
    Output:
    - Höhe in M
    '''

    lat_row = int(int(int(round(lat*1e6)) - int(lat)*1e6) * (YDIMS - 1) / 1e6)
    lat_perc = ((lat - int(lat)) - lat_row / (YDIMS - 1)) * (YDIMS - 1)
    
    lon_col = int(int(int(round(lon*1e6)) - int(lon)*1e6) * (XDIMS - 1) / 1e6)
    lon_perc = ((lon - int(lon)) - lon_col / (XDIMS - 1)) * (XDIMS - 1)
    diff_arr = np.array([[lat_perc, lat_perc], [1-lat_perc, 1-lat_perc]])
    diff_arr *= np.array([[1-lon_perc, lon_perc], [1-lon_perc, lon_perc]])

    raw_elevations = elevations[max(YDIMS - 2 - lat_row, 0):(YDIMS - lat_row), min(lon_col, XDIMS-2):lon_col+2]

    return int(round(np.sum(raw_elevations*diff_arr)))

@njit(fastmath=True)
def compute_srtm_elevation_numba_w(lat: CoordVal, lon: CoordVal, elevations: ElevArr, XDIMS: int, YDIMS: int):
    '''
    Ermittelt die Höheangabe zu einer Lat-Lon-Koordinate
    - Interpoliert die Höhenangabe für Koordinaten, die nicht genau auf die Rasterlinien fallen
    
    Input
    - lat, lon: floats
    - elevations: ein bereits geladenes Array mit SRTM-Höhenangaben
    
    Output:
    - Höhe in M
    '''
    lat_row = int(int(int(round(lat*1e6)) - int(lat)*1e6) * (YDIMS - 1) / 1e6)
    lat_perc = ((lat - int(lat)) - lat_row / (YDIMS - 1)) * (YDIMS - 1)
    
    lon = np.abs(lon)
    lon_col = int(int(int(round(lon*1e6)) - int(lon)*1e6) * (XDIMS - 1) / 1e6)
    lon_perc = ((lon - int(lon)) - lon_col / (XDIMS - 1)) * (XDIMS - 1)
    lon_col = (XDIMS - 1) - lon_col
    diff_arr = np.array([[lat_perc, lat_perc], [1-lat_perc, 1-lat_perc]])
    diff_arr *= np.array([[lon_perc, 1-lon_perc], [lon_perc, 1-lon_perc]])

    raw_elevations = elevations[max(YDIMS - 2 - lat_row, 0):(YDIMS - lat_row), min(lon_col, XDIMS-2):lon_col+2]

    return int(round(np.sum(raw_elevations*diff_arr)))


def get_elevation_from_tile(lat: CoordVal, lon: CoordVal, bbox: BBox, 
                            elevations: ElevArr | None = None) -> ElevInt | None:
    '''
    Ermittelt die Höheangabe zu einer Lat-Lon-Koordinate von einem Array mit Höheangaben aus ein XYZ-Tile
    
    Input
    - lat, lon: floats
    - bbox: 4-Tuple - Bounding Box des Tiles bzw des Arrays
    - elevations: Array mit Höhenangaben in M
    
    Output:
    - Höhe in M
    '''

    if elevations is None:
        return None
    
    min_lat, min_lon, max_lat, max_lon = bbox
    lat_rng = max_lat - min_lat
    lon_rng = max_lon - min_lon

    YDIMS, XDIMS = elevations.shape
    print("YDIMS, XDIMS", YDIMS, XDIMS )

    lat_row = int((lat - min_lat) / lat_rng * (YDIMS - 1))
    lon_col = int((lon - min_lon) / lon_rng * (XDIMS - 1))
    print("lat_row, lon_col", lat_row, lon_col)

    raw_elevations = elevations[max(YDIMS - 2 - lat_row, 0):(YDIMS - lat_row), lon_col:lon_col+2]
    print("raw_elevations", raw_elevations.shape)
    print("raw_elevations", raw_elevations)
    print("------------")
    if raw_elevations.min() <= 0:
        if raw_elevations.max() > 40:
            return int(raw_elevations.min())

    lat_perc = ((lat - min_lat) / lat_rng - lat_row / (YDIMS - 1)) * (YDIMS - 1)
    lon_perc = ((lon - min_lon) / lon_rng - lon_col / (XDIMS - 1)) * (XDIMS - 1)

    diff_arr = np.array([[lat_perc, lat_perc], [1-lat_perc, 1-lat_perc]])
    diff_arr *= np.array([[1-lon_perc, lon_perc], [1-lon_perc, lon_perc]])

    return int(round(np.sum(raw_elevations*diff_arr)))


def get_elevation_from_tile_numba(lat: CoordVal, lon: CoordVal, bbox: BBox, elevations: ElevArr | None = None) -> ElevInt | None:
    '''
    Ermittelt die Höheangabe zu einer Lat-Lon-Koordinate von einem Array mit Höheangaben aus ein XYZ-Tile
    
    Input
    - lat, lon: floats
    - bbox: 4-Tuple - Bounding Box des Tiles bzw des Arrays
    - elevations: Array mit Höhenangaben in M
    
    Output:
    - Höhe in M
    '''

    if elevations is None:
        return None
    
    elev, raw_elevations = compute_elevation_from_tile_numba(lat, lon, bbox, elevations)
    
    if raw_elevations.min() <= 0:
        if raw_elevations.max() > 40:
            return raw_elevations.min() 

    return elev

@njit(fastmath=True)
def compute_elevation_from_tile_numba(lat: CoordVal, lon: CoordVal, bbox: BBox, elevations: ElevArr) -> tuple[ElevInt, ElevArr]:
    '''
    Ermittelt die Höheangabe zu einer Lat-Lon-Koordinate von einem Array mit Höheangaben aus ein XYZ-Tile
    
    Input
    - lat, lon: floats
    - bbox: 4-Tuple - Bounding Box des Tiles bzw des Arrays
    - elevations: Array mit Höhenangaben in M
    
    Output:
    - Höhe in M
    '''

    min_lat, min_lon, max_lat, max_lon = bbox
    lat_rng = max_lat - min_lat
    lon_rng = max_lon - min_lon

    YDIMS, XDIMS = elevations.shape

    lat_row = int((lat - min_lat) / lat_rng * (YDIMS - 1))
    lon_col = int((lon - min_lon) / lon_rng * (XDIMS - 1))

    raw_elevations = elevations[max(YDIMS - 2 - lat_row, 0):(YDIMS - lat_row), lon_col:lon_col+2]
    
    lat_perc = ((lat - min_lat) / lat_rng - lat_row / (YDIMS - 1)) * (YDIMS - 1)
    lon_perc = ((lon - min_lon) / lon_rng - lon_col / (XDIMS - 1)) * (XDIMS - 1)

    diff_arr = np.array([[lat_perc, lat_perc], [1-lat_perc, 1-lat_perc]])
    diff_arr *= np.array([[1-lon_perc, lon_perc], [1-lon_perc, lon_perc]])

    return int(round(np.sum(raw_elevations*diff_arr))), raw_elevations


def get_elevations_from_bbox_in_tile(bbox: BBox, elevations: ElevArr) -> ElevArr:
    '''
    Ermittelt die Höheangaben für einen definierten Bereich (Bbox)
    
    Input
    - bbox: Tuple von (min_lat, min_lon, max_lat, max_lon - vier Floats)
    - elevations: ein bereits geladenes Array mit Höhenangaben
    
    Output:
    - Array mit Höhen in M
    '''
    min_lat, min_lon, max_lat, max_lon = bbox

    if elevations is None:
        return None

    YDIMS, XDIMS = elevations.shape

    min_lat_row = int(np.ceil((min_lat*1e6 - int(min_lat)*1e6) * (YDIMS - 1) / 1e6))
    max_lat_row = int(np.floor((max_lat*1e6 - int(max_lat)*1e6) * (YDIMS - 1) / 1e6))

    min_lon_col = int(np.ceil((min_lon*1e6 - int(min_lon)*1e6) * (XDIMS - 1) / 1e6))
    max_lon_col = int(np.floor((max_lon*1e6 - int(max_lon)*1e6) * (XDIMS - 1) / 1e6))
    
    #print(min_lat_row, max_lat_row, min_lon_col, max_lon_col)

    return elevations[max(YDIMS - 2 - max_lat_row, 0):(YDIMS - min_lat_row), min_lon_col:max_lon_col+2]

def upsample_1d_arr(arr: ElevArr, new_len: int, mode: str = 'linear') -> ElevArr:
    if new_len < len(arr):
        return downsample_1d_arr(arr, new_len)
    elif new_len == len(arr):
        return arr
    new_arr = np.zeros(new_len, dtype=arr.dtype)
    x1 = np.linspace(0, 1, len(arr))
    x2 = np.linspace(0, 1, len(new_arr))
    
    for i, x2_pos_val in enumerate(x2):
        x1_pos = np.argwhere(x1 == x2_pos_val)
        if len(x1_pos) > 0:
            x1_pos = x1_pos[0,0]
            new_arr[i] = arr[x1_pos]
        else:
            x1_pos_l = np.argwhere(x1 < x2_pos_val)[-1,0]
            x1_pos_r = np.argwhere(x1 > x2_pos_val)[0,0]
            #print(i, x1_pos_l, x1_pos_r)
            a, b, c = x2_pos_val, x1[x1_pos_l], x1[x1_pos_r]
            if mode == "min":
                new_val = min(arr[x1_pos_l], arr[x1_pos_r])
            elif mode == "max":
                new_val = max(arr[x1_pos_l], arr[x1_pos_r])
            else:
                rng = c - b
                diff_ab = a - b
                diff_ca = c - a
                new_val = (1-diff_ab/rng) * arr[x1_pos_l] + (1-diff_ca/rng) * arr[x1_pos_r]

            new_arr[i] = new_val    

    return new_arr

def downsample_1d_arr(arr: ElevArr, new_len: int) -> ElevArr:
    if new_len > len(arr):
        return upsample_1d_arr(arr, new_len)
    elif new_len == len(arr):
        return arr
    
    new_arr = np.zeros(new_len, dtype=arr.dtype)
    x1 = np.linspace(0, 1, len(arr))
    x2 = np.linspace(0, 1, len(new_arr))
    
    for i, x2_pos_val in enumerate(x2):
        x1_pos = np.argwhere(x1 == x2_pos_val)
        if len(x1_pos) > 0:
            x1_pos = x1_pos[0,0]
            new_arr[i] = arr[x1_pos]
        else:
            x1_pos_l = np.argwhere(x1 < x2_pos_val)[-1,0]
            x1_pos_r = np.argwhere(x1 > x2_pos_val)[0,0]
            a, b, c = x2_pos_val, x1[x1_pos_l], x1[x1_pos_r]
            rng = c - b
            diff_ab = a - b
            diff_ca = c - a
            new_val = diff_ab/rng * arr[x1_pos_l] + diff_ca/rng * arr[x1_pos_r]
            new_arr[i] = new_val    
    return new_arr


