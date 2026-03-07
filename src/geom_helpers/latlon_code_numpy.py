from __future__ import annotations
import pandas as pd
import numpy as np
import numpy.typing as npt
from typing import TypeAlias, Callable, Any, cast
# import math, os, json, pickle
# from itertools import product
from geom_helpers.latlon_code import (get_len_m_lttrs_by_lat, get_reg_cell_base_coords, REV_MAP_CELL)

from basic_helpers.config_reg_code import (AZOR_CELLS, MADE_CELLS, CANA_CELLS, CABO_CELLS, 
                                           ICEL_CELLS, MEDI_CELLS, LATLON_CELL_PARAMS, CellParams)

from geom_helpers.osm_reader_helper import get_coord_cell_numba

from basic_helpers.types_base import CoordVal, FlexNumeric #, Coordinate, CoordinateInt, BBox

VectorizedStringFunc: TypeAlias = Callable[[npt.NDArray[Any], npt.NDArray[Any]], npt.NDArray[np.str_]]

LTTRS: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
LTTRS_ARR = np.array(LTTRS)
LTTRS_MAP_DICT: dict[str, int] = {c: i for i, c in enumerate(LTTRS)}

def get_char_at_index(s: str, i: int) -> str:
    return s[i]

def get_substring_from_string(s: str, len_seq: int) -> str:
    return s[:len_seq]

def get_latlon_params_for_point(lat: CoordVal, lon: CoordVal) -> CellParams:
    """
    Returns the appropriate LATLON_CELL_PARAMS dictionary based on
    the integer latitude and longitude.
    """
    #print("lat, lon", lat, lon, (lat, lon) in MEDI_CELLS, (lat, lon) in CANA_CELLS, (lat, lon) in ICEL_CELLS,
    #      (lat, lon) in AZOR_CELLS, (lat, lon) in CABO_CELLS, (lat, lon) in MADE_CELLS)
    #print("x", lat, lon)
    if (lat, lon) in MEDI_CELLS:
        params = LATLON_CELL_PARAMS['base'].copy()
        base_lat = int(lat)
        base_lon = int(lon)
    elif (lat, lon) in CANA_CELLS:
        params = LATLON_CELL_PARAMS['canary'].copy()
        base_lat, base_lon = get_reg_cell_base_coords(CANA_CELLS)
    elif (lat, lon) in ICEL_CELLS:
        params = LATLON_CELL_PARAMS['iceland'].copy()
        base_lat = int(lat)
        #base_lon = -26 - ((-26-lon)//4+1)*4  => FEHLERHAFT
        base_lon = -26 - 4 * int((-26-lon)/4)
    elif (lat, lon) in AZOR_CELLS:
        params = LATLON_CELL_PARAMS['azores'].copy()
        base_lat, base_lon = get_reg_cell_base_coords(AZOR_CELLS)
    elif (lat, lon) in CABO_CELLS:
        params = LATLON_CELL_PARAMS['caboverde'].copy()
        base_lat, base_lon = get_reg_cell_base_coords(CABO_CELLS)
    elif (lat, lon) in MADE_CELLS:
        params = LATLON_CELL_PARAMS['madeira'].copy()
        base_lat, base_lon = get_reg_cell_base_coords(MADE_CELLS)
    else:
        params = LATLON_CELL_PARAMS['base'].copy()
        base_lat = int(lat)
        base_lon = int(lon)
        #print("base", base_lat, base_lon)
    
    params['base_lat'] = base_lat
    params['base_lon'] = base_lon
    #params['move_lon'] = move_lon
    params['len_lttrs'] = get_len_m_lttrs_by_lat(base_lat)
    return params

# A more generic alias if the arrays could be float or int
NumericArray: TypeAlias = npt.NDArray[np.number]
AllParams1D = tuple[NumericArray, ...]

## OLD VERSION - likely also OK
#params1DArray: TypeAlias = npt.NDArray[np.number]
#AllParams1D = tuple[params1DArray, params1DArray, params1DArray, 
#                    params1DArray, params1DArray, params1DArray, 
#                    params1DArray, params1DArray, params1DArray, 
#                    params1DArray, params1DArray, params1DArray, params1DArray]


# SEE A WAY HOW TO SPEED UP THIS PROCESS WITH STRUCTURED ARRAYS at the end of this module:
def get_params_from_array(params: npt.NDArray[np.object_]) -> AllParams1D:
    # Converting the array of TypedDicts to a DataFrame
    params_df = pd.DataFrame(list(params))
    '''
    THE OLD VERSION created extra Numpy arrays - better without
    param_x1: NumericArray = params_df.x1.values
    param_x2: NumericArray = params_df.x2.values
    param_x3: NumericArray = params_df.x3.values
    param_y1: NumericArray = params_df.y1.values
    param_y2: NumericArray = params_df.y2.values
    param_y3: NumericArray = params_df.y3.values
    param_nlttr1: NumericArray = params_df.n_lttr1.values
    param_nlttr2: NumericArray = params_df.n_lttr2.values
    param_off1: NumericArray = params_df.off1.values
    param_off2: NumericArray = params_df.off2.values
    base_lat: NumericArray = params_df.base_lat.values
    base_lon: NumericArray= params_df.base_lon.values
    len_lttrs: NumericArray = params_df.len_lttrs.values
    return (param_x1, param_x2, param_x3, param_y1, param_y2, param_y3, param_nlttr1, 
            param_nlttr2, param_off1, param_off2, base_lat, base_lon, len_lttrs)
    '''
    
    # We return them as a tuple of arrays
    # .values returns the underlying numpy array
    return (params_df.x1.values, params_df.x2.values, params_df.x3.values, 
            params_df.y1.values, params_df.y2.values, params_df.y3.values, 
            params_df.n_lttr1.values, params_df.n_lttr2.values, 
            params_df.off1.values, params_df.off2.values, 
            params_df.base_lat.values, params_df.base_lon.values, params_df.len_lttrs.values)

def get_reg_cell_code_by_params_numpy(lats: npt.NDArray[np.number], 
                                      lons: npt.NDArray[np.number], 
                                      params: npt.NDArray[np.object_], 
                                      M: FlexNumeric = 111320) -> list[str]:
    res: AllParams1D = get_params_from_array(params)
    (param_x1, param_x2, param_x3, param_y1, param_y2, param_y3, param_nlttr1, 
     param_nlttr2, param_off1, param_off2, base_lat, base_lon, len_lttrs) = res

    #lons = lons
    M_REFs: npt.NDArray[np.floating] = np.cos(np.radians(lats)) * M / (param_x1 * param_x2 * param_x3)
    # Decimal parts
    #lat_dec = lats - np.floor(lats) # + 1e-7
    #lon_dec = np.abs(lons - np.floor(lons)) # + 1e-7
    lat_dec: npt.NDArray[np.number] = lats - base_lat
    lon_dec: npt.NDArray[np.number] = np.abs(lons - base_lon)
    #print("lat_dec / lon_dec", np.round(lat_dec, 2), np.round(lon_dec, 2))

    # Edge-case adjustments (vectorized with np.where)
    lat_dec = np.where(lat_dec == 0, lat_dec + 1e-7, lat_dec)
    lats = np.where(lat_dec == 0, lats + 1e-7, lats)
    lon_dec = np.where(lon_dec == 0, lon_dec + 1e-7 * np.sign(lons), lon_dec)
    lons = np.where(lon_dec == 0, lons + 1e-7 * np.sign(lons), lons)

    # Precompute some integer breakdowns
    y1: npt.NDArray[np.integer] = (lat_dec * param_y1).astype(int)
    lat_dec = lat_dec * param_y1 - y1
    #print("lat_dec / lon_dec", np.round(lat_dec, 2), np.round(lon_dec, 2))
    y2: npt.NDArray[np.integer] = (lat_dec * param_y2).astype(int)
    lat_dec = lat_dec * param_y2 - y2
    y3: npt.NDArray[np.integer] = (lat_dec * param_y3).astype(int)
    lat_dec = lat_dec * param_y3 - y3

    x1: npt.NDArray[np.integer] = (lon_dec * param_x1).astype(int)
    lon_dec = lon_dec * param_x1 - x1
    #print("lat_dec / lon_dec", np.round(lat_dec, 2), np.round(lon_dec, 2))
    x2: npt.NDArray[np.integer] = (lon_dec * param_x2).astype(int)
    lon_dec = lon_dec * param_x2 - x2
    x3: npt.NDArray[np.integer] = (lon_dec * param_x3).astype(int)
    lon_dec = lon_dec * param_x3 - x3

    # Example vectorized metric computations
    lat_res_M: npt.NDArray[np.integer] = np.round(lat_dec * M / (param_y1 * param_y2 * param_y3)).astype(int)
    lon_res_M: npt.NDArray[np.integer] = np.round(lon_dec * M_REFs).astype(int)

    cells: npt.NDArray[np.str_] = vec_get_coord_cell(lats, lons)
    lttrs: npt.NDArray[np.str_] = vec_get_substring(LTTRS_ARR, len_lttrs)

    c3: npt.NDArray[np.str_] = vec_get_char(LTTRS_ARR, y1 * param_nlttr1 + x1 + param_off1)
    c4: npt.NDArray[np.str_] = vec_get_char(LTTRS_ARR, y2 * param_nlttr2 + x2 + param_off2)
    c6: npt.NDArray[np.str_] = (y3 + 1).astype(str)
    c7: npt.NDArray[np.str_] = (x3 + 1).astype(str)

    idx_lat: npt.NDArray[np.integer] = (lat_res_M // len_lttrs.astype(int))
    c8: npt.NDArray[np.str_] = vec_get_char(lttrs, idx_lat)
    
    idx_lat = (lat_res_M % len_lttrs.astype(int))
    c9: npt.NDArray[np.str_] = vec_get_char(lttrs, idx_lat) # (lat_res_M % len_lttrs).astype(int))

    idx_lat = (lon_res_M // len_lttrs.astype(int))
    c10: npt.NDArray[np.str_] = vec_get_char(lttrs, idx_lat)

    idx_lat = (lon_res_M % len_lttrs.astype(int))
    c11: npt.NDArray[np.str_] = vec_get_char(lttrs, idx_lat)

    comp1: npt.NDArray[np.str_] = np.char.add(np.char.add(cells, c3), 
                                              np.char.add(c4, '-'))
    comp2: npt.NDArray[np.str_] = np.char.add(np.char.add(c6, c7), 
                                              np.char.add(c8, 
                                                          np.char.add(c9, 
                                                                      np.char.add(c10, c11))))
    # Fallback: loop only for string assembly
    codes: npt.NDArray[np.str_] = np.char.add(comp1, comp2)
    
    return cast(list[str], codes.tolist())

def get_rev_params_from_array(rev_params):
    params_df = pd.DataFrame(list(rev_params))
    param_x1 = params_df.x1.values
    param_x2 = params_df.x2.values
    param_x3 = params_df.x3.values
    param_y1 = params_df.y1.values
    param_y2 = params_df.y2.values
    param_y3 = params_df.y3.values
    param_nlttr1 = params_df.n_lttr1.values
    param_nlttr2 = params_df.n_lttr2.values
    param_off1 = params_df.off1.values
    param_off2 = params_df.off2.values
    #base_lat = params_df.base_lat.values
    #base_lon= params_df.base_lon.values
    len_lttrs = params_df.len_lttrs.values
    #print("base_lat", base_lat)
    #print("base_lon", base_lon)
    return (param_x1, param_x2, param_x3, param_y1, param_y2, param_y3, 
            param_nlttr1, param_nlttr2, param_off1, param_off2, len_lttrs)

def get_rev_reg_cell_code_by_params_numpy(codes, M=111320):
    codes_arr = np.array([list(REV_MAP_CELL[code[:2]][0]) + [LTTRS_MAP_DICT[c] for c in code[2:4] + code[5:]] for code in codes])
    base_lats = codes_arr[:, 0].astype(int)
    base_lons = codes_arr[:, 1].astype(int)

    rev_params = vec_get_latlon_params_for_point(base_lats.astype(int), base_lons.astype(int))
    (param_x1, param_x2, param_x3, 
     param_y1, param_y2, param_y3, 
     param_nlttr1, param_nlttr2,  
     param_off1, param_off2, len_lttrs) = get_rev_params_from_array(rev_params)

    x1 = (codes_arr[:, 2] - param_off1) % param_nlttr1
    x2 = (codes_arr[:, 3] - param_off2) % param_nlttr2
    x3 = (codes_arr[:, 5] - 1) % param_x3
    frac_x1 = 1 / param_x1
    frac_x2 = 1 / (param_x1 * param_x2)
    frac_x3 = 1 / (param_x1 * param_x2 * param_x3)

    y1 = (codes_arr[:, 2] - param_off1) // param_nlttr1
    y2 = (codes_arr[:, 3] - param_off2) // param_nlttr2
    y3 = (codes_arr[:, 4] - 1) % param_y3
    frac_y1 = 1 / param_y1
    frac_y2 = 1 / (param_y1 * param_y2)
    frac_y3 = 1 / (param_y1 * param_y2 * param_y3)

    lat_adds = (codes_arr[:, 6] * len_lttrs + codes_arr[:, 7]) / M
    lats = base_lats + y1 * frac_y1 + y2 * frac_y2 + y3 * frac_y3 + lat_adds
    
    M_REFs = np.cos(np.radians(lats)) * M
    lon_adds = (codes_arr[:, 8] * len_lttrs + codes_arr[:, 9]) / M_REFs
    lons = base_lons + x1 * frac_x1 + x2 * frac_x2 + x3 * frac_x3 + lon_adds

    return list(zip(np.round(lats, 9), np.round(lons, 9)))
    #return list(zip(np.float32(lats), np.float32(lons)))


### DISTANCES
def get_dist_numpy(pts1, pts2, M=111320): # 111229.83
    '''
    Input:
    - pts1, pts2: Zwei Numpy Arrays mit gleicher Länge
    - Jedes Array enthält die Koordinaten (Breite, Länge, Höhe) oder (Breite, Länge) von Trackpunkten (Ausgangspunkt - Zielpunkt),
    z.B.  np.array([(48.0552, 11.723, 510), (...)]), np.array([(48.0552, 11.723), (...)])
    
    Output:
    - Array mit Distanzen in M (float) zwischen den beiden Punkten
    
    * Ermittelt die Entfernung zwischen Ausgangspunkt und Zielpunkt mit Hilfe des Satzes von Pythagoras
    * berücksichtigt die spherische Form der Erde (nur geringfügiger Unterschied zur vergleichbaren "Haversine Distance")
        - Unterschied beträgt annähernd konstant 0.07%
        - auf 1km = 70cm
        - auf 200km = 140m
    * berücksichtigt auch einen etwaigen Höhenunterschied zwischen Ausgangspunkt und Zielpunkt (wobei der Einfluss eher gering ist)
    '''
    lats1 = pts1[:, 0]
    lons1 = pts1[:, 1]
    lats2 = pts2[:, 0]
    lons2 = pts2[:, 1]
    if pts1.shape[1] > 2:
        h = (pts1[:, 2] - pts2[:, 2])**2
    else:
        h = np.zeros(len(lats1))

    a2 = ((lats1 - lats2) * M)**2
    b2 = ((lons1 - lons2) * M * np.cos(np.radians(0.5*lats1 + 0.5*lats2)))**2
    c2 = a2 + b2
    
    return np.sqrt(c2+h)

def get_reg_cell_code_numpy(lats: npt.NDArray[np.floating], lons: npt.NDArray[np.floating]):
    params = vec_get_latlon_params_for_point(np.floor(lats).astype(int), np.floor(lons).astype(int))
    codes = get_reg_cell_code_by_params_numpy(lats, lons, params)
    return codes

'''
arr: npt.NDArray[np.numeric] = np.array(
    [[48.137154, 11.576124], [52.520008, 13.404954], [51.507351, -0.127758], 
     [55.755825, 37.617298], [65.755825, -20.617298], [33.755825, -17.617298],
     [38.755825, -28.617298], [27.755825, -16.617298], [16.25, -23.8], [48.856613, 2.352222]])

lats: npt.NDArray[np.numeric]  = arr[:, 0]
lons: npt.NDArray[np.numeric]  = arr[:, 1]
'''

vec_get_char: VectorizedStringFunc = np.vectorize(get_char_at_index)
vec_get_coord_cell: VectorizedStringFunc = np.vectorize(get_coord_cell_numba)
# vec_get_lttrs_by_lat = np.vectorize(get_lttrs_by_lat)
vec_get_substring: VectorizedStringFunc = np.vectorize(get_substring_from_string)
#vec_len = np.vectorize(len)
#vec_get_dist = np.vectorize(get_dist)
vec_get_latlon_params_for_point = np.vectorize(get_latlon_params_for_point, otypes=[object])

'''
params = vec_get_latlon_params_for_point(np.floor(lats).astype(int), np.floor(lons).astype(int))

codes = get_reg_cell_code_by_params_numpy(lats, lons, params)
codes

#codes_arr = np.array([list(code) for code in codes])
#cells_arr = np.char.add(codes_arr[:, 0], codes_arr[:, 1])
#base_latlons = np.array([get_rev_coord_cell(cell)[0] for cell in cells_arr])
codes_arr = np.array([list(REV_MAP_CELL[code[:2]][0]) + [LTTRS_MAP_DICT[c] for c in code[2:4] + code[5:]] for code in codes])
base_lats = codes_arr[:, 0].astype(int)
base_lons = codes_arr[:, 1].astype(int)

rev_params = vec_get_latlon_params_for_point(base_lats.astype(int), base_lons.astype(int))
pd.DataFrame(list(rev_params))

rev_arr = get_rev_reg_cell_code_by_params_numpy(codes)
np.round(get_dist_numpy(arr, np.array(rev_arr)),1)
'''


# Define the memory layout
CELL_PARAMS_DTYPE = np.dtype([
    ('x1', 'f8'),           # 64-bit float
    ('x2', 'i8'),           # 64-bit int
    ('x3', 'i8'),
    ('y1', 'f8'),
    ('y2', 'i8'),
    ('y3', 'i8'),
    ('extent_lon', 'i8'),
    ('extent_lat', 'i8'),
    ('n_lttr1', 'i8'),
    ('n_lttr2', 'i8'),
    ('off1', 'i8'),
    ('off2', 'i8'),
    ('len_lttrs', 'i8'),
    ('base_lat', 'i8'),
    ('base_lon', 'i8'),
])

def get_params_from_array_fast_new(params: npt.NDArray[Any]) -> AllParams1D:
    """
    Assumes 'params' is a NumPy Structured Array with CELL_PARAMS_DTYPE.
    """
    # No more pd.DataFrame(list(params))! 
    # Direct field access is O(1) and returns a view of the memory.
    return (
        params['x1'], params['x2'], params['x3'],
        params['y1'], params['y2'], params['y3'],
        params['n_lttr1'], params['n_lttr2'],
        params['off1'], params['off2'],
        params['base_lat'], params['base_lon'],
        params['len_lttrs']
    )


'''
## SPEED UP with STRUCTURED ARRAY

To get that $10\times$ (or more) speedup, we need to move away from **Python Objects** (dictionaries) and **Pandas overhead**.

When you store `TypedDicts` in a NumPy array of `dtype=object`, every time you access a value, Python has to "unwrap" the object, look up the key in a hash map, and then return the value.

The "Senior Dev" way to do this is using **NumPy Structured Arrays**. This allows you to define a fixed memory layout for your data, similar to a `struct` in C.

### 1. Define the Structured Dtype (SEE ABOVE)

Instead of a `TypedDict`, we define a `np.dtype`. This tells NumPy exactly how many bytes each field takes.

```python
import numpy as np
import numpy.typing as npt


### 2. Optimized `get_params_from_array`

By using a structured array (SEE ABOVE), we eliminate the need for `pd.DataFrame`. Accessing a column becomes a simple memory offset operation, which is nearly instantaneous.


def get_params_from_array_fast(params: npt.NDArray[Any]) -> AllParams1D:
    """
    Assumes 'params' is a NumPy Structured Array with CELL_PARAMS_DTYPE.
    """
    # No more pd.DataFrame(list(params))! 
    # Direct field access is O(1) and returns a view of the memory.
    return (
        params['x1'], params['x2'], params['x3'],
        params['y1'], params['y2'], params['y3'],
        params['n_lttr1'], params['n_lttr2'],
        params['off1'], params['off2'],
        params['base_lat'], params['base_lon'],
        params['len_lttrs']
    )

```

### 3. Comparison of Approaches

| Feature | Your Current Way (Pandas) | Structured Array Way |
| --- | --- | --- |
| **Storage** | Array of Pointers to Dicts | Continuous Block of Memory |
| **Column Access** | $O(N)$ (Creates copy) | $O(1)$ (Direct View) |
| **Memory Usage** | High (Python Dict overhead) | Very Low (Raw bytes) |
| **Type Checking** | `npt.NDArray[np.object_]` | `npt.NDArray[Any]` (or custom DType) |

---

### How to adapt your code:

When you create your initial data (wherever `LATLON_CELL_PARAMS` comes from), you would initialize your array like this:

```python
# Instead of: np.array([dict1, dict2], dtype=object)
# Do this:
data = np.zeros(len(input_coords), dtype=CELL_PARAMS_DTYPE)

# Fill it (example):
data['x1'] = 1.5
data['x2'] = 10
# ... etc

```

### Pro-Tip: Numba Compatibility

Since you mentioned `get_coord_cell_numba`, structured arrays are **significantly** 
easier to pass into Numba functions than dictionaries or Pandas DataFrames. 
Numba understands structured dtypes natively, allowing you to run your entire 
`get_reg_cell_code` logic inside a `@njit` decorated function 
if you wanted to reach maximum theoretical performance.

**Would you like me to help you write a small helper function that converts your existing 
`Dict[str, CellParams]` into this structured array format?**


'''