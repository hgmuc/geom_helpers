from __future__ import annotations
from math import floor
from geom_helpers.osm_reader_helper import (
    TOTAL_CELLS, BASE_CELLS, AZOR_CELLS, MADE_CELLS, CANA_CELLS, 
    CABO_CELLS, ICEL_CELLS, MEDI_CELLS, LATLON_CELL_PARAMS)
from geom_helpers.latlon_code import get_reg_cell_code, get_rev_reg_cell_code, get_coord_cell
from numba import njit
from typing import Callable, cast

from basic_helpers.types_base import CoordinateInt, Coordinate, CoordVal, FlexNumeric
from basic_helpers.config_reg_code import Cell, RegCode, ParamsKey, CellParams

LatInt = int
LonInt = int

BaseFunc = Callable[[Coordinate], CoordinateInt]

RC = tuple[int, int]
GridWH = tuple[int, int]
RCGridMap = dict[LatInt, RC]
RCGridNghbrs = dict[GridWH, dict[RC, list[list[RC]]]]
SubRCConfig = tuple[int, int, int, int, BaseFunc, FlexNumeric, FlexNumeric, int, int, ParamsKey]
RC_GRID_NGHBRS: RCGridNghbrs = {}

@njit(fastmath=True)
def get_base_latlon_icel(x: Coordinate) -> CoordinateInt:
    lat, lon = x
    diff_lon = abs(-26-lon)
    base_lon = -26 + int(diff_lon/4) * 4
    #print(f"{lon:.3f}  {diff_lon:.3f}  {base_lon}")
    return int(lat), base_lon

def get_base_latlon_cabo(x: Coordinate) -> CoordinateInt: # type: ignore
    return 14, -27 # - 1e-7

def get_base_latlon_made(x) -> CoordinateInt: # type: ignore
    return 32, -18

def get_base_latlon_azor(x: Coordinate) -> CoordinateInt: # type: ignore
    return 36, -32

def get_base_latlon_cana(x: Coordinate) -> CoordinateInt: # type: ignore
    return 26, -19

def base_default_func(x: Coordinate) -> CoordinateInt:
    return (int(x[0]), floor(x[1]))

def get_num_rows_cols(lat: CoordVal, lon: CoordVal) -> SubRCConfig:
    base_func: BaseFunc = base_default_func
    if (lat, lon) in MEDI_CELLS:
        grid_num_w, grid_num_h = LAT_RC_GRID_MAP[lat]
        params_key: ParamsKey = 'base'   # Es gibt keine speziellen Parameter für das Mittelmeer
        #params: CellParams = LATLON_CELL_PARAMS['base']
    elif (lat, lon) in CANA_CELLS:
        base_func = get_base_latlon_cana
        grid_num_w, grid_num_h = LAT_RC_GRID_MAP_CANA[lat]
        params_key = 'canary'
    elif (lat, lon) in ICEL_CELLS:
        base_func = get_base_latlon_icel
        grid_num_w, grid_num_h = LAT_RC_GRID_MAP_ICEL[lat]
        params_key = 'iceland'
    elif (lat, lon) in AZOR_CELLS:
        base_func = get_base_latlon_azor
        grid_num_w, grid_num_h = LAT_RC_GRID_MAP_AZOR[lat]
        params_key = 'azores'
    elif (lat, lon) in CABO_CELLS:
        base_func = get_base_latlon_cabo
        grid_num_w, grid_num_h = LAT_RC_GRID_MAP_CABO[lat]
        params_key = 'caboverde'
    elif (lat, lon) in MADE_CELLS:
        base_func = get_base_latlon_made
        grid_num_w, grid_num_h = LAT_RC_GRID_MAP_MADE[lat]
        params_key = 'madeira'
    elif (lat, lon) in BASE_CELLS:
        grid_num_w, grid_num_h = LAT_RC_GRID_MAP[lat]
        params_key = 'base'
    else:
        return cast(SubRCConfig, ())

    params: CellParams = LATLON_CELL_PARAMS[params_key]
    num_rows = params['n_lttr2'] * params['y2'] * params['y3']
    num_cols = params['n_lttr1'] * params['x2'] * params['x3']

    return (grid_num_w, grid_num_h, num_rows, num_cols, base_func, 
            params['x1'], params['y1'], params['n_lttr1'], params['n_lttr2'], params_key)


def get_sub_rc(lat: CoordVal, lon: CoordVal):
    grid_num_w, grid_num_h, num_rows, num_cols, base_func, param_x1, param_y1, param_n1, param_n2 = cast(SubRCConfig, NUM_ROWS_COLS[(int(lat), floor(lon))])[:9]
    base_lat, base_lon = base_func((lat,lon))
    return compute_sub_rc(lat, lon, grid_num_w, grid_num_h, num_rows, num_cols, param_x1, param_y1, param_n1, param_n2, base_lat, base_lon)

def get_sub_rc_code(lat: CoordVal, lon: CoordVal):
    grid_num_w, grid_num_h, num_rows, num_cols, base_func, param_x1, param_y1, param_n1, param_n2 = cast(SubRCConfig, NUM_ROWS_COLS[(int(lat), floor(lon))])[:9]
    base_lat, base_lon = base_func((lat,lon))
    code = get_reg_cell_code(lat, lon)
    r, c = compute_sub_rc(lat, lon, grid_num_w, grid_num_h, num_rows, num_cols, param_x1, param_y1, param_n1, param_n2, base_lat, base_lon)
    return f"{code[:7]}{r}{c}"


def get_sub_rc_all_nghbrs(code: RegCode, r: int, c: int):
    grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
    if (grid_num_w, grid_num_h) in RC_GRID_NGHBRS:
        if (r, c) in RC_GRID_NGHBRS[grid_num_w, grid_num_h]:
            print("FOUND", (grid_num_w, grid_num_h), (r, c))
            return RC_GRID_NGHBRS[grid_num_w, grid_num_h][r, c]
    else:
        RC_GRID_NGHBRS[grid_num_w, grid_num_h] = {}

    nghbrs_rc = [get_sub_rc_nghbrs(r, c, grid_num_w, grid_num_h)]
    nghbr_left = get_sub_rc_nghbr(r, c, -1, True, grid_num_w, grid_num_h)
    nghbr_right = get_sub_rc_nghbr(r, c, 1, True, grid_num_w, grid_num_h)
    nghbr_down = get_sub_rc_nghbr(r, c, -1, False, grid_num_w, grid_num_h)
    nghbr_up = get_sub_rc_nghbr(r, c, 1, False, grid_num_w, grid_num_h)
    #nghbrs_rc = [get_sub_rc_nghbrs(code, r, c, grid_num_w, grid_num_h)]
    #nghbr_left = get_sub_rc_nghbr(code, r, c, -1, True, grid_num_w, grid_num_h)
    #nghbr_right = get_sub_rc_nghbr(code, r, c, 1, True, grid_num_w, grid_num_h)
    #nghbr_down = get_sub_rc_nghbr(code, r, c, -1, False, grid_num_w, grid_num_h)
    #nghbr_up = get_sub_rc_nghbr(code, r, c, 1, False, grid_num_w, grid_num_h)
    
    for nghbr_rc in [nghbr_left, nghbr_right, nghbr_down, nghbr_up]:
        #print("nghbr_rc", nghbr_rc, nghbr_rc != (r, c))
        if nghbr_rc != (r, c):
            nghbrs_rc.append([(r, c), nghbr_rc])

    res_ls = [ls for ls in nghbrs_rc if ls]
    RC_GRID_NGHBRS[grid_num_w, grid_num_h][r, c] = res_ls
    print("RC_GRID_NGHBRS", len(RC_GRID_NGHBRS), len(RC_GRID_NGHBRS[grid_num_w, grid_num_h]))
    return res_ls

def get_sub_rc_nghbr(r: int, c: int, dval: int, horizontal: bool, 
                     grid_num_w: int, grid_num_h: int) -> tuple[int, int]:
    #grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
    if horizontal:
        #print("A", grid_num_w, c, dval, ":", r)
        return r, min(max(0, c + dval), grid_num_w-1)
    else:
        #print("B", grid_num_h, r, dval, ":", c)
        return min(max(0, r + dval), grid_num_h-1), c

def get_sub_rc_nghbrs(r: int, c: int, grid_num_w: int, grid_num_h: int) -> list[tuple[int, int]]:
    #grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
    if r < grid_num_h-1 and c < grid_num_w-1:
        return [(r, c), (r+1, c), (r, c+1), (r+1, c+1)]
    else:
        return []

def get_sub_rc_x_nghbrs(code: RegCode, r: int, c: int) -> list[tuple[int, int]]:
    grid_num_w, grid_num_h = NUM_ROWS_COLS_BY_CELL[code[:2]][:2]
    ls = [(min(max(0, r - 1), grid_num_h-1), min(max(0, c - 1), grid_num_w-1)), 
          (min(max(0, r + 1), grid_num_h-1), min(max(0, c - 1), grid_num_w-1)), 
          (min(max(0, r + 1), grid_num_h-1), min(max(0, c + 1), grid_num_w-1)), 
          (min(max(0, r - 1), grid_num_h-1), min(max(0, c + 1), grid_num_w-1))]
    return [x for x in ls if x != (r, c) and x != (r, c-1) and x != (r, c+1) and x != (r-1, c) and x != (r+1, c)]


'''
                 x1   x2   x3   y1   y2   y3  n_lttr1  n_lttr2  off1  off2
base       2.000000  3.0  9.0  3.0  3.0  9.0      2.0      3.0   1.0   1.0
azores     1.000000  6.0  9.0  1.5  6.0  9.0      8.0      6.0   0.0   0.0
madeira    3.000000  3.0  9.0  3.0  3.0  9.0      6.0      3.0   0.0   0.0
canary     1.000000  7.0  9.0  1.0  8.0  9.0      7.0      7.0   0.0   0.0
caboverde  1.333333  6.0  9.0  1.4  6.0  9.0      8.0      6.0   0.0   0.0
iceland    1.500000  3.0  9.0  3.0  3.0  9.0      6.0      3.0   0.0   0.0
'''

@njit(fastmath=True)
def compute_sub_rc(lat: CoordVal, lon: CoordVal, grid_num_w: int, grid_num_h: int, 
                   num_rows: int, num_cols: int, x1: FlexNumeric, y1: FlexNumeric, 
                   n1: int, n2: int, base_lat: int, base_lon: int) -> RC:
    lat_diff = (lat - base_lat) / (n2 / y1)
    lat_col = int(lat_diff * num_rows)
    col_lat_bott = lat_col / num_rows
    sub_row_num = abs(int((lat_diff - col_lat_bott) * num_rows * grid_num_h))
    
    lon_diff = (lon - base_lon) / (n1 / x1)
    lon_col = int(lon_diff * num_cols)
    col_lon_left = lon_col / num_cols
    sub_col_num = abs(int((lon_diff - col_lon_left) * num_cols * grid_num_w))

    return sub_row_num, sub_col_num


def get_rev_sub_rc(code: RegCode, r: int, c: int) -> Coordinate:
    rev_base_lat, rev_base_lon = get_rev_reg_cell_code(f"{code[:7]}0000")
    grid_num_w, grid_num_h, num_rows, num_cols = cast(SubRCConfig, 
                                                      NUM_ROWS_COLS[(int(rev_base_lat), 
                                                                     floor(rev_base_lon))])[:4]
    rev_lon = rev_base_lon + (c + 0.5) / (grid_num_w * num_cols) + 1e-6
    rev_lat = rev_base_lat + (r + 0.5) / (grid_num_h * num_rows) + 1e-6
    return rev_lat, rev_lon


LAT_RC_GRID_MAP: RCGridMap = {
    0: (10, 7), 1: (10, 7), 2: (10, 7), 3: (10, 7), 4: (10, 7), 5: (10, 7), 6: (10, 7), 7: (10, 7), 8: (10, 7), 
    9: (10, 7), 10: (10, 7), 11: (10, 7), 12: (10, 7), 13: (10, 7), 14: (10, 7), 15: (10, 7), 16: (10, 7), 
    17: (10, 7), 18: (10, 7), 19: (10, 7), 20: (10, 7), 21: (10, 7), 22: (10, 7), 23: (10, 7), 24: (10, 7), 
    25: (9, 7), 26: (9, 7), 27: (9, 7), 28: (9, 7), 29: (9, 7), 30: (9, 7), 31: (9, 7), 32: (10, 8), 33: (10, 8), 
    34: (10, 8), 35: (10, 8), 36: (10, 8), 37: (10, 8), 38: (9, 8), 39: (9, 8), 40: (9, 8), 41: (9, 8), 42: (10, 9), 
    43: (10, 9), 44: (10, 9), 45: (9, 9), 46: (9, 9), 47: (9, 9), 48: (9, 9), 49: (10, 10), 50: (10, 10), 51: (9, 10), 
    52: (9, 10), 53: (9, 10), 54: (7, 8), 55: (7, 8), 56: (7, 8), 57: (8, 10), 58: (8, 10), 59: (7, 9), 60: (6, 8), 
    61: (6, 8), 62: (7, 10), 63: (6, 9), 64: (6, 9), 65: (5, 8), 66: (6, 10), 67: (6, 10), 68: (5, 9), 69: (5, 9), 
    70: (4, 8), 71: (5, 10), 72: (4, 9), 73: (4, 9), 74: (4, 10), 75: (4, 10), 76: (3, 8), 77: (3, 9), 78: (3, 10), 
    79: (3, 10), 80: (2, 8), 81: (2, 9), 82: (2, 10), 83: (2, 10), 84: (1, 8), 85: (1, 8), 86: (1, 10), 87: (1, 10), 
    88: (1, 10), 89: (1, 10)}

LAT_RC_GRID_MAP_ICEL: RCGridMap = {62: (8, 9), 63: (8, 9), 64: (7, 8), 65: (7, 8), 66: (7, 9), 67: (7, 9)}
LAT_RC_GRID_MAP_AZOR: RCGridMap = {36: (10, 8), 37: (10, 8), 38: (9, 8), 39: (9, 8)}
LAT_RC_GRID_MAP_CABO: RCGridMap = {14: (9, 9), 15: (9, 9), 16: (9, 9), 17: (9, 9), 18: (9, 9)}
LAT_RC_GRID_MAP_CANA: RCGridMap = {26: (9, 9), 27: (9, 9), 28: (9, 9), 29: (9, 9), 30: (9, 9)}
LAT_RC_GRID_MAP_MADE: RCGridMap = {32: (9, 9), 33: (9, 9)}

# Durch TOTAL_CELLS muss rückwärts iteriert werden, damit bei Überschneidungen die richtige Region zugewiesen wird
NUM_ROWS_COLS: dict[CoordinateInt, tuple[int, int]] = {
    (lat, lon): get_num_rows_cols(lat, lon) for lat, lon in sorted(TOTAL_CELLS)[::-1]    # type: ignore
    if get_num_rows_cols(lat, lon)}
NUM_ROWS_COLS_BY_CELL: dict[Cell, tuple[int, int]] = {
    get_coord_cell(k[0], k[1]): v for k, v in NUM_ROWS_COLS.items()}

'''
from latlon_code import get_reg_cell_code, get_rev_reg_cell_code
from distance_helper import get_dist


#lat = 14.7

#for lon in range(12000,12100):
#for lon in range(18055,18112,2):
for i, lon in enumerate(range(26500,21001,-1)):
#for lon in range(2075,2094,2):
    if i % 125 not in [0,1,124]:
        continue
    if i % 125 == 124:
        print()
    #print(i)
    lat = 15.01 + i / 1000
    if lat >= 19:
        break
    lon /= -1000
    grid_num_w, grid_num_h, num_rows, num_cols, base_func = NUM_ROWS_COLS[(int(lat), floor(lon))][:5]
    code = get_reg_cell_code(lat, lon + 1e-7)
    rev_code_lat, rev_code_lon = get_rev_reg_cell_code(code)
    r, c = compute_sub_rc(lat, lon, grid_num_w, grid_num_h, num_rows, num_cols, *base_func((lat, lon)))
    rev_base_lat, rev_base_lon = get_rev_sub_rc(code, r, c)
    d = get_dist((lat, lon), (rev_base_lat, rev_base_lon))
    if d > 10:
        fstr1 = f"{lat:.3f}, {lon:.3f}  {code[:7]}  {i:5} {r} {c}   {code[7:]}  {rev_base_lat:.6f}, {rev_base_lon:.6f}"
        fstr2 = f"   {d:6.1f}  {rev_code_lat:.6f}, {rev_code_lon:.6f}"
        x = (c + 0.5) / (grid_num_w * num_cols) + 1e-6
        fstr3 = f" |  {x:.6f} {rev_base_lon:.6f} |  {rev_base_lon + (c + 0.5) / (grid_num_w * num_cols) + 1e-6:.6f}"
        #print(f"{lat:.3f}, {lon:.3f}  {code[:7]}  {i:5} {r} {c}   {code[7:]}  {rev_lat:.6f}, {rev_lon:.6f}   {d:6.1f}  {rev_code_lat:.6f}, {rev_code_lon:.6f}")
        print(f"{i:4}", fstr1, fstr2)



for i, lon in enumerate(range(8000,12001,1)):
#for lon in range(2075,2094,2):
    #print(i)
    lat = 64.01 + i / 1000
    if lat >= 68:
        break
    lon /= 1000
    grid_num_w, grid_num_h, num_rows, num_cols, base_func = NUM_ROWS_COLS[(int(lat), floor(lon))][:5]
    code = get_reg_cell_code(lat, lon + 1e-8)
    rev_code_lat, rev_code_lon = get_rev_reg_cell_code(code)
    r, c = compute_sub_rc(lat, lon, grid_num_w, grid_num_h, num_rows, num_cols, *base_func((lat, lon)))
    rev_base_lat, rev_base_lon = get_rev_sub_rc(code, r, c)
    d = get_dist((lat, lon), (rev_base_lat, rev_base_lon))
    if d > 105:
        fstr1 = f"{lat:.3f}, {lon:.3f}  {code[:7]}  {i:5} {r} {c}   {code[7:]}  {rev_base_lat:.6f}, {rev_base_lon:.6f}"
        fstr2 = f"   {d:6.1f}  {rev_code_lat:.6f}, {rev_code_lon:.6f}"
        x = (c + 0.5) / (grid_num_w * num_cols) + 1e-6
        fstr3 = f" |  {x:.6f} {rev_base_lon:.6f} |  {rev_base_lon + (c + 0.5) / (grid_num_w * num_cols) + 1e-6:.6f}"
        #print(f"{lat:.3f}, {lon:.3f}  {code[:7]}  {i:5} {r} {c}   {code[7:]}  {rev_lat:.6f}, {rev_lon:.6f}   {d:6.1f}  {rev_code_lat:.6f}, {rev_code_lon:.6f}")
        print(f"{i:4}", fstr1, fstr2)

        
'''