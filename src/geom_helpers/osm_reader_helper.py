from __future__ import annotations
import numpy as np
import numpy.typing as npt
from numba import njit
#from math import pow, cos, radians, sqrt
#from itertools import product
from typing import Literal, TypeAlias, Callable, overload, cast

#from geom_helpers.distance_helper import get_dist

from basic_helpers.types_base import CoordVal, Coordinate, CoordinateInt # FlexNumeric
from basic_helpers.config_reg_code import (
    CellCode, Cell, Subcell, ParamsKey, # CellSubcell, Subcell, 
    TOTAL_CELLS, BASE_CELLS, AZOR_CELLS, MADE_CELLS, CANA_CELLS, 
    CABO_CELLS, ICEL_CELLS, MEDI_CELLS, LATLON_CELL_PARAMS)


MapFunc: TypeAlias = Callable[[CoordVal, CoordVal], str]
CodeFunc: TypeAlias = Callable[[CoordVal, CoordVal], tuple[int, int, int, int]]
CellFunc: TypeAlias = Callable[[CoordVal, CoordVal], tuple[int, int]]

#from file_helper import do_gzip_pkl, do_ungzip_pkl
class InvalidCoordinateError(Exception):
    """Exception raised for invalid coordinates."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class InvalidCellCodeError(Exception):
    """Exception raised for invalid cell codes."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

LTTRS: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
LTTRS_MAP_DICT: dict[str, int] = {c: i for i, c in enumerate(LTTRS)}
PWRS: npt.NDArray[np.int64] = np.array([len(LTTRS)**i for i in range(6)], dtype=np.int64)

AZOR_CELLS_LS: list[CoordinateInt] = sorted(AZOR_CELLS)
CABO_CELLS_LS: list[CoordinateInt] = sorted(CABO_CELLS)
CANA_CELLS_LS: list[CoordinateInt] = sorted(CANA_CELLS)
MADE_CELLS_LS: list[CoordinateInt] = sorted(MADE_CELLS)
ICEL_CELLS_LS: list[CoordinateInt] = sorted(ICEL_CELLS)

# SEE AT THE BOTTOM of script for this one as the function needs to be defined first:
# REV_CELL_PARAMS_KEY = init_rev_cell_params_key()

######## Encode and Decode IDs
def encode_id(id: int) -> str:
    idx0, idx1, idx2, idx3, idx4, idx5 = encode_numba6(id)
    return f"{LTTRS[idx0]}{LTTRS[idx1]}{LTTRS[idx2]}{LTTRS[idx3]}{LTTRS[idx4]}{LTTRS[idx5]}"
    
@njit(fastmath=True)
def encode_numba6(id: int) -> tuple[int, int, int, int, int, int]:
    idx0 = int(id // PWRS[5])
    id = id % PWRS[5]

    idx1 = int(id // PWRS[4])
    id = id % PWRS[4]

    idx2 = int(id // PWRS[3])
    id = id % PWRS[3]

    idx3 = int(id // PWRS[2])
    id = id % PWRS[2]

    idx4 = int(id // PWRS[1])
    
    idx5 = id % PWRS[1]

    return idx0, idx1, idx2, idx3, idx4, idx5

def decode_id(code: str) -> int:
    idx_arr: npt.NDArray[np.int64] = np.array([LTTRS_MAP_DICT[c] for c in code[::-1]])
    return cast(int, decode_numba(idx_arr))

@njit(fastmath=True)
def decode_numba(idx_arr: npt.NDArray[np.int64]) -> int:
    # PWRS * idx_arr results in an array, np.sum reduces it to a scalar
    return int(np.sum(PWRS * idx_arr))

######### Numba versions
@njit(fastmath=True)
def get_coord_code_base_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int, int, int]:
    a, b = get_coord_cell_base_numba(lat, lon)
    lat_dec = int((lat - int(lat)) * 10)
    lon_dec = abs(int((lon - int(lon)) * 10))
    
    return a, b, lat_dec, lon_dec

@njit(fastmath=True)
def get_coord_code_azores_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int, int, int]:
    base_lat = 36
    base_lon = -32
    a, b = get_coord_cell_azores_numba(lat, lon)
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)

    return a, b, lat_dec, lon_dec

@njit(fastmath=True)
def get_coord_code_madeira_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int, int, int]:
    base_lat = 32
    base_lon = -18
    a, b = get_coord_cell_madeira_numba(lat, lon)
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)
    
    return a, b, lat_dec, lon_dec

@njit(fastmath=True)
def get_coord_code_canary_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int, int, int]:
    base_lat = 26 
    base_lon = -19
    a, b = get_coord_cell_canary_numba(lat, lon)
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)
    
    return a, b, lat_dec, lon_dec

@njit(fastmath=True)
def get_coord_code_caboverde_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int, int, int]:
    base_lat = 14 
    base_lon = -27
    a, b = get_coord_cell_caboverde_numba(lat, lon)
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)
    
    return a, b, lat_dec, lon_dec

@njit(fastmath=True)
def get_coord_code_iceland_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int, int, int]:
    # base_lat = 62 
    base_lon = -26
    a, b = get_coord_cell_iceland_numba(lat, lon)

    lat_dec = int((lat - int(lat)) * 10)
    lon_dec = int(((lon - base_lon) / 4 * 10) % 10)
    
    return a, b, lat_dec, lon_dec

@njit(fastmath=True)
def get_coord_code_med_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int, int, int]:   
    a, b = get_coord_cell_med_numba(lat, lon)
    lat_dec = int((lat - int(lat)) * 10)
    lon_dec = abs(int((lon - int(lon)) * 10))
    
    return a, b, lat_dec, lon_dec

@njit
def get_coord_cell_azores_numba(lat: CoordVal, lon: CoordVal) -> tuple[int, int]:
    return 3, 0

@njit
def get_coord_cell_madeira_numba(lat: CoordVal, lon: CoordVal) -> tuple[int, int]:
    return 2, 0

@njit
def get_coord_cell_canary_numba(lat: CoordVal, lon: CoordVal) -> tuple[int, int]:
    return 1, 0

@njit
def get_coord_cell_caboverde_numba(lat: CoordVal, lon: CoordVal) -> tuple[int, int]:
    return 0, 0

@njit(fastmath=True)
def get_coord_cell_iceland_numba(lat: CoordVal, lon: CoordVal) -> tuple[int, int]:
    base_lon = -26
    lat_int = max(0, min(int(lat-36), 35))
    lon_int = max(0, min(int((lon-base_lon)//4) , 40))

    return lat_int, lon_int

@njit(fastmath=True)
def get_coord_cell_base_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int]:
    lat_int = max(0, min(int(lat-36), 35))
    lon_int = max(0, min(int(lon+11), len(LTTRS)-1))
    
    return lat_int, lon_int

@njit(fastmath=True)
def get_coord_cell_med_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> tuple[int, int]:
    if lat < 36:
        lat_int = max(-2, min(int(lat-36+1e-10)-1, 35))
    else:
        lat_int = max(0, min(int(lat-36), 35))
    lon_int = max(0, min(int(lon+11), len(LTTRS)-1))
    
    return lat_int, lon_int

# Regarding overload and the ellipsis (...) syntax see Gemini Chat: https://gemini.google.com/gem/a188a8e0246c/a3457cb8277e6cdd
# Standard Python doesn't support "true" overloading. In Python, the last function defined with a specific name wins.
# The @overload decorator from the typing module allows you to describe multiple ways a single function can be called.
@overload
def get_f_map_func_numba(lat: CoordVal, lon: CoordVal, mode: Literal['code'] = 'code') -> CodeFunc: ...

# Regarding overload and the ellipsis (...) syntax see Gemini Chat: https://gemini.google.com/gem/a188a8e0246c/a3457cb8277e6cdd
@overload
def get_f_map_func_numba(lat: CoordVal, lon: CoordVal, mode: Literal['cell']) -> CellFunc: ...


def get_f_map_func_numba(lat: CoordVal, lon: CoordVal, mode: Literal['code', 'cell'] = 'code') -> CodeFunc | CellFunc:
    if (lat, lon) in BASE_CELLS:
        target = get_coord_code_base_numba if mode=='code' else get_coord_cell_base_numba
    elif (lat, lon) in CANA_CELLS:
        target = get_coord_code_canary_numba if mode=='code' else get_coord_cell_canary_numba
    elif (lat, lon) in MADE_CELLS:
        target = get_coord_code_madeira_numba if mode=='code' else get_coord_cell_madeira_numba
    elif (lat, lon) in AZOR_CELLS:
        target = get_coord_code_azores_numba if mode=='code' else get_coord_cell_azores_numba
    elif (lat, lon) in CABO_CELLS:
        target = get_coord_code_caboverde_numba if mode=='code' else get_coord_cell_caboverde_numba
    elif (lat, lon) in ICEL_CELLS:
        target = get_coord_code_iceland_numba if mode=='code' else get_coord_cell_iceland_numba
    elif (lat, lon) in MEDI_CELLS:
        target = get_coord_code_med_numba if mode=='code' else get_coord_cell_med_numba
    else:
        #return None
        raise InvalidCoordinateError(
            f"Coordinates out of bounds - get_f_map_func_numba: lat: {lat:.6f}, lon: {lon:.6f})")
    
    if mode == 'code':
        return cast(CodeFunc, target)
    else:
        return cast(CellFunc, target)

'''
OLD
def get_f_map_func_numba(lat: CoordVal, lon: CoordVal, mode: Literal['code', 'cell']='code'):
    f_base = get_coord_code_base_numba if mode=='code' else get_coord_cell_base_numba
    f_azores = get_coord_code_azores_numba if mode=='code' else get_coord_cell_azores_numba
    f_madeira = get_coord_code_madeira_numba if mode=='code' else get_coord_cell_madeira_numba
    f_canary = get_coord_code_canary_numba if mode=='code' else get_coord_cell_canary_numba
    f_caboverde = get_coord_code_caboverde_numba if mode=='code' else get_coord_cell_caboverde_numba
    f_iceland = get_coord_code_iceland_numba if mode=='code' else get_coord_cell_iceland_numba
    f_med = get_coord_code_med_numba if mode=='code' else get_coord_cell_med_numba

    if (lat, lon) in MEDI_CELLS:
        return f_med
    elif (lat, lon) in CANA_CELLS:
        return f_canary
    elif (lat, lon) in ICEL_CELLS:
        return f_iceland
    elif (lat, lon) in AZOR_CELLS:
        return f_azores
    elif (lat, lon) in CABO_CELLS:
        return f_caboverde
    elif (lat, lon) in MADE_CELLS:
        return f_madeira
    elif (lat, lon) in BASE_CELLS:
        return f_base
    else:
        return None

'''

F_MAP_CELL_NUMBA: dict[CoordinateInt, CellFunc] = {
    (lat, lon): get_f_map_func_numba(lat, lon, "cell") for lat, lon in TOTAL_CELLS
}

F_MAP_CODE_NUMBA: dict[CoordinateInt, CodeFunc] = {
    (lat, lon): get_f_map_func_numba(lat, lon, "code") for lat, lon in TOTAL_CELLS
}

def get_coord_cell_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0, 
                         f_map: dict[CoordinateInt, CellFunc] = F_MAP_CELL_NUMBA) -> Cell:
    a, b = f_map[(int(lat), int(lon))](lat, lon)
    return f"{LTTRS[a]}{LTTRS[b]}"

def get_coord_code_numba(lat: CoordVal = 48.0, lon: CoordVal=12.0, 
                         f_map: dict[CoordinateInt, CodeFunc] = F_MAP_CODE_NUMBA) -> CellCode:
    a, b, c, d = f_map[(int(lat), int(lon))](lat, lon)
    return f"{LTTRS[a]}{LTTRS[b]}{LTTRS[c]}{LTTRS[d]}"


######### Out-of-main-area cell codes
def get_coord_cell_azores(lat: CoordVal, lon: CoordVal) -> Cell:
    return "30"

def get_coord_cell_madeira(lat: CoordVal, lon: CoordVal) -> Cell:
    return "20"

def get_coord_cell_canary(lat: CoordVal, lon: CoordVal) -> Cell:
    return "10"

def get_coord_cell_caboverde(lat: CoordVal, lon: CoordVal) -> Cell:
    return "00"

def get_coord_cell_iceland(lat: CoordVal, lon: CoordVal, base_lon: int = -26) -> Cell:
    lat_int = max(0, min(int(lat-36), 35))
    lon_int = max(0, min(int((lon-base_lon)//4) , 40))
    return f"{LTTRS[lat_int]}{LTTRS[lon_int]}"

def get_coord_cell_base(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> Cell:
    lat_int = max(0, min(int(lat-36), 35))
    lon_int = max(0, min(int(lon+11), len(LTTRS)-1))
    
    return f"{LTTRS[lat_int]}{LTTRS[lon_int]}"

def get_coord_cell_med(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> Cell:
    if lat < 36:
        lat_int = max(-2, min(int(lat-36+1e-10)-1, 35))
    else:
        lat_int = max(0, min(int(lat-36), 35))
    lon_int = max(0, min(int(lon+11), len(LTTRS)-1))
    
    return f"{LTTRS[lat_int]}{LTTRS[lon_int]}"

def get_coord_code_base(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> CellCode:
    lat_dec = "{:012.6f}".format(lat)[6]
    lon_dec = "{:012.6f}".format(lon)[6]
    if (int(lat), int(lon)) in BASE_CELLS:
        return f"{get_coord_cell_base(lat, lon)}{lat_dec}{lon_dec}"

    # Since we raise an error, the return type remains 'str'
    raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_cell_base({lat:.6f}, {lon:.6f})")

def get_coord_code_azores(lat: CoordVal = 48.0, lon: CoordVal=12.0, 
                          base_lat: int = 36, base_lon: int = -32) -> CellCode:
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)
    if (int(lat), int(lon)) in AZOR_CELLS:
        return f"{get_coord_cell_azores(lat, lon)}{lat_dec}{lon_dec}"

    # Since we raise an error, the return type remains 'str'
    raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_code_azores({lat:.6f}, {lon:.6f})")

def get_coord_code_madeira(lat: CoordVal = 48.0, lon: CoordVal=12.0, 
                           base_lat: int = 32, base_lon: int = -18) -> CellCode:
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)
    if (int(lat), int(lon)) in MADE_CELLS:
        return f"{get_coord_cell_madeira(lat, lon)}{lat_dec}{lon_dec}"

    # Since we raise an error, the return type remains 'str'
    raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_code_madeira({lat:.6f}, {lon:.6f})")

def get_coord_code_canary(lat: CoordVal, lon: CoordVal, 
                          base_lat: int = 26, base_lon: int = -19) -> CellCode:
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)
    if (int(lat), int(lon)) in CANA_CELLS:
        return f"{get_coord_cell_canary(lat, lon)}{lat_dec}{lon_dec}"

    # Since we raise an error, the return type remains 'str'
    raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_code_canary({lat:.6f}, {lon:.6f})")

def get_coord_code_caboverde(lat: CoordVal, lon: CoordVal, 
                             base_lat: int = 14, base_lon:int = -27) -> CellCode:
    lat_dec = int(lat - base_lat)
    lon_dec = int(lon - base_lon)
    if (int(lat), int(lon)) in CABO_CELLS:
        return f"{get_coord_cell_caboverde(lat, lon)}{lat_dec}{lon_dec}"

    # Since we raise an error, the return type remains 'str'
    raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_code_caboverde({lat:.6f}, {lon:.6f})")

def get_coord_code_iceland(lat: CoordVal, lon: CoordVal, base_lat = 62, base_lon=-26) -> CellCode:
    lat_int = max(0, min(int(lat-36), 35))
    lon_int = max(0, min(int((lon-base_lon)//4), 40))

    lat_dec = "{:012.6f}".format(lat)[6]
    lon_dec = f"{((lon-base_lon)%4)/4:.8f}"[-8]
    
    if (int(lat), int(lon)) in ICEL_CELLS:
        return f"{LTTRS[lat_int]}{LTTRS[lon_int]}{lat_dec}{lon_dec}"

    # Since we raise an error, the return type remains 'str'
    raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_code_iceland({lat:.6f}, {lon:.6f})")    

def get_coord_code_med(lat: CoordVal = 48.0, lon: CoordVal=12.0) -> CellCode:   
    lat_dec = "{:012.6f}".format(lat)[6]
    lon_dec = "{:012.6f}".format(lon)[6]
    if (int(lat), int(lon)) in MEDI_CELLS:
        return f"{get_coord_cell_med(lat, lon)}{lat_dec}{lon_dec}"

    # Since we raise an error, the return type remains 'str'
    raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_cell_med({lat:.6f}, {lon:.6f})")    

def get_rev_coord_cell_base(cell: Cell | CellCode) -> CoordinateInt:
    lat_idx = LTTRS.index(cell[0]) - len(LTTRS) if cell[0] in 'yz' else LTTRS.index(cell[0])
    lon_idx = LTTRS.index(cell[1])
    return lat_idx + 36, lon_idx - 11

def get_rev_coord_code_base(cell_code: CellCode) -> Coordinate:
    lat, lon = get_rev_coord_cell_base(cell_code[:2])
    
    lat += int(cell_code[2]) * 0.1
    lon += int(cell_code[3]) * 0.1 * (1 if lon >= 0 else -1) + (0 if lon >= 0 else 1)
    
    return lat, lon
    
def get_rev_coord_code_accm(code: CellCode, base_lat: int, base_lon: int) -> Coordinate:
    add_lat = int(code[2])
    add_lon = int(code[3])
    return base_lat + add_lat, base_lon + add_lon

def get_rev_coord_code_azores(code: CellCode) -> Coordinate:
    base_lat, base_lon = AZOR_CELLS_LS[0]
    lat, lon = get_rev_coord_code_accm(code, base_lat, base_lon)
    if (lat, lon) in AZOR_CELLS:
        return lat, lon
    else:
        raise InvalidCellCodeError(f"Invalid cell code AZORES: {code}")

def get_rev_coord_code_madeira(code: CellCode) -> Coordinate:
    base_lat, base_lon = MADE_CELLS_LS[0]
    lat, lon = get_rev_coord_code_accm(code, base_lat, base_lon)
    if (lat, lon) in MADE_CELLS:
        return lat, lon
    else:
        raise InvalidCellCodeError(f"Invalid cell code MADEIRA: {code}")

def get_rev_coord_code_canary(code: CellCode) -> Coordinate:
    base_lat, base_lon = CANA_CELLS_LS[0]
    lat, lon = get_rev_coord_code_accm(code, base_lat, base_lon)
    if (lat, lon) in CANA_CELLS:
        return lat, lon
    else:
        raise InvalidCellCodeError(f"Invalid cell code CANARY: {code}")

def get_rev_coord_code_caboverde(code: CellCode) -> Coordinate:
    base_lat, base_lon = CABO_CELLS_LS[0]
    lat, lon = get_rev_coord_code_accm(code, base_lat, base_lon)
    if (lat, lon) in CABO_CELLS:
        return lat, lon 
    else:
        raise InvalidCellCodeError(f"Invalid cell code CABOVERDE: {code}")

def get_rev_coord_code_iceland(code: CellCode) -> Coordinate:
    lat_idx = LTTRS.index(code[0])
    base_lat = lat_idx + 36
    base_lon = ICEL_CELLS_LS[0][1]
    if (base_lat, base_lon) in ICEL_CELLS:
        add_lat = int(code[2]) / 10
        add_lon = (int(code[1]) * 4) + (int(code[3]))/10*4
        return base_lat + add_lat, base_lon + add_lon
    else:
        raise InvalidCellCodeError(f"Invalid cell code ICELAND: {code}")

def get_f_map_func(lat: CoordVal, lon: CoordVal, mode: Literal['code', 'cell'] = 'code') -> MapFunc:
    is_code = mode=='code'

    if (lat, lon) in BASE_CELLS:
        return cast(MapFunc, get_coord_code_base if is_code else get_coord_cell_base)
    elif (lat, lon) in CANA_CELLS:
        return cast(MapFunc, get_coord_code_canary if is_code else get_coord_cell_canary)
    elif (lat, lon) in MADE_CELLS:
        return cast(MapFunc, get_coord_code_madeira if is_code else get_coord_cell_madeira)
    elif (lat, lon) in AZOR_CELLS:
        return cast(MapFunc, get_coord_code_azores if is_code else get_coord_cell_azores)
    elif (lat, lon) in CABO_CELLS:
        return cast(MapFunc, get_coord_code_caboverde if is_code else get_coord_cell_caboverde)
    elif (lat, lon) in ICEL_CELLS:
        return cast(MapFunc, get_coord_code_iceland if is_code else get_coord_cell_iceland)
    elif (lat, lon) in MEDI_CELLS:
        return cast(MapFunc, get_coord_code_med if is_code else get_coord_cell_med)
    else:
        #return None
        raise InvalidCoordinateError(f"Coordinates out of bounds - get_f_map_func: lat: {lat:.6f}, lon: {lon:.6f})") 




def get_rev_map_cell(f_map_cell: dict[CoordinateInt, MapFunc]) -> dict[Cell, list[CoordinateInt]]:
    rev_map_cell: dict[Cell, list[CoordinateInt]] = {}
    for (lat, lon) in sorted(f_map_cell):
        #lat += 1e-7
        #lon += 1e-7 #if lon >= 0 else -1e-7
        cell: Cell = f_map_cell[(int(lat), int(lon))](lat, lon)
        if cell in rev_map_cell:
            rev_map_cell[cell].append((int(lat), int(lon)))
        else:
            rev_map_cell[cell] = [(int(lat), int(lon))]
    
    return rev_map_cell

RevCodeFunc: TypeAlias = Callable[[CellCode], tuple[float, float]]

def get_rev_map_code(f_map_code: dict[CoordinateInt, MapFunc], 
                     icel_codes: set[Cell], 
                     print_details: bool = True) -> dict[Cell, tuple[RevCodeFunc, float, float]]:
    rev_map_code: dict[Cell, tuple[RevCodeFunc, float, float]] = {}

    for (lat, lon) in sorted(f_map_code):
        try:
            f = f_map_code[(lat, lon)]
            code = f(lat, lon)
            cell = cast(Cell, code[:2])
        
            if cell in icel_codes: 
                if f == get_coord_code_iceland:
                    rev_map_code[cell] = (get_rev_coord_code_iceland, 0.1, 0.4)
            elif code[:2] == '10': 
                if f == get_coord_code_canary:
                    rev_map_code[cell] = (get_rev_coord_code_canary, 1.0, 1.0)
            elif code[:2] == '00':
                if f == get_coord_code_caboverde:
                    rev_map_code[cell] = (get_rev_coord_code_caboverde, 1.0, 1.0)
            elif code[:2] == '30':
                if f == get_coord_code_azores:
                    rev_map_code[cell] = (get_rev_coord_code_azores, 1.0, 1.0)
            elif code[:2] == '20':
                if f == get_coord_code_madeira:
                    rev_map_code[cell] = (get_rev_coord_code_madeira, 1.0, 1.0)
            else:
                rev_map_code[cell] = (cast(RevCodeFunc, get_rev_coord_code_base), 0.1, 0.1)
        except Exception as e:
            if print_details:
                print(f"Error processing {lat}, {lon}: {e}")
            continue

    return rev_map_code


F_MAP_CELL: dict[CoordinateInt, MapFunc] = {(lat, lon): get_f_map_func(lat, lon, "cell") for lat, lon in TOTAL_CELLS}
F_MAP_CODE: dict[CoordinateInt, MapFunc] = {(lat, lon): get_f_map_func(lat, lon, "code") for lat, lon in TOTAL_CELLS}
ICEL_CODES: set[Cell] = set([F_MAP_CELL[(lat, lon)](lat, lon) for lat, lon in ICEL_CELLS])
REV_MAP_CELL: dict[Cell, list[CoordinateInt]] = get_rev_map_cell(F_MAP_CELL)
REV_MAP_CODE: dict[Cell, tuple[Callable, float, float]] = get_rev_map_code(F_MAP_CODE, ICEL_CODES)

#####  Cell Codes für Koordinaten
def get_coord_cell(lat: CoordVal = 48.0, lon: CoordVal = 12.0, 
                   f_map: dict[CoordinateInt, MapFunc]=F_MAP_CELL) -> Cell:
    #f_base = get_coord_cell_base
    #return f_base(lat, lon)
    return f_map[(int(lat), int(lon))](lat, lon)
    #return f_map[(int(lat), int(lon if lon >= 0 else lon-1))](lat, lon)

def get_coord_code(lat: CoordVal = 48.0, lon: CoordVal = 12.0, 
                   f_map: dict[CoordinateInt, MapFunc]=F_MAP_CODE) -> CellCode:
    if (int(lat), int(lon)) in f_map:
        return f_map[(int(lat), int(lon))](lat, lon)
    else:
        raise InvalidCoordinateError(f"Coordinates out of bounds - get_coord_code({lat:.6f}, {lon:.6f})")

def get_coord_code_lng(lat: CoordVal = 48.0, lon: CoordVal = 12.0) -> CellCode:
    lat_cent = "{:012.6f}".format(lat)[7:9]
    lon_cent = "{:012.6f}".format(lon)[7:9]
    
    return get_coord_code(lat, lon) + "-" + lat_cent + lon_cent

def get_coord_code_lng2(lat: CoordVal = 48.0, lon: CoordVal = 12.0) -> CellCode:
    if lat < 0:
        lat_cent = "{:010.6f}".format(lat).replace(".", "")[:5]
    else:
        lat_cent = "{:09.6f}".format(lat).replace(".", "")[:4]
        
    if lon < 0:
        lon_cent = "{:010.6f}".format(lon).replace(".", "")[:5]
    else:
        lon_cent = "{:09.6f}".format(lon).replace(".", "")[:4]
    
    return lat_cent + " " + lon_cent

def get_rev_coord_cell(cell: Cell, rev_map_cell: dict[Cell, list[CoordinateInt]]=REV_MAP_CELL) -> list[CoordinateInt]:
    try:
        return rev_map_cell[cell]
    except KeyError:
        raise InvalidCellCodeError(f"Invalid cell code: {cell}")

def get_rev_coord_code(cell_code: CellCode, 
                       rev_map_code: dict[Cell, tuple[RevCodeFunc, float, float]] = REV_MAP_CODE) -> Coordinate:
    cell: Cell = cell_code[:2]
    return rev_map_code[cell][0](cell_code)

def get_subcell_dim(cell_code: CellCode, 
                    rev_map_code: dict[Cell, tuple[RevCodeFunc, float, float]] = REV_MAP_CODE) -> tuple[float, float]:
    cell: Cell = cell_code[:2]
    return rev_map_code[cell][1:]
    
def get_rev_coord_cell_OLD(cell):
    letters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lat_idx = letters.index(cell[0])
    lon_idx = letters.index(cell[1])
    
    return lat_idx + 36, lon_idx - 11

def get_rev_coord_code_OLD(cell_code: CellCode) -> Coordinate:
    lat, lon = get_rev_coord_cell(cell_code[:2])[0]
    
    lat += int(cast(int, cell_code[2])) * 0.1
    lon += int(cast(int, cell_code[3])) * 0.1 * (1 if lon >= 0 else -1) + (0 if lon >= 0 else 1)
    
    return lat, lon

def get_rev_coord_code_OLD_WRONG(cell_code: CellCode) -> Coordinate:
    lat, lon = get_rev_coord_cell(cell_code[:2])[0]
    
    lat += int(cell_code[2]) * 0.1
    lon += int(cell_code[3]) * 0.1
    
    return lat, lon

def get_rev_coord_code_lng(cell_code_lng: CellCode) -> Coordinate:
    lat, lon = cast(tuple[float, float], get_rev_coord_code(cast(str, cell_code_lng[:4])))
    
    lat += round(float(cell_code_lng[5:7]) * 0.001, 6)
    lon += round(float(cell_code_lng[7:]) * 0.001, 6)
    
    return lat, lon

def get_cell_subcell(lat: CoordVal, lon: CoordVal) -> tuple[Cell, Subcell]:
    code: CellCode = get_coord_code(lat, lon)
    subcell: Subcell = code[-2:]
    cell: Cell = code[:2]
    return cell, subcell



def init_rev_cell_params_key() -> dict[Cell, ParamsKey]:
    REV_CELL_PARAMS_KEY = {}
    for lat, lon in TOTAL_CELLS:
        cell = get_coord_cell(lat, lon)
        if (lat, lon) in BASE_CELLS:
            params_key = 'base'
        elif (lat, lon) in ICEL_CELLS:
            params_key = 'iceland'
        elif (lat, lon) in AZOR_CELLS:
            params_key = 'azores'
        elif (lat, lon) in CABO_CELLS:
            params_key = 'caboverde'
        elif (lat, lon) in CANA_CELLS:
            params_key = 'canary'
        elif (lat, lon) in MADE_CELLS:
            params_key = 'madeira'
        elif (lat, lon) in MEDI_CELLS:
            params_key = 'base'

        REV_CELL_PARAMS_KEY[cell] = params_key
        if params_key not in LATLON_CELL_PARAMS:
            print("ERROR REV_CELL_PARAMS_KEY", lat, lon, cell, params_key)

    return REV_CELL_PARAMS_KEY

REV_CELL_PARAMS_KEY = init_rev_cell_params_key()
