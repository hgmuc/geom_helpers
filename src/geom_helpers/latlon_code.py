from __future__ import annotations
from numba import njit
from math import cos, radians
from itertools import product
from typing import Callable, TypeAlias, Literal, overload

from basic_helpers.types_base import CoordVal, FlexNumeric, Coordinate, CoordinateInt, BBox
from basic_helpers.config_reg_code import CellParams, RegCode, Cell

from geom_helpers.osm_reader_helper import LATLON_CELL_PARAMS, REV_MAP_CELL, BASE_CELLS, AZOR_CELLS, CABO_CELLS, CANA_CELLS, ICEL_CELLS, MADE_CELLS, MEDI_CELLS, TOTAL_CELLS
from geom_helpers.osm_reader_helper import get_coord_cell, get_rev_coord_cell, get_coord_cell_numba, REV_CELL_PARAMS_KEY

MapFunc: TypeAlias = Callable[[CoordVal, CoordVal], str]
CodeFunc: TypeAlias = Callable[[CoordVal, CoordVal, int], str]
RevCodeFunc: TypeAlias = Callable[[str],  Coordinate]
#CodeFunc: TypeAlias = Callable[[CoordVal, CoordVal], tuple[int, int, float, float]]
#CellFunc: TypeAlias = Callable[[CoordVal, CoordVal], tuple[int, int]]


LTTRS: str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
LTTRS_MAP_DICT: dict[str, int] = {c: i for i, c in enumerate(LTTRS)}

if 'extent_lon' not in LATLON_CELL_PARAMS['base']:
    LATLON_CELL_PARAMS['base']['extent_lon'] = 1
    LATLON_CELL_PARAMS['base']['extent_lat'] = 1
    LATLON_CELL_PARAMS['azores']['extent_lon'] = 8
    LATLON_CELL_PARAMS['azores']['extent_lat'] = 4
    LATLON_CELL_PARAMS['madeira']['extent_lon'] = 2
    LATLON_CELL_PARAMS['madeira']['extent_lat'] = 2
    LATLON_CELL_PARAMS['canary']['extent_lon'] = 7
    LATLON_CELL_PARAMS['canary']['extent_lat'] = 5
    LATLON_CELL_PARAMS['caboverde']['extent_lon'] = 6
    LATLON_CELL_PARAMS['caboverde']['extent_lat'] = 5
    LATLON_CELL_PARAMS['iceland']['extent_lon'] = 4
    LATLON_CELL_PARAMS['iceland']['extent_lat'] = 1

class InvalidCoordinateError(Exception):
    """Exception raised for invalid coordinates."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class InvalidRegCodeError(Exception):
    """Exception raised for invalid cell codes."""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


@njit
def get_len_m_lttrs_by_lat(lat: int) -> int:
    return 38 if lat > 45 else min(38 + 46 - int(lat), 45)

@njit(fastmath=True)
def get_base_latlon(lat: CoordVal, lon: CoordVal, 
                    base_lat: int | None, base_lon: int | None) -> tuple[int, int, CoordVal]:
    base_lat = int(lat) if base_lat is None else base_lat
    if base_lon is None:
        lon += 11
        base_lon = int(lon)  
    #else:
    #    base_lon = int(lon)
    return base_lat, base_lon, lon

@njit(fastmath=True)
def get_m_ref(lat: CoordVal, px, M: int=111320) -> float:
    return float(cos(radians(lat)) * M / px)

@njit(fastmath=True)
def get_idx_vals(lat: CoordVal, lon: CoordVal, 
                 base_lat: int, base_lon: int, 
                 param_x1: FlexNumeric, param_x2: FlexNumeric, 
                 param_x3: FlexNumeric, param_y1: FlexNumeric, 
                 param_y2: FlexNumeric, param_y3: FlexNumeric) -> tuple[int, int, int, int, int, int, 
                                                                        CoordVal, CoordVal]:
    lat_dec = lat - base_lat
    if lat_dec == 0:
        lat_dec += 1e-7
        lat += 1e-7
    #print("lat_dec", lat_dec)
    lon_dec = abs(lon - base_lon)
    if lon_dec == 0:
        lon_dec += 1e-7
        lon += 1e-7
    #print("lon_dec", lon_dec)

    y1 = int(lat_dec * param_y1)
    lat_dec = lat_dec * param_y1 - y1
    y2 = int(lat_dec * param_y2)
    lat_dec = lat_dec * param_y2 - y2
    y3 = int(lat_dec * param_y3)
    lat_dec = lat_dec * param_y3 - y3
    
    x1 = int(lon_dec * param_x1)
    lon_dec = lon_dec * param_x1 - x1
    x2 = int(lon_dec * param_x2)
    lon_dec = lon_dec * param_x2 - x2
    x3 = int(lon_dec * param_x3)
    lon_dec = lon_dec * param_x3 - x3

    return x1, x2, x3, y1, y2, y3, lat_dec, lon_dec

@njit(fastmath=True)
def get_idx12(x1: int, x2: int, 
              y1: int, y2: int, 
              n_lttr1: int, n_lttr2: int, 
              off1: int, off2: int) -> tuple[int, int]:
    idx1 = y1 * n_lttr1 + x1 + off1
    idx2 = y2 * n_lttr2 + x2 + off2
    
    return idx1, idx2

@njit(fastmath=True)
def get_prod_params(p1: FlexNumeric, p2: FlexNumeric, p3: FlexNumeric) -> FlexNumeric:
    return p1*p2*p3

@njit(fastmath=True)
def get_subcode_idxs(lat_dec, lon_dec, len_lttrs, py, M_REF, M):
    lat_res_M = round(lat_dec * M / py)
    lon_res_M = round(lon_dec * M_REF)
    
    return lat_res_M // len_lttrs, lat_res_M % len_lttrs, lon_res_M // len_lttrs, lon_res_M % len_lttrs

def get_reg_cell_code_by_params(lat: CoordVal, lon: CoordVal, params: CellParams, M: int = 111320) -> RegCode:
    cell: Cell = get_coord_cell_numba(lat, lon)
    base_lat, base_lon, lon = get_base_latlon(lat, lon, params.get('base_lat'), params.get('base_lon'))
    
    px: FlexNumeric = get_prod_params(params['x1'], params['x2'], params['x3'])
    py: FlexNumeric = get_prod_params(params['y1'], params['y2'], params['y3'])
    
    M_REF: FlexNumeric = get_m_ref(lat, px, M)
    
    #assert 'len_lttrs' in params, "len_lttrs is not defined"
    len_lttrs = params.get('len_lttrs')
    #m_lttrs = LTTRS[:len_lttrs]

    x1, x2, x3, y1, y2, y3, lat_dec, lon_dec = get_idx_vals(lat, lon, base_lat, base_lon, 
                                                            params['x1'], params['x2'], params['x3'], 
                                                            params['y1'], params['y2'], params['y3'])

    idx1, idx2 = get_idx12(x1, x2, y1, y2, params['n_lttr1'], 
                           params['n_lttr2'], params['off1'], params['off2'])

    sub_idx1, sub_idx2, sub_idx3, sub_idx4 = get_subcode_idxs(lat_dec, lon_dec, len_lttrs, py, M_REF, M)

    code = f"{cell}{LTTRS[idx1]}{LTTRS[idx2]}-{y3+1}{x3+1}{LTTRS[sub_idx1]}{LTTRS[sub_idx2]}{LTTRS[sub_idx3]}{LTTRS[sub_idx4]}"

    return code


def get_reg_cell_code_base(lat: CoordVal, lon: CoordVal, M: int = 111320) -> RegCode:
    #   57	58	59	67	68	69
    #   54	55	56	64	65	66
    #   51	52	53	61	62	63
    #   37	38	39	47	48	49
    #   34	35	36	44	45	46
    #   31	32	33	41	42	43
    #   17	18	19	27	28	29
    #   14	15	16	24	25	26
    #   11	12	13	21	22	23

    len_lttrs = get_len_m_lttrs_by_lat(int(lat))
    params = LATLON_CELL_PARAMS['base']
    params['len_lttrs'] = len_lttrs
    
    return get_reg_cell_code_by_params(lat, lon, params, M)

def get_reg_cell_base_coords(ls: set[CoordinateInt] | list[CoordinateInt]) -> CoordinateInt:
    base_lat, base_lon = sorted(ls)[0]
    return base_lat, base_lon

def get_reg_cell_code_madeira(lat: CoordVal, lon: CoordVal, M: int = 111320) -> RegCode:
    # Madeira: 32, -18, 34, -16 (32.0000000, -17.2653876, 33.1071999, -16.3216117) -  2 x 2 => 4 x 9
    ### DEPRECATED (Madeira: 32.6496497, -17.2653876, 33.1071999, -16.3216117  -  2 x 2 => 6 x 6=
    ### DEPRECATED 30  MADEIRA       96804  48400  32135  -> unbewohnte Insel mit Overlap zu Canary Islands
    ### DEPRECATED 31  MADEIRA       95600  47800  32135  -> unbewohnte Insel mit Overlap zu Canary Islands
    # 32  MADEIRA       94405  47200  32135
    # 33  MADEIRA       93361  46700  31120

    #   -18    -17    -16
    #     U V W | X Y Z
    #     O P Q | R S T
    # 33  I J K | L M N
    #     ------|------
    #     C D E | F G H
    #     6 7 8 | 9 A B
    # 32  0 1 2 | 3 4 5


    base_lat, base_lon = get_reg_cell_base_coords(MADE_CELLS) # 32, -18
    len_lttrs = get_len_m_lttrs_by_lat(base_lat)
    params = LATLON_CELL_PARAMS['madeira']
    params['base_lat'] = base_lat
    params['base_lon'] = base_lon
    params['len_lttrs'] = len_lttrs
    
    #params = {'base_lat': base_lat, 'base_lon': base_lon, 'len_lttrs': len_lttrs, 
    #          'x1': 3, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
    #          'n_lttr1': 6, 'n_lttr2': 3, 'off1': 0, 'off2': 0}
    
    return get_reg_cell_code_by_params(lat, lon, params, M)

def get_reg_cell_code_azores(lat: CoordVal, lon: CoordVal, M: int = 111320) -> RegCode:
    # Azoren:  36.9294528, -31.2623230, 39.7239193, -25.0151975  -  4 x 8 => 6 x 8
    # 36  AZORES        90060   45030   30020
    # 37  AZORES        88904   44452   29968
    # 38  AZORES        87721   43860   29240
    # 39  AZORES        86512   43256   28837

    #      e  |  f  |  g  |  h  |  i  |  j  |  k  |  l
    #      W  |  X  |  Y  |  Z  |  a  |  b  |  c  |  d
    #      O  |  P  |  Q  |  R  |  S  |  T  |  U  |  V
    #      G  |  H  |  I  |  J  |  K  |  L  |  M  |  N
    #      8  |  9  |  A  |  B  |  C  |  D  |  E  |  F
    #      0  |  1  |  2  |  3  |  4  |  5  |  6  |  7

    base_lat, base_lon = get_reg_cell_base_coords(AZOR_CELLS) # 36, -32
    len_lttrs = get_len_m_lttrs_by_lat(base_lat)
    params = LATLON_CELL_PARAMS['azores']
    params['base_lat'] = base_lat
    params['base_lon'] = base_lon
    params['len_lttrs'] = len_lttrs
    
    #params = {'base_lat': base_lat, 'base_lon': base_lon, 'len_lttrs': len_lttrs, 
    #          'x1': 1, 'x2': 6, 'x3': 9, 'y1': 1.5, 'y2': 6, 'y3': 9, 
    #          'n_lttr1': 8, 'n_lttr2': 6, 'off1': 0, 'off2': 0}
    
    return get_reg_cell_code_by_params(lat, lon, params, M)

def get_reg_cell_code_canary(lat: CoordVal, lon: CoordVal, M: int = 111320) -> RegCode:
    # Canary Islands: 26.36117, -18.92352, 30.25648, -12.47875   -  5 x 7 => 5 x 7
    # 26  CANARY       100054   50027   33351
    # 27  CANARY        99187   49593   33062
    # 28  CANARY        98290   49145   32763
    # 29  CANARY        97363   48681   32454
    # 30  CANARY        96406   48203   32135

    #       -18   -17   -16   -15   -14   -13   -12
    #  30    S  |  T  |  U  |  V  |  W  |  X  |  Y
    #  29    L  |  M  |  N  |  O  |  P  |  Q  |  R  
    #  28    E  |  F  |  G  |  H  |  I  |  J  |  K
    #  27    7  |  8  |  9  |  A  |  B  |  C  |  D  
    #  26    0  |  1  |  2  |  3  |  4  |  5  |  6

    base_lat, base_lon = get_reg_cell_base_coords(CANA_CELLS) # 26, -19
    len_lttrs = get_len_m_lttrs_by_lat(base_lat)
    params = LATLON_CELL_PARAMS['canary']
    params['base_lat'] = base_lat
    params['base_lon'] = base_lon
    params['len_lttrs'] = len_lttrs
    
    #params = {'base_lat': base_lat, 'base_lon': base_lon, 'len_lttrs': len_lttrs, 
    #          'x1': 1, 'x2': 7, 'x3': 9, 'y1': 1, 'y2': 8, 'y3': 9, 
    #          'n_lttr1': 7, 'n_lttr2': 7, 'off1': 0, 'off2': 0}
    
    return get_reg_cell_code_by_params(lat, lon, params, M)

def get_reg_cell_code_caboverde(lat: CoordVal, lon: CoordVal, M: int = 111320) -> RegCode:
    # Cape Verde: 14.00485, -26.32379, 18.005, -21.39096         -  5 x 6 => 7 x 8
    # 14  CABO         108013    54006    36004   
    # 15  CABO         107527    53763    35842   
    # 16  CABO         107008    53504    35669   
    # 17  CABO         106456    53228    35485   
    # 18  CABO         105872    52936    35290   

    #      m  |  n  |  o  |  p  |  q  |  r  |  s  |  t
    #      e  |  f  |  g  |  h  |  i  |  j  |  k  |  l
    #      W  |  X  |  Y  |  Z  |  a  |  b  |  c  |  d
    #      O  |  P  |  Q  |  R  |  S  |  T  |  U  |  V
    #      G  |  H  |  I  |  J  |  K  |  L  |  M  |  N
    #      8  |  9  |  A  |  B  |  C  |  D  |  E  |  F
    #      0  |  1  |  2  |  3  |  4  |  5  |  6  |  7

    base_lat, base_lon = get_reg_cell_base_coords(CABO_CELLS) # 14, -27
    len_lttrs = get_len_m_lttrs_by_lat(base_lat)
    params = LATLON_CELL_PARAMS['caboverde']
    params['base_lat'] = base_lat
    params['base_lon'] = base_lon
    params['len_lttrs'] = len_lttrs
    
    #params = {'base_lat': base_lat, 'base_lon': base_lon, 'len_lttrs': len_lttrs, 
    #          'x1': 8/6, 'x2': 6, 'x3': 9, 'y1': 7/5, 'y2': 6, 'y3': 9, 
    #          'n_lttr1': 8, 'n_lttr2': 6, 'off1': 0, 'off2': 0}
    
    return get_reg_cell_code_by_params(lat, lon, params, M)

def get_reg_cell_code_iceland(lat: CoordVal, lon: CoordVal, M: int = 111320) -> RegCode:
    # Island:  62.84553, -25.7, 67.50085, -12.41708              -  6 x 14
    # 62  ICEL          52262   26131    17420
    # 63  ICEL          50538   25269    16846
    # 64  ICEL          48799   24399    16266
    # 65  ICEL          47046   23523    15682
    # 66  ICEL          45278   22639    15092
    # 67  ICEL          43496   21748    14498 

    #        -25  -24  -23  -22  -21  -20  -19  -18  -17  -16  -15  -14  -13  -12  
    # 67  V   0    0    0    0    1    1    1    1    2    2    2    2    3    3   
    # 66  U   0    0    0    0    1    1    1    1    2    2    2    2    3    3   
    # 65  T   0    0    0    0    1    1    1    1    2    2    2    2    3    3   
    # 64  S   0    0    0    0    1    1    1    1    2    2    2    2    3    3   
    # 63  R   0    0    0    0    1    1    1    1    2    2    2    2    3    3   
    # 62  Q   0    0    0    0    1    1    1    1    2    2    2    2    3    3   

    #         C  D  E  F  G  H
    #         6  7  8  9  A  B
    # 62  Q   0  1  2  3  4  5

    cell: Cell = get_coord_cell_numba(lat, lon)
    base_lat, base_lon = int(lat), -26 + int(cell[1]) * 4
    len_lttrs = get_len_m_lttrs_by_lat(base_lat)
    params = LATLON_CELL_PARAMS['iceland']
    params['base_lat'] = base_lat
    params['base_lon'] = base_lon
    params['len_lttrs'] = len_lttrs
    
    #params = {'base_lat': base_lat, 'base_lon': base_lon, 'len_lttrs': len_lttrs, 
    #          'x1': 1.5, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
    #          'n_lttr1': 6, 'n_lttr2': 3, 'off1': 0, 'off2': 0}
    
    return get_reg_cell_code_by_params(lat, lon, params, M)

@overload
def get_f_map_func(lat: CoordVal, lon: CoordVal, mode: Literal[1]=1) -> CodeFunc: ...   # type: ignore

@overload
def get_f_map_func(lat: CoordVal, lon: CoordVal, mode: Literal[2]=2) -> RevCodeFunc: ...

def get_f_map_func(lat: CoordVal, lon: CoordVal, mode: Literal[1, 2]=1) -> CodeFunc | RevCodeFunc:
    is_get_code = mode == 1

    if (lat, lon) in BASE_CELLS:
        return get_reg_cell_code_base if is_get_code else get_rev_reg_cell_code_base
    elif (lat, lon) in CANA_CELLS:
        return get_reg_cell_code_canary if is_get_code else get_rev_reg_cell_code_canary
    elif (lat, lon) in MADE_CELLS:
        return get_reg_cell_code_madeira if is_get_code else get_rev_reg_cell_code_madeira
    elif (lat, lon) in AZOR_CELLS:
        return get_reg_cell_code_azores if is_get_code else get_rev_reg_cell_code_azores
    elif (lat, lon) in CABO_CELLS:
        return get_reg_cell_code_caboverde if is_get_code else get_rev_reg_cell_code_caboverde
    elif (lat, lon) in ICEL_CELLS:
        return get_reg_cell_code_iceland if is_get_code else get_rev_reg_cell_code_iceland
    elif (lat, lon) in MEDI_CELLS:
        return get_reg_cell_code_base if is_get_code else get_rev_reg_cell_code_base
    else:
        raise InvalidCoordinateError(f"Coordinates out of bounds - get_f_map_func: lat: {lat:.6f}, lon: {lon:.6f})") 

def get_rev_reg_cell_code_base(code: RegCode, M: int = 111320) -> Coordinate:
    #params = {'x1': 2, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
    #          'n_lttr1': 2, 'n_lttr2': 3, 'off1': 1, 'off2': 1}
    params = LATLON_CELL_PARAMS['base']
    return get_rev_reg_cell_code_by_params(code, params, M)

def get_rev_reg_cell_code_azores(code: RegCode, M: int = 111320) -> Coordinate:
    #         'x1': 9/7, 'x2': 6, 'x3': 9, 'y1': 1.5, 'y2': 6, 'y3': 9, 
    #         'n_lttr1': 9, 'n_lttr2': 6, 'off1': 0, 'off2': 0}
    #params = {'x1': 1, 'x2': 6, 'x3': 9, 'y1': 1.5, 'y2': 6, 'y3': 9, 
    #          'n_lttr1': 8, 'n_lttr2': 6, 'off1': 0, 'off2': 0}
    params = LATLON_CELL_PARAMS['azores']
    return get_rev_reg_cell_code_by_params(code, params, M)

def get_rev_reg_cell_code_madeira(code: RegCode, M: int = 111320) -> Coordinate:
    #          'x1': 3, 'x2': 3, 'x3': 9, 'y1': 2, 'y2': 3, 'y3': 9, 
    #          'n_lttr1': 6, 'n_lttr2': 3, 'off1': 0, 'off2': 0}
    #params = {'x1': 3, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
    #          'n_lttr1': 6, 'n_lttr2': 3, 'off1': 0, 'off2': 0}
    params = LATLON_CELL_PARAMS['madeira']
    return get_rev_reg_cell_code_by_params(code, params, M)

def get_rev_reg_cell_code_canary(code: RegCode, M: int = 111320) -> Coordinate:
    #params = {'x1': 1, 'x2': 7, 'x3': 9, 'y1': 1, 'y2': 8, 'y3': 9, 
    #          'n_lttr1': 7, 'n_lttr2': 7, 'off1': 0, 'off2': 0}
    params = LATLON_CELL_PARAMS['canary']
    return get_rev_reg_cell_code_by_params(code, params, M)

def get_rev_reg_cell_code_caboverde(code: RegCode, M: int = 111320) -> Coordinate:
    #         'x1': 8/6, 'x2': 6, 'x3': 9, 'y1': 7/5, 'y2': 6, 'y3': 9, 
    #         'n_lttr1': 8, 'n_lttr2': 6, 'off1': 0, 'off2': 0}
    #params = {'x1': 8/6, 'x2': 6, 'x3': 9, 'y1': 7/5, 'y2': 6, 'y3': 9, 
    #          'n_lttr1': 8, 'n_lttr2': 6, 'off1': 0, 'off2': 0}
    params = LATLON_CELL_PARAMS['caboverde']
    return get_rev_reg_cell_code_by_params(code, params, M)

def get_rev_reg_cell_code_iceland(code: RegCode, M: int = 111320) -> Coordinate:
    #params = {'x1': 1.5, 'x2': 3, 'x3': 9, 'y1': 3, 'y2': 3, 'y3': 9, 
    #          'n_lttr1': 6, 'n_lttr2': 3, 'off1': 0, 'off2': 0}
    params = LATLON_CELL_PARAMS['iceland']
    return get_rev_reg_cell_code_by_params(code, params, M)

@njit
def get_len_m_lttrs_from_code(code0: str, use_max_len: bool, code0_idx: int) -> int:
    len_lttrs = 38
    #if code0 in '0123yz':
    if use_max_len:
        len_lttrs = 45
    else:
    #elif code0 in LTTRS[:10]:
        len_lttrs += code0_idx + 1 #LTTRS[:10][::-1].index(code0) + 1

    return len_lttrs

@njit(fastmath=True)
def get_base_latlon_from_code(code0: str, c0: int, c1: int) -> CoordinateInt:
    lat_int = c0 + (36 if code0 not in 'yz' else -26)
    lon_int = c1 - 11
    base_lat, base_lon = lat_int, lon_int

    return base_lat, base_lon

@njit(fastmath=True)
def get_rev_code_idx_vals(c2: int, c3: int, c5: int, c6: int, 
                          param_x1: FlexNumeric, param_x2: FlexNumeric, param_x3: FlexNumeric, 
                          param_y1: FlexNumeric, param_y2: FlexNumeric, param_y3: FlexNumeric, 
                          n_lttr1: int, n_lttr2: int, off1: int, off2: int) -> tuple[int, int, FlexNumeric, 
                                                                                     int, int, FlexNumeric, 
                                                                                     FlexNumeric, FlexNumeric, 
                                                                                     FlexNumeric, FlexNumeric, 
                                                                                     FlexNumeric, FlexNumeric]:

    x1 = (c2 - off1) % n_lttr1
    x2 = (c3 - off2) % n_lttr2
    x3 = (c6 - 1) % param_x3
    frac_x1 = 1 / param_x1
    frac_x2 = 1 / (param_x1 * param_x2)
    frac_x3 = 1 / (param_x1 * param_x2 * param_x3)

    y1 = (c2 - off1) // n_lttr1
    y2 = (c3 - off2) // n_lttr2
    y3 = (c5 - 1) % param_y3
    frac_y1 = 1 / param_y1
    frac_y2 = 1 / (param_y1 * param_y2)
    frac_y3 = 1 / (param_y1 * param_y2 * param_y3)

    return x1, x2, x3, y1, y2, y3, frac_x1, frac_x2, frac_x3, frac_y1, frac_y2, frac_y3

@njit(fastmath=True)
def get_rev_latlon(base_lat: int, base_lon: int, 
                   x1: int, x2: int, x3: FlexNumeric, 
                   y1: int, y2: int, y3: FlexNumeric, 
                   frac_x1: FlexNumeric, frac_x2: FlexNumeric, frac_x3: FlexNumeric, 
                   frac_y1: FlexNumeric, frac_y2: FlexNumeric, frac_y3: FlexNumeric, 
                   c7: int, c8: int, c9: int, c10: int, len_lttrs: int) -> Coordinate:
    M: FlexNumeric = 111320
    lat = base_lat + y1 * frac_y1 + y2 * frac_y2 + y3 * frac_y3
    lon = base_lon + x1 * frac_x1 + x2 * frac_x2 + x3 * frac_x3

    lat_add = c7 * len_lttrs + c8
    lat_add /= M # (M/81 * 81)
    lat += lat_add
    
    M_REF = cos(radians(lat)) * M # * frac_y3
    lon_add = c9 * len_lttrs + c10
    lon_add /= M_REF  # (M_REF * 54)
    lon += lon_add

    return lat, lon

def get_rev_reg_cell_code_by_params(code: RegCode, params: CellParams, M: int = 111320) -> Coordinate:
    #len_lttrs = get_len_m_lttrs_from_code(code)
    len_lttrs = get_len_m_lttrs_from_code(code[0], code[0] in '0123yz', 
                                          LTTRS[:10][::-1].index(code[0]) if code[0] in LTTRS[:10] else -1)
    #print("REV code", code, len_lttrs)
    #print("REV params", params)
    rev_coord_cells = get_rev_coord_cell(code[:2])
    #print("rev_coord_cells", rev_coord_cells)
    if len(rev_coord_cells) > 1:
        base_lat, base_lon = rev_coord_cells[0]
    else:
        base_lat, base_lon = get_base_latlon_from_code(code[0], LTTRS.index(code[0]), LTTRS.index(code[1]))

    #if len(rev_coord_cells) > 1:
    #    base_lat, base_lon = rev_coord_cells[0]
    #    #base_lon -= 1 if code[1] == '0' else 0
    #else:
        #lat_int = LTTRS.index(code[0]) + (36 if code[0] not in 'yz' else -26)
        #lon_int = LTTRS.index(code[1]) - 11
    #    lat_int = LTTRS_MAP_DICT[code[0]] + (36 if code[0] not in 'yz' else -26)
    #    lon_int = LTTRS_MAP_DICT[code[1]] - 11
    #    base_lat, base_lon = lat_int, lon_int

    #x1 = (LTTRS_MAP_DICT[code[2]] - params['off1']) % params['n_lttr1']
    #x2 = (LTTRS_MAP_DICT[code[3]] - params['off2']) % params['n_lttr2']
    #x3 = (LTTRS_MAP_DICT[code[6]] - 1) % params['x3']
    #frac_x1 = 1 / params['x1']
    #frac_x2 = 1 / (params['x1'] * params['x2'])
    #frac_x3 = 1 / (params['x1'] * params['x2'] * params['x3'])

    #y1 = (LTTRS_MAP_DICT[code[2]] - params['off1']) // params['n_lttr1']
    #y2 = (LTTRS_MAP_DICT[code[3]] - params['off2']) // params['n_lttr2']
    #y3 = (LTTRS_MAP_DICT[code[5]] - 1) % params['y3']
    #frac_y1 = 1 / params['y1']
    #frac_y2 = 1 / (params['y1'] * params['y2'])
    #frac_y3 = 1 / (params['y1'] * params['y2'] * params['y3'])

    (x1, x2, x3, y1, y2, y3, frac_x1, frac_x2, 
     frac_x3, frac_y1, frac_y2, frac_y3) = get_rev_code_idx_vals(LTTRS_MAP_DICT[code[2]], LTTRS_MAP_DICT[code[3]], 
                                                                 LTTRS_MAP_DICT[code[5]], LTTRS_MAP_DICT[code[6]], 
                                                                 params['x1'], params['x2'], params['x3'], 
                                                                 params['y1'], params['y2'], params['y3'], 
                                                                 params['n_lttr1'], params['n_lttr2'], 
                                                                 params['off1'], params['off2'])
    
    #lat = base_lat + y1 * frac_y1 + y2 * frac_y2 + y3 * frac_y3
    #lon = base_lon + x1 * frac_x1 + x2 * frac_x2 + x3 * frac_x3

    #lat_add = LTTRS_MAP_DICT[code[7]] * len_lttrs + LTTRS_MAP_DICT[code[8]]
    #lat_add /= M # (M/81 * 81)
    #lat += lat_add
    
    #M_REF = math.cos(math.radians(lat)) * M # * frac_y3
    #lon_add = LTTRS_MAP_DICT[code[9]] * len_lttrs + LTTRS_MAP_DICT[code[10]]
    #lon_add /= M_REF  # (M_REF * 54)
    #lon += lon_add
    #print("lat, lon", code, "=>", base_lat, base_lon, "=>", lat, lon)
    #print("len_lttrs", len_lttrs)
    lat, lon = get_rev_latlon(base_lat, base_lon, x1, x2, x3, y1, y2, y3, 
                              frac_x1, frac_x2, frac_x3, frac_y1, frac_y2, frac_y3, 
                              LTTRS_MAP_DICT[code[7]], LTTRS_MAP_DICT[code[8]], 
                              LTTRS_MAP_DICT[code[9]], LTTRS_MAP_DICT[code[10]], len_lttrs)
    return lat, lon

LATLON_MAP_CELL: dict[CoordinateInt, CodeFunc] = {(lat, lon): get_f_map_func(lat, lon, mode=1) 
                                                  for lat, lon in TOTAL_CELLS}
REV_LATLON_MAP_CELL: dict[Cell, RevCodeFunc] = {get_coord_cell(lat, lon): get_f_map_func(lat, lon, mode=2) 
                                               for lat, lon in TOTAL_CELLS \
                                               if not (len(REV_MAP_CELL[get_coord_cell(lat, lon)]) > 1 and \
                                                       get_f_map_func(lat, lon, mode=2) == get_rev_reg_cell_code_base)}
INVALID_LATLONS: set[tuple[Cell, int, int]] = {(get_coord_cell(lat, lon), lat, lon) 
                                              for lat, lon in TOTAL_CELLS \
                                              if (len(REV_MAP_CELL[get_coord_cell(lat, lon)]) > 1 and \
                                                  get_f_map_func(lat, lon, mode=2) == get_rev_reg_cell_code_base)}


def get_reg_cell_code(lat: CoordVal, lon: CoordVal, M: int = 111320, 
                      f_map_func: dict[CoordinateInt, CodeFunc] = LATLON_MAP_CELL) -> str:
    return f_map_func[(int(lat), int(lon))](lat, lon, M)

def get_rev_reg_cell_code(code: RegCode, M: int = 111320, 
                          f_map_func: dict[str, Callable]=REV_LATLON_MAP_CELL) -> Coordinate:
    return f_map_func[code[:2]](code, M)



### Get BBOX params from Code (with 3, 4 or 7 characters)
def get_rev_reg_bbox(code: RegCode) -> BBox:
    params_key = REV_CELL_PARAMS_KEY[code[:2]]
    params = LATLON_CELL_PARAMS[params_key]
    if len(code) == 3:
        min_lat, min_lon = get_rev_reg_cell_code(f"{code}{params['off2']}-110000")
        max_lat = min_lat + 1/params['y1']
        max_lon = min_lon + 1/params['x1']
    elif len(code) == 4:
        min_lat, min_lon = get_rev_reg_cell_code(f"{code}-110000")
        max_lat = min_lat + 1/(params['y1'] * params['y2'])
        max_lon = min_lon + 1/(params['x1'] * params['x2'])
    elif len(code[:7]) >= 7:
        min_lat, min_lon = get_rev_reg_cell_code(f"{code[:7]}0000")
        max_lat = min_lat + 1/(params['y1'] * params['y2'] * params['y3'])
        max_lon = min_lon + 1/(params['x1'] * params['x2'] * params['x3'])

    return min_lon, min_lat, max_lon, max_lat


def get_c3_by_cell(cell: Cell) -> list[RegCode]:
    params_key = REV_CELL_PARAMS_KEY[cell]
    #x1 = LATLON_CELL_PARAMS[params_key]['x1']
    y1 = LATLON_CELL_PARAMS[params_key]['y1']
    #ext_lon = LATLON_CELL_PARAMS[params_key]['extent_lon']
    ext_lat = LATLON_CELL_PARAMS[params_key]['extent_lat']
    n_lttr1 = LATLON_CELL_PARAMS[params_key]['n_lttr1']
    #n_lttr2 = LATLON_CELL_PARAMS[params_key]['n_lttr2']
    off1 = LATLON_CELL_PARAMS[params_key]['off1']
    #print("params_key, off1:", params_key, n_lttr1, n_lttr2, off1)
    
    c3_gen = product(range(int(ext_lat*y1)), range(n_lttr1))
    return [f"{cell}{LTTRS[y1 * n_lttr1 + x1 + off1]}" for y1, x1 in c3_gen]


def get_c4_by_cell(cell: Cell) -> list[RegCode]:
    params_key = REV_CELL_PARAMS_KEY[cell]
    #x1 = LATLON_CELL_PARAMS[params_key]['x1']
    y1 = LATLON_CELL_PARAMS[params_key]['y1']
    #x2 = LATLON_CELL_PARAMS[params_key]['x2']
    y2 = LATLON_CELL_PARAMS[params_key]['y2']
    #ext_lon = LATLON_CELL_PARAMS[params_key]['extent_lon']
    ext_lat = LATLON_CELL_PARAMS[params_key]['extent_lat']
    n_lttr1 = LATLON_CELL_PARAMS[params_key]['n_lttr1']
    n_lttr2 = LATLON_CELL_PARAMS[params_key]['n_lttr2']
    off1 = LATLON_CELL_PARAMS[params_key]['off1']
    off2 = LATLON_CELL_PARAMS[params_key]['off2']
    #print("params_key, off1, off2:", params_key, n_lttr1, n_lttr2, off1, off2, x2*y2)
    
    c4_gen = product(range(int(ext_lat*y1)), range(n_lttr1), range(y2), range(n_lttr2))
    return [f"{cell}{LTTRS[y1 * n_lttr1 + x1 + off1]}{LTTRS[y2 * n_lttr2 + x2 + off2]}" for y1, x1, y2, x2 in c4_gen]

