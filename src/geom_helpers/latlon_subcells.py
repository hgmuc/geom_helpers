from __future__ import annotations
import math
import numpy as np
from itertools import product
from geom_helpers.osm_reader_helper import get_coord_code, get_rev_coord_code, InvalidCoordinateError
from geom_helpers.distance_helper import get_dist
from geom_helpers.latlon_code import get_reg_cell_code
from typing import Any, cast

from basic_helpers.types_base import Coordinate, FlexNumeric
from basic_helpers.config_reg_code import Cell, Subcell

SubcellDict = dict[Cell, set[Subcell | str]]

# Hilfsfunktionen, um alle Subcells zu bekommen, die in einem bestimmten Bereich liegen
def get_relevant_cells_from_latlon(pt1: Coordinate, pt2: Coordinate, buffer: float = 0.0) -> list[Cell]:
    """Get the cells which are intersecting with BBox defined by the two points"""
    subcell_dict = get_subcells_in_bbox(pt1, pt2, buffer, dimx=1, dimy=1)
    return list(subcell_dict)

def get_subcells_in_bbox(pt1: Coordinate, pt2: Coordinate, buffer: float = 0.0, 
                         dimx: FlexNumeric = 1/10, dimy: FlexNumeric = 1/10, 
                         code_type: Any ='default') -> SubcellDict:
    """
    Returns all subcells of the specified code type and resolution 
    
    Inputs:
    pt1: SW corner of Bbox
    pt2: NE corner of Bbox
    buffer: optional

    dimx / dimy: Resolution of subcells
    code_type: 'default' - classic coord_code from osm_reader_helper -> typical values for resolution: 0.1 / 0.1
    code_type: anything != 'default': new reg_cell_code from latlon_code typical values for resolution:
        -> dimx, dimy = 1/2, 1/3 => typically 6 subcells per cell in base_region 
                => 3-character codes e.g. "AB1, AB2, AB3, ... AB6" - (2 x 3)
        -> dimx, dimy = 1/6, 1/9 => typically 54 subcells per cell in base_region
                => 4-character codes e.g. "AB11, AB12, ..., AB69" - (6 x 9)
        
        Other regions:
        caboverde for entire 5 x 6 degrees region:
            default mode:
            - dimx, dimy = 0.5, 0.5 => max.   30 x 4-character codes: "0000, 0001, 0010, ... 0045" - (5 x 6)
            non-default mode:
            - dimx, dimy = 0.5, 0.5 => max.   56 x 3-character codes: "000, 001, 002, ... 00t" - (7 x 8)
            - dimx, dimy = 0.1, 0.1 => max. 2016 x 4-character codes: "0000, 0001, 0002, ... 00tZ" - (56 x 36)
        
        madeira for entire 2 x 2 degrees region:
            default mode:
            - dimx, dimy = 0.5, 0.5 => max.   4 x 4-character codes: "2000, 2001, 2010, 2011"
            non-default mode: 
            - dimx, dimy = 0.5, 0.5 => max.   35 x 3-character codes: "200, 201, 202, ... 20Z" - (6 x 6)
            - dimx, dimy = 0.1, 0.1 => max. 324 x 4-character codes: "2000, 2001, 2002, ... 20Z8" - (36 x 9)

        azores for entire 4 x 8 degrees region: 
            default mode:
            - dimx, dimy = 0.5, 0.5 => max.   32 x 4-character codes: "3000, 3001, 3002, ... 3037" - (4 x 8)
            non-default mode:
            - dimx, dimy = 0.5, 0.5 => max.   35 x 3-character codes: "300, 301, 302, ... 30l" - (6 x 8)
            - dimx, dimy = 0.1, 0.1 => max. 1728 x 4-character codes: "3001, 1001, 1002, ... 10lZ" - (48 x 36)
        iceland - per 1 x 4 degree cell
            default mode:
            - dimx, dimy = 0.1, 0.5 => max.   30 x 4-character codes: "S000, S001, S002, ..., S009, S050, S051, ... S099" - (5 x 7)
            non-default mode:
            - dimx, dimy = 0.1, 0.5 => max.   18 x 3-character codes: "S00, S01, S02, ... S0H" - (3 x 6)
            - dimx, dimy = 0.1, 0.1 => max.  162 x 4-character codes: "S000, S001, S002, ... S0H8" - (18 x 9)
        canary for entire 5 x 7 degrees region:
            default mode:
            - dimx, dimy = 0.5, 0.5 => max.   35 x 4-character codes: "1000, 1001, 1010, ... 1046" - (5 x 7)
            non-default mode:
            - dimx, dimy = 0.5, 0.5 => max.   35 x 3-character codes: "100, 101, 102, ... 10Y" - (5 x 7)
            - dimx, dimy = 0.1, 0.1 => max. 1960 x 4-character codes: "1001, 1001, 1002, ... 10Yt" - (35 x 56)

    """
    
    len_subcell_code = 4
    if code_type != 'default' and (dimx >= 1/3 or dimy > 1/3):
        len_subcell_code = 3
    
    min_lat, min_lon = pt1
    max_lat, max_lon = pt2
    
    if buffer > 0:
        min_lat -= buffer
        min_lon -= buffer
        max_lat += buffer
        max_lon += buffer
    else:    
        if int(max_lat*10) - max_lat*10 == 0:
            max_lat -= 1e-6
        if int(max_lon*10) - max_lon*10 == 0:
            max_lon -= 1e-6
    
    p = product(list(np.arange(min_lat, max_lat+dimy, dimy)[:-1]) + [max_lat], 
                list(np.arange(min_lon, max_lon+dimx, dimx)[:-1]) + [max_lon])
    #p = list(p)
    #print(p)
    subcell_dict: SubcellDict = {}
    
    if code_type == 'default':
        for lat, lon in p:
            try:
                code = get_coord_code(lat,lon)[:len_subcell_code]
            except InvalidCoordinateError:
                continue
            except Exception as e:
                print("ERROR get_subcells_in_bbox > get_coord_code", e, lat, lon)
                continue
            
            if code[:2] in subcell_dict:
                subcell_dict[code[:2]].add(code[2:])
            else:
                subcell_dict[code[:2]] = set([code[2:]])
            
    else:
        for lat, lon in p:
            try:
                code = get_reg_cell_code(lat,lon)[:len_subcell_code]
            except InvalidCoordinateError:
                continue
            except Exception as e:
                print("ERROR get_subcells_in_bbox > get_coord_code", e, lat, lon)
                continue
            
            if code[:2] in subcell_dict:
                subcell_dict[code[:2]].add(code[2:])
            else:
                subcell_dict[code[:2]] = set([code[2:]])
        
    return subcell_dict


def get_subcells_in_circle(center_pt: Coordinate, radius: FlexNumeric = 10000, 
                           lat_buffer: float = 0.0, M: int = 111320) -> SubcellDict:
    # radius in m
    # Alle Subcells, durch die die Kreislinie läuft und komplett innerhalb liegen, werden ins Ergebnis aufgenommen
    radius += lat_buffer * M
    
    lat, lon = center_pt
    lat_n, lat_s = (lat + (radius / M), lat - (radius / M))
    lon_o, lon_w = (lon + (radius / (M * math.cos(math.radians(lat)))), 
                    lon - (radius / (M * math.cos(math.radians(lat)))))
        
    # 1. Filter BBox
    subcell_dict = get_subcells_in_bbox((lat_s, lon_w), (lat_n, lon_o))
    
    # 2. Filter Abstand vom Mittelpunkt
    for cell in subcell_dict:
        new_ls = []
        for subcell in subcell_dict[cell]:
            subcenter_pt = cast(Coordinate, tuple([x+0.05 for x in get_rev_coord_code(cell+subcell)]))
            if get_dist(center_pt, subcenter_pt) < radius:
                new_ls.append(subcell)
        subcell_dict[cell] = set(new_ls)
        
    # Entfernen von Cells mit leerem Set
    subcell_dict = {k: v for k, v in subcell_dict.items() if len(v) > 0}
    
    return subcell_dict

def get_subcells_in_polygon(poly):
    # poly = Liste von [(lat, lon), (lat, lon), ...] - kann geschlossen sein oder wird geschlossen
    # Alle Subcells, deren Mittelpunkt innerhalb des Polygons liegt werden ins Ergebnis aufgenommen
    
    lat_n = max([pt[0] for pt in poly])
    lat_s = min([pt[0] for pt in poly])
    lon_o = max([pt[1] for pt in poly])
    lon_w = min([pt[1] for pt in poly])
    
    subcell_dict = get_subcells_in_bbox((lat_s, lon_w), (lat_n, lon_o))
    
    for cell in subcell_dict:
        new_subcells = set()
        for subcell in subcell_dict[cell]:
            subcenter_pt = tuple([x+0.05 for x in get_rev_coord_code(cell+subcell)])
            if point_inside_polygon(subcenter_pt, poly):
                new_subcells.add(subcell)
                
        subcell_dict[cell] = new_subcells.copy()
        
    subcell_dict = {k: v for k, v in subcell_dict.items() if len(v) > 0}
    
    return subcell_dict

def point_inside_polygon(pt, poly):
    # aus Stackoverflow /questions/22521982/check-if-a-point-is-inside-a-polygon
    # WICHTIG: lat, lon sind in dieser Funktion zu vertauschen, d.h. y=lat, y=lon - sowohl bei pt als auch bei den poly-Punkten
    y, x = pt
    if pt in poly:
        return True
    
    inside = False
    
    for i, j in zip(list(range(len(poly))), [len(poly)-1] + list(range(len(poly)-1))):
        xi = poly[i][1]
        yi = poly[i][0]
        xj = poly[j][1]
        yj = poly[j][0]
        
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        #print(i, j, xi, yi, xj, yj, intersect)
        if intersect:
            inside = not(inside)
            
    return inside
