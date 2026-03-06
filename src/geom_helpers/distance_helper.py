from math import cos, radians, sqrt, pow
import numpy as np
import numpy.typing as npt
from numba import njit
from shapely.geometry import Point, LineString, MultiLineString
from typing import cast

from basic_helpers.types_base import Coordinate, FlexNumeric, CoordVal

M = 111320

@njit(fastmath=True)
def get_dist(pt1: Coordinate, pt2: Coordinate, M: FlexNumeric = 111320): # old: 111229.83
    '''
    Input:
    - pt1, pt2: Zwei Tupel mit den Koordinaten (Breite, Länge, Höhe) oder (Breite, Länge) von Trackpunkten (Ausgangspunkt - Zielpunkt),
    z.B.  (48.0552, 11.723, 510), (48.0552, 11.723)
    
    Output:
    - Distanz in M (float) zwischen den beiden Punkten
    
    * Ermittelt die Entfernung zwischen Ausgangspunkt und Zielpunkt mit Hilfe des Satzes von Pythagoras
    * berücksichtigt die spherische Form der Erde (nur geringfügiger Unterschied zur vergleichbaren "Haversine Distance")
        - Unterschied beträgt annähernd konstant 0.07%
        - auf 1km = 70cm
        - auf 200km = 140m
    * berücksichtigt auch einen etwaigen Höhenunterschied zwischen Ausgangspunkt und Zielpunkt (wobei der Einfluss eher gering ist)
    '''
    if len(pt1) == 3:
        lat1, lon1, elev1 = pt1
        lat2, lon2, elev2 = pt2
        
        h2 = (elev1 - elev2)**2
    else:
        lat1, lon1 = pt1
        lat2, lon2 = pt2
        h2 = 0
    
    a2 = pow((lat1 - lat2) * M, 2)
    b2 = pow((lon1 - lon2) * M * cos(radians(0.5*lat1 + 0.5*lat2)), 2)    
    c2 = a2 + b2
    
    return sqrt(c2+h2)


# Create circle / ellipse around a point with a given distance
def get_dist_circle_coords(d: FlexNumeric, lat: CoordVal, lon: CoordVal, 
                           M: FlexNumeric = 111320, num_points: int = 21, 
                           geo_json: bool=True) -> list[tuple[CoordVal, CoordVal]]:
    '''
    Create a circle around a point with a given distance in meter:
    In reality it is a circle, but in Web Mercator Projection and in degrees it is distorted to an ellipse
    To get the lat-lon coordinates of the circle points we need to compute an ellipse.

    Input:
    - d:   Distanz in M (float) vom Mittelpunkt
    - lat, lon: Koordinaten des Mittelpunkts
    - M:   Konstante für Umrechnung von Grad in Meter
    - num_points: Anzahl der Punkte, durch die der Kreis geführt wird. 
    
    
    Output:
    - Distanz in M (float) zwischen den beiden Punkten
    - Die Abschnitte zwischen den Punkte sind Geraden => kein echter Kreis, sondern ein Vieleck
    '''
    arr: npt.NDArray[np.floating] = np.ones((num_points,2)) * np.array([lon, lat])

    # Parameters for the ellipse => now in array formula
    #a = d / M / (math.cos(math.radians(lat))) # x-Richtung = größerer Wert
    #b = d / M   # y-Richtung

    # Generate points on the ellipse
    theta: npt.NDArray[np.floating] = np.linspace(0, 2*np.pi, num_points)
    arr[:, 1] += d / M * np.sin(theta)
    arr[:, 0] += (d/M) / np.cos(np.radians(0.5 * (arr[:,1] + lat))) * np.cos(theta)
    
    if geo_json:
        return list(zip(arr[:, 0], arr[:, 1]))
    else:
        return list(zip(arr[:, 1], arr[:, 0]))

@njit(fastmath=True)
def convert_lonlat_to_m(lon: CoordVal, lat: CoordVal, 
                        base_lon: int, base_lat: int) -> tuple[np.floating, np.floating]:
    M: int = 111320
    lat_m = np.float32((lat - base_lat) * M)
    #lon_m = np.float32((lon - base_lon) * M * (cos(radians(0.5*(lat + base_lat)))))
    lon_m = np.float32((lon - base_lon) * M * (cos(radians(lat))))
    return lon_m, lat_m

# Distance of points to a line in M
def get_dist_point_to_line(pt: Point, line: LineString | MultiLineString) -> float | None:
    '''
    Approximates the shortest distance of a Point to a line

    pt = Shapely Point
    line = Shapely LineString or MultiLineString

    returns the shortest distance in M
    '''

    # Convert to M
    base_lon, base_lat = np.floor(pt.x), np.floor(pt.y)
    pt_m = Point(*convert_lonlat_to_m(pt.x, pt.y, base_lon, base_lat))
    if line.geom_type == 'LineString':
        line_m = LineString([convert_lonlat_to_m(*c, base_lon, base_lat) for c in line.coords])
    elif line.geom_type == 'MultiLineString':
        line_m = MultiLineString([[convert_lonlat_to_m(*c, base_lon, base_lat) for c in g.coords] for g in line.geoms])
    else:
        return None
        
    # Computation with Shapely
    d = float(pt_m.distance(line_m))

    return d

   
# Compute length of a LineString in M
def get_dist_from_linestring(lstr: str) -> float | None:
    '''
    lstr = Text-represenation of Shapely LineString (or MultiLineString), e.g. 'LINESTRING (0 1, 1 0, 2 2)'

    returns length in M (= sum of the lengths of all line segments)
    '''
    if str(lstr).startswith("MULTILINE"):
        return get_dist_from_multilinestring(lstr)
    
    nodes = [(float(xy[1]), float(xy[0])) for xy in [n.split(" ") for n in str(lstr)[12:-1].split(", ")]]
    
    total_dist = sum([get_dist(pt1, pt2) for pt1, pt2 in zip(nodes[:-1], nodes[1:])])
    
    return cast(float, total_dist)

# Compute length of a MultiLineString in M
def get_dist_from_multilinestring(lstr: str) -> float:
    '''
    lstr = Text-represenation of Shapely MultiLineString (e.g. 'MULTILINESTRING ((0 0, 2 0), (0 1, 1 0, 2 2))')

    returns length in M (= sum of the lengths of all lines in MultiLineString)
    '''
    total_dist = 0
    for sngl_line in str(lstr)[17:-1].split("), ("):
        nodes = [(float(xy[1].replace(")", "")), 
                  float(xy[0].replace("(", ""))) for xy in [n.split(" ") for n in sngl_line.split(", ")]]

        total_dist += sum([get_dist(pt1, pt2) for pt1, pt2 in zip(nodes[:-1], nodes[1:])])
        #print("====", total_dist)
    return total_dist


