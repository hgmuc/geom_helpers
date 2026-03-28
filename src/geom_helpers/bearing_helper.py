import math
from numba import njit

from basic_helpers.types_base import Coordinate # FlexNumeric

@njit(fastmath=True)
def get_bearing(pt1: Coordinate, pt2: Coordinate) -> float:
    '''
    Input:
    - pt1, pt2: Zwei Tupel mit den Koordinaten (Breite, Länge) von Trackpunkten (Ausgangspunkt - Zielpunkt), z.B. (48.0552, 11.723)
    
    Output:
    - brng: Richtung in Grad, um vom Ausgangspunkt - Zielpunkt zu gelangen (Nord: 0 Grad - Ost: 90 Grad - Süd: 180 Grad - West: 270 Grad)
    
    * liefert Richtung in Grad, um vom Ausgangspunkt - Zielpunkt zu gelangen
    * sämtliche Gradangaben werden dabei für die Berechnung in Radians umgewandelt
    '''    

    lat1, lon1  = pt1[:2]
    lat2, lon2  = pt2[:2]
    if lat1 == lat2 and lon1 == lon2:
        return 0
    
    dlon = lon2 - lon1
    
    y = math.sin(math.radians(dlon)) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) 
    x -= math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dlon))
    
    brng = math.atan2(y, x)  
    brng = math.degrees(brng)
    
    brng = (brng + 360) % 360
    
    return brng 


def get_track_bearings(track_pts: list[Coordinate], printout: bool = False) -> list[str]:
    '''
    Input:
    - Liste von Track-Punkt Koordinaten: list[tuple[lat, lon]]
    
    Output:
    - bearing_ls: Liste mit als String formatierte Gradangaben

    Optional: Printausgabe aller Wegpunkte und der Gradangaben
    '''
    print("-----")
    bearing_ls: list[str] = []
    for i, tpt in enumerate(track_pts[:-1]):
        bearing = get_bearing(tpt, track_pts[i+1])
        bearing_ls.append("{:7.2f}".format(bearing))
        print(i, tpt, bearing, bearing_ls)
        if printout:
            frmt_str = "{:5}    ({:10.6f}, {:10.6f}) -> ({:10.6f}, {:10.6f})  ===>   {:7.2f}"
            print(frmt_str.format(i+1, *tpt[:2], *track_pts[i+1][:2], bearing))
        
    return bearing_ls


def get_track_bearings_num(track_pts: list[Coordinate]) -> list[float]:
    '''
    Input:
    - Liste von Track-Punkt Koordinaten: list[tuple[lat, lon]]
    
    Output:
    - bearing_ls: Liste der gerundeten Gradangaben für den Track
    '''
    bearing_ls: list[float] = []
    for i, tpt in enumerate(track_pts[:-1]):
        bearing_ls.append(round(get_bearing(tpt, track_pts[i+1]), 1))

    return bearing_ls




