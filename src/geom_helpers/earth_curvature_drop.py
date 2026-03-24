import math
import numpy as np
from itertools import combinations

from geom_helpers.distance_helper import get_dist
from basic_helpers.types_base import FlexNumeric

# FlexNumeric: TypeAlias = int | float | np.floating

def calculate_curvature_drop(distance: FlexNumeric, radius: int) -> tuple[FlexNumeric, FlexNumeric]:
    curvature_drop = (distance ** 2) / (2 * radius)   # Einfache Formel gemäß Wiki: y = L**2 / (2*R)
    curvature_drop_exact = np.sqrt(distance**2 + radius**2) - radius # Genauere Formel y = SQRT(L**2 + R**2) - R
    return curvature_drop, curvature_drop_exact

# Sichtbare Höhe
def calculate_visible_height(curvature_drop: tuple[FlexNumeric, FlexNumeric], peak_height: int, 
                             obs_elev: float = 0, k: float = 0.13, incl_terr_refract: bool = True) -> FlexNumeric:
    visible_height = peak_height - curvature_drop[0]*1000 + obs_elev
    if incl_terr_refract:
        visible_height = visible_height + curvature_drop[0]*1000 * k
    return visible_height

# Höhenwinkel (sichtbare Höhe ausgdrückt als Winkel; wenn größer 0, dann sichtbar)
def calculate_angle_of_elevation(curvature_drop: tuple[FlexNumeric, FlexNumeric], peak_height: int, 
                                 distance: FlexNumeric, obs_elev: float = 0, k: float = 0.13, 
                                 incl_terr_refract: bool = True, visible_height: FlexNumeric | None = None):
    if visible_height is None:
        visible_height = calculate_visible_height(curvature_drop, peak_height, obs_elev, k, incl_terr_refract)
    angle_radians = math.atan(visible_height / (distance * 1000))
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


# Input values
#distance_between_points = 100  # in kilometers
#earth_radius = 6371  # 6371 = Earth's radius in kilometers  / In Wikipedia wird mit 6378 km gerechnet

#curvature_drop, curvature_drop_exact = calculate_curvature_drop(distance_between_points, earth_radius)

#for d in [1,10,20,50,100,150,250,500]:
#    curvature_drop, curvature_drop_exact = calculate_curvature_drop(d, earth_radius)
#    print(f"The drop in height due to Earth's curvature over a distance of {d} km is approximately {curvature_drop*1000:.3f} m / {curvature_drop_exact*1000:.3f} m.")



if __name__ == "__main__":
    mtns = [
        ("Attenham", 47.93135, 11.55925, 710), ("Ebersberg", 48.0901382, 11.9611031, 630), 
        ("Taching am See", 47.9633356, 12.7275560, 470), ("Wendelstein", 47.7034761, 12.0120518, 1838), 
        ("Arber", 49.1124718, 13.1361900, 1456), ("Watzmann", 47.5543889, 12.9220342, 2713), 
        ("Dachstein", 47.4751873, 13.6056895, 2995), ("Zugspitze", 47.4212150, 10.9862970, 2962), 
        ("Ortler", 46.5089909, 10.5448653, 3905), ("Großvenediger", 47.1092664, 12.3453356, 3657), 
        ("Wallberg", 47.6658847, 11.7967506, 1722), ("Zwiesel", 47.7235595, 11.4884501, 1348), 
        ("Schafberg", 47.7765811, 13.4334854, 1783), ("Feldberg", 47.8739912, 8.0046735, 1493),
        ("Mont Blanc", 45.8327056, 6.8651706, 4806), ("Matterhorn", 45.9764263, 7.6586024, 4478)]

    for idx1, idx2 in combinations(range(len(mtns)), r=2):
        mtn1_name, mtn1_lat, mtn1_lon, mtn1_ele = mtns[idx1]
        mtn2_name, mtn2_lat, mtn2_lon, mtn2_ele = mtns[idx2]
        dist = get_dist((mtn1_lat, mtn1_lon), (mtn2_lat, mtn2_lon))/1000

        curvature_drop = calculate_curvature_drop(dist, 6371)
        angle_of_elevation = calculate_angle_of_elevation(curvature_drop, mtn2_ele, dist, mtn1_ele, 0.13, False)
        print(f"{mtn1_name:20} {mtn2_name:20}    {round(dist):4} km   {angle_of_elevation:6.2f} degrees")


    print("\n-------------------------\n")

    # Input values
    peak_height = 4810  # Peak height in meters
    obs_elevation = 4470 # Height of observation point in meters
    distance_to_peak = 50  # Distance to peak in kilometers
    curvature_drop = calculate_curvature_drop(distance_to_peak, 6371)  # Calculating curvature drop over x km

    for distance_to_peak in [50,100,150,200,250,300]:
        curvature_drop = calculate_curvature_drop(distance_to_peak, 6371)
        angle_of_elevation = calculate_angle_of_elevation(curvature_drop, peak_height, distance_to_peak, obs_elevation, 0.13, True)
        print(f"The angle of elevation from your viewpoint to the peak ({distance_to_peak} km) is approximately {angle_of_elevation:.2f} degrees.")













