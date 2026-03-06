from __future__ import annotations
from basic_helpers.types_base import CoordVal

UtmCrsString = str

'''
# Example usage
lat = 45.0
lon = 7.0

utm_zone, hemisphere = get_utm_zone(lat, lon)
utm_crs = get_utm_crs(lat, lon)
utm_crs_number = get_utm_crs_number(lat, lon)

print(f"UTM Zone: {utm_zone}")
print(f"Hemisphere: {hemisphere}")
print(f"UTM CRS: {utm_crs}")
print(f"UTM CRS Number: {utm_crs_number}")

'''

def get_utm_zone(lat: CoordVal, lon: CoordVal) -> tuple[int, str]:
    """
    Get the UTM zone number and hemisphere (north/south) for a given latitude and longitude.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        utm_zone (int): UTM zone number.
        hemisphere (str): 'north' or 'south' hemisphere.
    """
    # Calculate the UTM zone number
    utm_zone = int((lon + 180) // 6) + 1

    # Determine the hemisphere
    hemisphere = 'north' if lat >= 0 else 'south'

    return utm_zone, hemisphere

def get_utm_crs(lat: CoordVal, lon: CoordVal) -> UtmCrsString:
    """
    Get the EPSG code for the UTM zone corresponding to the given latitude and longitude.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        utm_crs (str): EPSG code for the UTM zone.
    """
    utm_zone, hemisphere = get_utm_zone(lat, lon)

    # EPSG code for UTM in the northern hemisphere starts at 32600
    # and for southern hemisphere starts at 32700.
    if hemisphere == 'north':
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone

    return f'epsg:{epsg_code}'

def get_utm_crs_number(lat: CoordVal, lon: CoordVal) -> int:
    """
    Get the EPSG code for the UTM zone corresponding to the given latitude and longitude.

    Args:
        lat (float): Latitude in degrees.
        lon (float): Longitude in degrees.

    Returns:
        utm_crs (str): EPSG code for the UTM zone.
    """
    utm_zone, hemisphere = get_utm_zone(lat, lon)

    # EPSG code for UTM in the northern hemisphere starts at 32600
    # and for southern hemisphere starts at 32700.
    if hemisphere == 'north':
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone

    return epsg_code

