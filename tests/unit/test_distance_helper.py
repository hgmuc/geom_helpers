# type: ignore
import pytest
from shapely.geometry import Point, LineString, MultiLineString
from geom_helpers.distance_helper import (
    get_dist, 
    get_dist_circle_coords, 
    convert_lonlat_to_m, 
    get_dist_point_to_line,
    get_dist_from_linestring,
    get_dist_from_multilinestring
)

# Constants for verification
M_CONST = 111320

## --- get_dist Tests ---

def test_get_dist_2d():
    """Test distance calculation between two 2D points."""
    pt1 = (48.0, 11.0)
    pt2 = (48.1, 11.1)
    # Expected: Pythagoras with M and cosine correction
    # lat_diff = 0.1 * 111320 = 11132
    # lon_diff = 0.1 * 111320 * cos(rad(48.05)) ≈ 77441.5
    # dist ≈ sqrt(11132^2 + 7441.5^2) ≈ 13390
    dist = get_dist(pt1, pt2)
    assert isinstance(dist, float)
    assert dist > 0
    assert pytest.approx(dist, rel=1e-3) == 13390.2

def test_get_dist_3d():
    """Test distance calculation including elevation (3rd element)."""
    pt1 = (48.0, 11.0, 500.0)
    pt2 = (48.0, 11.0, 600.0)
    # Same lat/lon, 100m elevation diff
    dist = get_dist(pt1, pt2)
    assert pytest.approx(dist) == 100.0

## --- get_dist_circle_coords Tests ---

def test_get_dist_circle_coords_geojson_true():
    """Verify GeoJSON format (Lon, Lat) and point count."""
    num_pts = 10
    coords = get_dist_circle_coords(d=1000, lat=48.0, lon=11.0, num_points=num_pts, geo_json=True)
    
    assert len(coords) == num_pts
    # GeoJSON is (Lon, Lat)
    assert coords[0][0] > 11.0  # Theta 0 starts at positive x (lon) shift

def test_get_dist_circle_coords_geojson_false():
    """Verify standard format (Lat, Lon)."""
    coords = get_dist_circle_coords(d=1000, lat=48.0, lon=11.0, num_points=5, geo_json=False)
    # Standard is (Lat, Lon)
    assert coords[0][0] == 48.0 # At theta=0, sin(0)=0, so lat stays same

## --- convert_lonlat_to_m Tests ---

def test_convert_lonlat_to_m():
    """Test the Numba-jitted conversion to meter offsets."""
    res = convert_lonlat_to_m(11.1, 48.1, 11, 48)
    assert len(res) == 2
    # lat_m = (48.1 - 48) * 111320 = 11132
    assert pytest.approx(res[1]) == 11132.0

## --- get_dist_point_to_line Tests ---

def test_get_dist_point_to_line_linestring():
    """Test shortest distance from a point to a LineString."""
    pt = Point(11.0, 48.0)
    # Line 111.32m away (0.001 degrees north)
    line = LineString([(10.9, 48.001), (11.1, 48.001)])
    dist = get_dist_point_to_line(pt, line)
    assert pytest.approx(dist, abs=0.01) == 111.32

def test_get_dist_point_to_line_multilinestring():
    """Test shortest distance from a point to a MultiLineString."""
    pt = Point(11.0, 48.0)
    line = MultiLineString([
        [(10.0, 45.0), (10.1, 45.1)], # Far away
        [(10.9, 48.001), (11.1, 48.001)] # Close
    ])
    dist = get_dist_point_to_line(pt, line)
    assert pytest.approx(dist, abs=1) == 111.32

## --- WKT String Parsing Tests ---

def test_get_dist_from_linestring_parsing():
    """Test the manual string parsing for LINESTRING."""
    # Note: Function expects 'LINESTRING (lon lat, lon lat)' format
    wkt = "LINESTRING (11.0 48.0, 11.0 48.001)"
    dist = get_dist_from_linestring(wkt)
    assert pytest.approx(dist, abs=1) == 111.32

def test_get_dist_from_multilinestring_parsing():
    """Test the manual string parsing for MULTILINESTRING."""
    wkt = "MULTILINESTRING ((11.0 48.0, 11.0 48.001), (12.0 49.0, 12.0 49.001))"
    dist = get_dist_from_multilinestring(wkt)
    # Should sum both 111.32m segments
    assert pytest.approx(dist, abs=2) == 222.64

def test_linestring_delegates_to_multiline():
    """Ensure get_dist_from_linestring correctly redirects MULTILINE strings."""
    wkt = "MULTILINESTRING ((11.0 48.0, 11.0 48.001))"
    dist = get_dist_from_linestring(wkt)
    assert pytest.approx(dist, abs=1) == 111.32
