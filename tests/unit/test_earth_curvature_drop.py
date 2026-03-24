import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from geom_helpers.earth_curvature_drop import (
    calculate_curvature_drop,
    calculate_visible_height,
    calculate_angle_of_elevation
)

# Mocking the FlexNumeric type for local testing context
# In a real run, this is imported from basic_helpers.types_base

@pytest.fixture
def mountain_data():
    """Load mountain data from the generated CSV."""
    data_path = Path(__file__).parent.parent / "data" / "mountains.csv"
    return pd.read_csv(data_path)

def test_calculate_curvature_drop_basic():
    """Test the basic and exact curvature formulas for 100km."""
    dist = 100
    radius = 6371
    drop, exact = calculate_curvature_drop(dist, radius)
    
    # Simple: (100^2) / (2 * 6371) = 0.7848 km
    assert pytest.approx(drop, rel=1e-4) == 0.784806
    # Exact: sqrt(100^2 + 6371^2) - 6371 = 0.78475 km
    assert pytest.approx(exact, rel=1e-4) == 0.784758

def test_calculate_curvature_drop_zero():
    """Test that zero distance results in zero drop."""
    drop, exact = calculate_curvature_drop(0, 6371)
    assert drop == 0
    assert exact == 0

def test_calculate_visible_height_no_refraction():
    """Test visible height without atmospheric refraction."""
    # Drop of 1km, Peak of 5000m, Obs at 0m.
    # 5000 - (1 * 1000) = 4000m
    drop_tuple = (1.0, 1.0)
    res = calculate_visible_height(drop_tuple, 5000, obs_elev=0.0, incl_terr_refract=False)
    assert res == 4000.0

def test_calculate_visible_height_with_refraction():
    """Test visible height including refraction (k=0.13)."""
    # 5000 - 1000 + (1000 * 0.13) = 4130m
    drop_tuple = (1.0, 1.0)
    res = calculate_visible_height(drop_tuple, 5000, obs_elev=0.0, k=0.13, incl_terr_refract=True)
    assert res == 4130.0

def test_calculate_angle_of_elevation_from_comments(mountain_data):
    """
    Test using coordinates from the prompt.
    Scenario: Zugspitze seen from Attenham.
    Approx distance is ~104km.
    """
    # Extract data
    attenham = mountain_data[mountain_data['name'] == 'Attenham'].iloc[0]
    zugspitze = mountain_data[mountain_data['name'] == 'Zugspitze'].iloc[0]
    
    dist_km = 104.0  # Assumed distance for unit test stability
    radius = 6371
    
    drop = calculate_curvature_drop(dist_km, radius)
    angle = calculate_angle_of_elevation(
        curvature_drop=drop,
        peak_height=int(zugspitze['ele']),
        distance=dist_km,
        obs_elev=float(attenham['ele']),
        incl_terr_refract=False
    )
    
    # Expected angle should be positive as Zugspitze (2962) is much higher than Attenham (710)
    assert angle > 0
    assert isinstance(angle, float)

@pytest.mark.parametrize("dist, expected_drop", [
    (1, 0.000078),
    (10, 0.007848),
])
def test_curvature_scaling(dist, expected_drop):
    """Verify scaling logic for small distances."""
    drop, _ = calculate_curvature_drop(dist, 6371)
    assert pytest.approx(drop, rel=1e-2) == expected_drop

def test_numpy_compatibility():
    """Ensure the functions handle numpy floating types correctly."""
    dist = np.float64(50.0)
    radius = 6371
    drop, exact = calculate_curvature_drop(dist, radius)
    assert isinstance(drop, (float, np.floating))
    assert isinstance(exact, (float, np.floating))