import os
import pytest
import numpy as np

from geom_helpers.elevation.elevation_helper import get_elevation_data, get_ref_full_bbox

eps = 1e-10

def test_get_elevation_data1():
    fpath = "tests/data/1/0/1.png"
    assert os.path.exists(fpath)
    
    arr = get_elevation_data(fpath)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 256
    assert arr.shape[1] == 256
    assert np.abs(np.sum(arr)) > 100_000

def test_get_elevation_data2():
    fpath = "tests/data/2/1/2.png"
    assert os.path.exists(fpath)
    
    arr = get_elevation_data(fpath)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == 256
    assert arr.shape[1] == 256
    assert np.abs(np.sum(arr)) > 100_000


@pytest.mark.parametrize("x, y, z, expected", [
    (0, 0, 1, (0, -180, 85, 0)), 
    (1, 0, 1, (0, 0, 85, 180)), 
    (0, 1, 1, (-85, -180, 0, 0)), 
    (1, 1, 1, (-85, 0, 0, 180)), 
    (1, 1, 2, (0, -90, 67, 0)), 
    (1, 2, 2, (-67, -90, 0, 0)), 
    (2, 1, 2, (0, 0, 67, 90)), 
    (4, 2, 3, (41, 0, 67, 45)), 
    (4, 3, 3, (0, 0, 41, 45)), 
    (4, 4, 3, (-41, 0, 0, 45)), 
    (8, 5, 4, (41, 0, 56, 23)), 
    (8, 6, 4, (22, 0, 41, 23)), 
])
def test_get_ref_full_bbox(x, y, z, expected, capsys):
    bbox = get_ref_full_bbox([x], [y], z)
    #with capsys.disabled():
    #    print(f"{x}, {y}, {z} -", bbox)
    
    assert round(bbox[0] + eps) == expected[0]
    assert round(bbox[1] + eps) == expected[1]
    assert round(bbox[2] + eps) == expected[2]
    assert round(bbox[3] + eps) == expected[3]
