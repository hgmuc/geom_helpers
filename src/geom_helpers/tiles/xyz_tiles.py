import os
import requests  # type: ignore
from shapely.geometry import Polygon

from math import radians, tan, atan, sinh, asinh, degrees, pi

from basic_helpers.types_base import CoordVal, Url

TILES_DIRECTORY = "C:/05_Python/tiles"
ELEV_DATA_PATH = "C:/05_Python/awstiles"
REF_BBOX = Polygon([[-8,30], [-8,80], [32,80], [32,30]])
HIGH_RES_BBOX = Polygon([[5,43], [5,56], [15,56], [15,50], [19,50], [19,43]])

def deg2num(lat_deg: CoordVal, lon_deg: CoordVal, zoom: int) -> tuple[int, int]:
    lat_rad = radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - asinh(tan(lat_rad)) / pi) / 2.0 * n)
    return xtile, ytile

def num2deg(x: int, y: int, zoom: int) -> tuple[CoordVal, CoordVal]:
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = atan(sinh(pi * (1 - 2 * y / n)))
    lat_deg = degrees(lat_rad)
    return (lat_deg, lon_deg)

def make_folder(x: int, z: int, root: str = "") -> None:
    path_z = os.path.join(root, str(z))
    path_x = os.path.join(root, str(z), str(x))
    
    if not os.path.exists(path_z):
        #print("mkdir", path_z)
        os.mkdir(path_z)
        
    if not os.path.exists(path_x):
        #print("mkdir", path_x)
        os.mkdir(path_x)

def check_path(x: int, z: int, path: str) -> bool:
    if not os.path.exists(path):
        print("Base path does not exist:", path)
        return False
    
    if not os.path.exists(os.path.join(path, str(z))):
        print("Create zoom folder:", os.path.join(path, str(z)))
        os.mkdir(os.path.join(path, str(z)))

    if not os.path.exists(os.path.join(path, str(z), str(x))):
        print("Create x folder:", os.path.join(path, str(z), str(x)))
        os.mkdir(os.path.join(path, str(z), str(x)))
        
    return True

def download_remote_file(url: Url) -> str | None:
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.content     # type: ignore
        elif response.status_code == 404:
            print("Fehler 404", response.status_code, url)
        else:
            print("Fehler <Code>:", response.status_code, url)
        return None
    except Exception as e:
        print("Fehler OFFLINE", url, e)
        return None

def write_file(content: str | bytes | None, fpath: str) -> bool:
    if content is None:
        return False
    with open(fpath, "wb") as f:
        if isinstance(content, bytes):
            f.write(content)
        else:
            f.write(str(content).encode("UTF-8"))

    return True

def load_file(fpath: str) -> bytes:
    with open(fpath, "rb") as f:
        data = f.read()

    return data

