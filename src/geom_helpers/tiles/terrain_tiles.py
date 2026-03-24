from geom_helpers.tiles.terrain_tiles_copernicus import create_terrain_tile_from_copernicus
from geom_helpers.tiles.terrain_tiles_mapzen import create_terrain_tile_from_mapzen
from matplotlib.axes import Axes

def handle_terrain_tile_request(z: int, x: int, y: int, ax: Axes) -> bool:
    if int(z) < 10:
        return create_terrain_tile_from_mapzen(z, x, y, ax)
    else:
        res = 30 if int(z) > 11 else 90
        return create_terrain_tile_from_copernicus(int(z), int(x), int(y), ax, res)

