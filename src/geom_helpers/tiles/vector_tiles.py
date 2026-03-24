import os
#from geojson2vt.geojson2vt import geojson2vt
from vt2pbf import Tile

from typing import TypeAlias, Any

from basic_helpers.file_helper import do_unpickle

# The result of serialize_to_bytestring() is raw binary data
TilePbf: TypeAlias = bytes

LayerName: TypeAlias = str
LayerSource: TypeAlias = tuple[str, str | None, str | None]
LayersDef: TypeAlias = dict[LayerName, list[LayerSource]]

layers_def: LayersDef = {
    'admin_label': [('admin_0_countries_deu.geojson', 'objid', 'name')], 
    'glacier': [('10m_glaciated_areas.geojson', None, None)], 
    'admin': [('10m_admin_0_countries_deu.geojson', None, None)], 
    'urbanarea': [('10m_urban_areas.geojson', None, None)], 
    'water': [('10m_lakes.geojson', None, None), 
            ('10m_lakes_europe.geojson', None, None)], 
    'waterway': [('10m_rivers_lake_centerlines.geojson', None, None), 
                ('10m_rivers_europe.geojson', None, None)], 
    'road': [('10m_roads.geojson', None, None)], 
    'railway': [('10m_railroads.geojson', None, None)], 
    'place': [('10m_populated_places.geojson', 'objid', 'name')]}
    # admin1
    # coastline
    # ocean


GEOJSON_DIRECTORY = "C:/01_AnacondaProjects/osmium/natural_earth"

TILES_INDEX_DICT: dict[LayerName, Any] = {
    layer_name: do_unpickle(os.path.join(GEOJSON_DIRECTORY, "tiles_index", 
                                         f"{layer_name}.pkl")) for layer_name in layers_def}

ref_layer_name: LayerName = list(TILES_INDEX_DICT.keys())[0]
# Initializing empty collections with types
TILES_ZXY: set[tuple[int, int, int]] = set()
XYZ_DICT: dict[tuple[int, int, int], Any] = {}

# Accessing __dict__ is a bit 'un-Pythonic' for type checkers. 
# If 'options' is a dict, access it normally.
#EXTENT: int = TILES_INDEX_DICT[ref_layer_name].__dict__['options']['extent']
#BUFFER: float = TILES_INDEX_DICT[ref_layer_name].__dict__['options']['buffer']
# We cast to int/float to ensure the types are locked in.
EXTENT: int = int(TILES_INDEX_DICT[ref_layer_name].options['extent'])
BUFFER: float = float(TILES_INDEX_DICT[ref_layer_name].options['buffer'])

MAX_ZOOM: int = 14    
INDEX_MAX_ZOOM: int = 9

print("EXTENT", EXTENT)
print("BUFFER", BUFFER)

# Der BUFFER > 0 ist wichtig für Polygone. Beim Clipping der Polygone für die passende Tilegröße wird der gewünschte 
# (ausgeschnittene) Linienabschnitt entlang der Ränder des Tiles zu einer Fläche komplettiert, so dass er farbig gefüllt
# werden kann. Ohne Buffer verläuft die zusätzlichen Linien genau am Rand (z.B. x=0) und erscheinen dadurch wie ein Rand 
# des Tiles. Durch den Buffer wird diese künstliche Linie außerhalb des Tiles gezeichnet (in SVG für den User also 
# unsichtbar gezeichnet) und es tritt kein unerwünschter Rand zwischen den Tiles auf.
# vgl. Ussue https://github.com/mapbox/geojson-vt/issues/158

# --- Functions ---

def handle_mvt_request(z: int, x: int, y: int) -> tuple[bool, TilePbf | None]:
    """
    Determines if a tile should be served and fetches it.
    """
    if z <= INDEX_MAX_ZOOM:
        return True, None
    else:
        pbf = get_mvt_tile(z, x, y)
        if pbf is not None:
            return True, pbf
        else:
            return False, None

def get_mvt_tile(z: int, x: int, y: int) -> TilePbf | None:
    """
    Generates the PBF bytes for a specific ZXY coordinate.
    """
    try:
        vector_tile = Tile(extend=EXTENT)
        for layer_name, index in TILES_INDEX_DICT.items():
            tile = index.get_tile(z, x, y)
            if tile is not None:
                vector_tile.add_layer(layer_name, features=tile.get('features'))

        pbf: TilePbf = vector_tile.serialize_to_bytestring()
        return pbf
    except Exception as e:
        print("Get MVT Fehler:", e)
        return None


## generate_tile_index() und write_mvt_tiles() zurzeit nicht genutzt, da Tiles mit einem Notebook erstellt werden.

'''
def generate_tile_index():
    for layer_name, config in layers_def.items():
        print(layer_name)
        data = None
        for item in config:
            (fname, id, label) = item
            #print(fname)
            if data is None:
                with open(os.path.join(GEOJSON_DIRECTORY, fname), "rb") as f:
                    data = json.loads(f.read())
            else:
                with open(os.path.join(GEOJSON_DIRECTORY, fname), "rb") as f:
                    data2 = json.loads(f.read())
                    
                for i, val in enumerate(data2['bbox']):
                    if i < 2 and val < data['bbox'][i]:
                        data['bbox'][i] = val
                    elif i >= 2 and val > data['bbox'][i]:
                        data['bbox'][i] = val
                    
                data['features'] += data2['features']

        tile_index = geojson2vt(data, {'indexMaxZoom': INDEX_MAX_ZOOM, 
                                        'maxZoom': MAX_ZOOM, 
                                        'indexMaxPoints': 10,    
                                        'extent': EXTENT, 
                                        'buffer': BUFFER, 
                                        'promote_id': id,
                                        'tolerance': 2, 
                                        'lineMetrics': False,})

        TILES_INDEX_DICT[layer_name] = tile_index

        for coords_dict in tile_index.tile_coords:
            z, x, y = coords_dict.values()
            TILES_ZXY.add((z,x,y))

        #break

def write_mvt_tiles() -> None:
    for z in range(16):
        XYZ_DICT[z] = [*deg2num(30, -8, z)] + [*deg2num(80, 32, z)]

    for (z, x, y) in sorted(TILES_ZXY):
        if z > INDEX_MAX_ZOOM:
            continue
        elif x < XYZ_DICT[z][0] - 1 or x > XYZ_DICT[z][2] + 1:
            continue
        elif y > XYZ_DICT[z][1] + 1 or y < XYZ_DICT[z][3] - 1:
            continue
            
        vector_tile = Tile(extend=EXTENT)
        for layer_name in TILES_INDEX_DICT:
            tile = TILES_INDEX_DICT[layer_name].get_tile(z, x, y)
            if tile is not None:
                vector_tile.add_layer(layer_name, features=tile.get('features'))
            
        # encode vector tile into pbf
        #pbf = vt2pbf(vector_tile, layer_name="water")
        pbf = vector_tile.serialize_to_bytestring()
        #print(pbf)
        
        if os.path.exists(os.path.join(TILES_DIRECTORY, str(z))) == False:
            os.mkdir(os.path.join(TILES_DIRECTORY, str(z)))

        if os.path.exists(os.path.join(TILES_DIRECTORY, str(z), str(x))) == False:
            os.mkdir(os.path.join(TILES_DIRECTORY, str(z), str(x)))

        with open(os.path.join(TILES_DIRECTORY, str(z), str(x), f"{y}.mvt"), "wb") as f:
            f.write(pbf)        

'''
