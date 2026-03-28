import os
import rasterio
import numpy as np
from abc import ABC, abstractmethod
from typing import Callable, Any, TypeAlias, cast

from basic_helpers.file_helper import do_pickle, do_ungzip_pkl, do_unpickle
from geom_helpers.tiles.xyz_tiles import deg2num, download_remote_file, write_file  #, num2deg
from geom_helpers.elevation.elevation_helper import get_elevation_data, get_ref_full_bbox
from geom_helpers.elevation.pt_elevation import (get_srtm_elevation_numba, get_elevation_from_tile_numba, 
                                                 downsample_1d_arr, upsample_1d_arr)

from basic_helpers.types_base import CoordVal, Url, BBox, ElevArr, ElevInt 


KeyedDataKey: TypeAlias = str | tuple[int, ...]
ZoomInt: TypeAlias = int

DataFilesDict: TypeAlias = dict[int, ElevArr | None]

class ElevationDataSource(ABC):
    def __init__(self, filefmt: str | None = None, res: int = 90, dim: int | str = 1201, 
                 preprocessor: Callable[[Any], ElevArr] | None = None, 
                 path: str | None = None, with_key: bool = True, 
                 try_download: bool = False) -> None:
        self.srctype = 'cell'
        self.filefmt = filefmt
        self.res = res
        self.dim = dim
        self.preprocessor = preprocessor
        self.path = path
        self.with_key = with_key
        self.offset_lon = 0
        self.offset_lat = 0
        self.try_download = try_download

        if with_key:
            self.data: dict[KeyedDataKey, ElevArr] | ElevArr | None = {}
        else:
            self.data = None

        if self.path is not None:
            self.check_path()

    @abstractmethod
    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr:
        ...

    @abstractmethod
    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        ...

    @abstractmethod
    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> Url:
        raise NotImplementedError("Subclasses should implement get_remote_url")
        
    @abstractmethod
    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        return []

    def load_data(self, lat: CoordVal, lon: CoordVal) -> bool:
        data_file = self.load_file(lat, lon)
        if self.preprocessor is None:
            self.data = data_file
        else:
            #print("load_data preprocessing", self.preprocessor)
            self.data = self.preprocessor(data_file)
        
        #print("load_data", type(self.data))
        return True

    def load_keyed_data(self, key: KeyedDataKey, lat: CoordVal, lon: CoordVal) -> bool:
        assert isinstance(self.data, dict)
        data_file = self.load_file(lat, lon)
        
        assert data_file is not None
        if self.preprocessor is None:
            self.data[key] = data_file
        else:
            #print("load_keyed_data preprocessing", self.preprocessor)
            self.data[key] = self.preprocessor(data_file)
            
        return True

    def get_res(self) -> int:
        return self.res

    def get_dim(self) -> int | str:
        return self.dim

    def get_filefmt(self) -> str | None:
        return self.filefmt

    def get_path(self) -> str:
        assert self.path
        return os.path.join(self.path)
    
    def check_path(self) -> None:
        assert self.path
        os.makedirs(self.path, exist_ok=True)                

    def get_fullpath(self, fname: str | None, subfolders: list[str]):
        if fname is None and len(subfolders) > 0:
            return os.path.join(self.get_path(), *subfolders)
        elif fname is None:
            return os.path.join(self.get_path())
        else:                    
            return os.path.join(self.get_path(), *subfolders, f"{fname}.{self.get_filefmt()}")
        
    def get_elevation(self, lat: CoordVal, lon: CoordVal, elevations: ElevArr | None = None) -> ElevInt | None:
        if elevations is None:
            if self.can_load_data(lat, lon):
                elevations = self.get_data_arr(lat, lon, self.with_key)
                if elevations is None:
                    return None
                #return get_srtm_elevation(lat, lon, elevations=elevations, DIMS=self.get_dim())
                dims = cast(int, self.get_dim()) if isinstance(self.dim, int) else elevations.shape[0]
                return get_srtm_elevation_numba(lat, lon, elevations=elevations, DIMS=dims)
            else:
                return None
        else:
            dims = cast(int, self.get_dim()) if isinstance(self.dim, int) else elevations.shape[0]
            return get_srtm_elevation_numba(lat, lon, elevations=elevations, DIMS=dims)

    def get_data_key(self, lat: CoordVal, lon: CoordVal) -> str:
        return self.get_fname(lat, lon)
        
    def has_data_loaded(self, lat: CoordVal, lon: CoordVal) -> bool:
        if isinstance(self.data, dict):
            return self.get_data_key(lat, lon) in self.data
        elif isinstance(self.data, np.ndarray):
            return True
        else:
            return False
    
    def can_load_data(self, lat: CoordVal, lon: CoordVal) -> bool:
        full_path = self.get_fullpath(self.get_fname(lat, lon), 
                                      self.get_subfolders(lat, lon))
        return os.path.exists(full_path)
    
    def get_data_arr(self, lat: CoordVal, lon: CoordVal, with_key: bool = True) -> ElevArr | None:
        if with_key and isinstance(self.data, dict):
            if self.has_data_loaded(lat, lon):
                return self.data[self.get_data_key(lat, lon)]
            elif self.can_load_data(lat, lon):
                done = self.load_keyed_data(self.get_data_key(lat, lon), lat, lon)
                return self.data[self.get_data_key(lat, lon)]
            elif self.try_download :
                url = cast(Url, self.get_remote_url(lat, lon))
                fpath = self.get_fullpath(self.get_fname(lat, lon), 
                                          self.get_subfolders(lat, lon))
                file_content = download_remote_file(url)
                file_ok = write_file(file_content, fpath)
                done = self.load_keyed_data(self.get_data_key(lat, lon), lat, lon)
                return self.data[self.get_data_key(lat, lon)]
            else:
                return None
        else:
            if isinstance(self.data, np.ndarray) and len(self.data) > 0:
                return self.data
            elif self.can_load_data(lat, lon):
                done = self.load_data(lat, lon)
                assert done and isinstance(self.data, np.ndarray)
                return self.data
            elif self.try_download :
                url = self.get_remote_url(lat, lon)
                fpath = self.get_fullpath(self.get_fname(lat, lon), 
                                        self.get_subfolders(lat, lon))
                file_content = download_remote_file(url)
                file_ok = write_file(file_content, fpath)
                assert file_ok
                done = self.load_data(lat, lon)
                assert done and isinstance(self.data, np.ndarray)
                return self.data
                #done = self.load_keyed_data(self.get_data_key(lat, lon), lat, lon)
                #return self.data[self.get_data_key(lat, lon)]
            else:
                return None
            
class ElevationDataFromTiles(ElevationDataSource):
    def __init__(self, filefmt: str | None = None, dim: int | str = 256, zoom_lvls: list[ZoomInt] = [9], 
                 readonly: bool = True, preprocessor: Callable[[Any], ElevArr] | None = None, 
                 path: str | None = None, with_key: bool = True, try_download: bool = False) -> None:
        self.srctype: str = 'tile'
        self.filefmt = filefmt
        self.dim = dim
        self.zoom_lvls = zoom_lvls
        self.readonly = readonly
        self.preprocessor = preprocessor
        self.path = path
        self.with_key = with_key
        self.try_download = try_download

        if with_key:
            self.data: dict[ZoomInt, dict[KeyedDataKey, ElevArr]] | None = {z: {} for z in zoom_lvls}
        else:
            self.data = None

    @abstractmethod
    def load_tile_by_zoom(self, x: int, y: int, zoom: ZoomInt) -> ElevArr:
        ...

    @abstractmethod
    def get_remote_tile_url(self, x: int, y: int, zoom: ZoomInt) -> Url:
        raise NotImplementedError("Subclasses should implement get_remote_url")

    def get_xy_from_latlon(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt) -> tuple[int, int]:
        x, y = deg2num(lat, lon, zoom)
        return x, y
    
    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        raise NotImplementedError("An XYZ tile's name is required. Use self.get_tile_fname(y).", lat, lon)

    def get_tile_fname(self, y: int) -> str:
        return str(y)

    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        raise NotImplementedError("Tile-based sources require zoom. Use get_subfolders_with_zoom().", lat, lon)
    
    def get_subfolders_with_zoom(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt) -> list[str]:
        x, _ = deg2num(lat, lon, zoom)
        return [str(zoom), str(x)]

    def get_tile_subfolders(self, x: int, zoom: ZoomInt) -> list[str]:
        return [str(zoom), str(x)]

    def get_fullpath_from_latlonzoom(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt) -> str:
        x, y = self.get_xy_from_latlon(lat, lon, zoom)
        fname = self.get_tile_fname(y)
        subfolders = self.get_subfolders_with_zoom(lat, lon, zoom)
        return cast(str, self.get_fullpath(fname, subfolders))

    def get_data_key_with_zoom(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt) -> KeyedDataKey:
        return deg2num(lat, lon, zoom)

    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr:
        assert isinstance(zoom, int), f"Zoom {zoom} must be an int"
        x, y = deg2num(lat, lon, zoom)
        return cast(ElevArr, self.load_tile_by_zoom(x, y, zoom))

    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> Url:
        assert isinstance(zoom, int), f"Zoom {zoom} must be an int"
        x, y = deg2num(lat, lon, zoom)
        return cast(Url, self.get_remote_tile_url(x, y, zoom))

    def has_data_loaded(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> bool:
        assert isinstance(zoom, int), f"Zoom {zoom} must be an int"
        return (self.get_data_key_with_zoom(lat, lon, zoom) in self.data[zoom] 
                if isinstance(self.data, dict) and zoom in self.data else False)

    def can_load_data(self, lat: CoordVal, lon: CoordVal) -> bool:
        raise NotImplementedError("can_load_data(lat, lon) is not implemented. Use can_load_tile(lat, lon, zoom) instead.", lat, lon)

    def can_load_tile(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None) -> bool:
        assert isinstance(zoom, int), f"Zoom {zoom} must be an int"
        return os.path.exists(self.get_fullpath_from_latlonzoom(lat, lon, zoom))

    def get_tile_bbox(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None) -> BBox:
        assert isinstance(zoom, int), f"Zoom {zoom} must be an int"
        x, y = deg2num(lat, lon, zoom)
        return get_ref_full_bbox([x], [y], zoom)

    def check_full_path(self, full_path: str) -> None:
        assert full_path
        os.makedirs(full_path, exist_ok=True)                

    def get_data_arr(self, lat: CoordVal, lon: CoordVal, with_key: bool = True) -> ElevArr | None:
        for zoom in self.zoom_lvls:
            data_arr = self.get_data_arr_with_zoom(lat, lon, zoom, with_key)
            if data_arr is not None:
                return data_arr
        
        return None

    def get_data_arr_with_zoom(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt, with_key: bool = True) -> ElevArr | None:
        x, y = deg2num(lat, lon, zoom)
        data_files: DataFilesDict = {i: None for i in range(4)}
        if with_key and isinstance(self.data, dict):            
            if self.has_data_loaded(lat, lon, zoom):
                return self.data[zoom][(x, y)]
            elif self.can_load_tile(lat, lon, zoom) or self.try_download:
                for i, (getx, gety) in enumerate([(x, y), (x+1, y), (x, y+1), (x+1, y+1)]):
                    try:
                        data_files = self.load_data_files(i, getx, gety, zoom, data_files)
                    except Exception as e:
                        if self.try_download:
                            try:
                                url = self.get_remote_tile_url(getx, gety, zoom)
                                dir_only = self.get_fullpath(None, self.get_tile_subfolders(getx, zoom))
                                self.check_full_path(dir_only)
                                fpath = self.get_fullpath(self.get_tile_fname(gety), 
                                                          self.get_tile_subfolders(getx, zoom))
                                file_content = download_remote_file(url)
                                file_ok = write_file(file_content, fpath)
                                if file_ok:
                                    data_files = self.load_data_files(i, getx, gety, zoom, data_files)
                            except Exception as e2:
                                print(i, "could not download tile", url, "and store it to", fpath, e2)
                        elif i == 0:
                            print("cannot load tile", getx, gety, zoom, i, e)
                    
                done = self.load_keyed_tile_data(x, y, zoom, data_files)

                return self.data[zoom][(x, y)]

        else:
            if isinstance(self.data, np.ndarray) and self.data is not None:
                return self.data
            elif self.can_load_tile(lat, lon, zoom):
                for i, (getx, gety) in enumerate([(x, y), (x+1, y), (x, y+1), (x+1, y+1)]):
                    try:
                        data_files = self.load_data_files(i, getx, gety, zoom, data_files)
                    except Exception as e:
                        if self.try_download:
                            try:
                                url = self.get_remote_tile_url(getx, gety, zoom)
                                dir_only = self.get_fullpath(None, self.get_tile_subfolders(getx, zoom))
                                self.check_full_path(dir_only)
                                fpath = self.get_fullpath(self.get_tile_fname(gety), 
                                                          self.get_tile_subfolders(getx, zoom))
                                file_content = download_remote_file(url)
                                file_ok = write_file(file_content, fpath)
                                if file_ok:
                                    data_files = self.load_data_files(i, getx, gety, zoom, data_files)
                            except Exception as e2:
                                print(i, "could not download tile", url, "and store it to", fpath, e2)
                        elif i == 0:
                            print("cannot load tile", getx, gety, zoom, i, e)
                    
                done = self.load_data_xyz(x, y, zoom, data_files)
                assert done and isinstance(self.data, np.ndarray)    
                return self.data

        return None

    def load_data(self, lat: Any, lon: Any) -> bool:
        raise NotImplementedError("Function load_data(lat, lon) not implemented here. Use load_data_xyz(x, y, zoom, data_files) instead", lat, lon)
    
    def load_data_xyz(self, x: int, y: int, zoom: ZoomInt, data_files: DataFilesDict) -> bool:
        xdim, ydim = self.get_data_arr_dims(data_files)
        data_arr = np.zeros((ydim+1, xdim+1), dtype=int)
        data_arr[:ydim, :xdim] = data_files[0]
        
        if 1 in data_files and data_files[1] is not None:
            data_arr = self.append_final_row(data_arr, ydim, data_files)
        if 2 in data_files and data_files[2] is not None:
            data_arr = self.append_final_col(data_arr, xdim, data_files)
        if 3 in data_files and data_files[3] is not None:
            data_arr = self.append_final_cell(data_arr, data_files)
            
        assert isinstance(self.data, dict) # isinstance(data_arr, np.ndarray)
        self.data[zoom][(x, y)] = data_arr
        
        return True

    def load_data_files(self, i: int, x: int, y: int, zoom: ZoomInt, data_files: DataFilesDict) -> DataFilesDict:
        data_file = self.load_tile_by_zoom(x, y, zoom)
        if self.preprocessor is None:
            data_file = data_file
        else:
            #print("load_data preprocessing", self.preprocessor)
            data_file = self.preprocessor(data_file)
        
        data_files[i] = data_file
        return data_files

    def get_tile_list_by_zoom(self, zoom: ZoomInt) -> list[tuple[int, int]]:
        tile_ls = []
        for x in os.listdir(os.path.join(self.get_path(), str(zoom))):
            for y in os.listdir(os.path.join(self.get_path(), str(zoom), x)):
                tile_ls.append((int(x), int(y.replace(f".{self.get_filefmt()}", ""))))

        return tile_ls

    def get_data_arr_dims(self, data_files: DataFilesDict) -> tuple[int, int]:
        if self.dim == 'auto' and data_files[0] is not None:
            ydim, xdim = data_files[0].shape
        else:
            ydim = xdim = cast(int, self.get_dim())
            
        return ydim, xdim
            
    def append_final_row(self, data_arr: ElevArr, ydim: int, data_files: DataFilesDict) -> ElevArr:
        assert data_files[0] is not None
        data_arr[:ydim, -1] = data_files[0][:, -1]
        if data_files[1] is not None:
            upd_len = min(ydim, len(data_files[1][:,0]))
            if len(data_files[1][:,0]) == ydim:
                data_arr[:upd_len, -1] = data_files[1][:upd_len,0]
            
        return data_arr
        
    def append_final_col(self, data_arr: ElevArr, xdim: int, data_files: DataFilesDict) -> ElevArr:
        assert data_files[0] is not None
        data_arr[-1, :xdim] = data_files[0][-1, :]
        if data_files[2] is not None:
            upd_len = min(xdim, len(data_files[2][0, :]))
            data_arr[-1, :upd_len] = data_files[2][0, :upd_len]
            
        return data_arr

    def append_final_cell(self, data_arr: ElevArr, data_files: DataFilesDict) -> ElevArr:
        if data_files[3] is not None:
            data_arr[-1, -1] = data_files[3][0, 0]
        else:
            assert data_files[0] is not None
            data_arr[-1, -1] = data_files[0][-1, -1]

        return data_arr            
            
    def load_keyed_tile_data(self, x: int, y: int, zoom: ZoomInt, data_files: DataFilesDict) -> bool:
        key = (x, y)
        ydim, xdim = self.get_data_arr_dims(data_files)
        data_arr = np.zeros((ydim+1, xdim+1), dtype=int)

        data_arr[:ydim, :xdim] = data_files[0]
        
        if 1 in data_files and data_files[1] is not None:
            data_arr = self.append_final_row(data_arr, ydim, data_files)
        if 2 in data_files and data_files[2] is not None:
            data_arr = self.append_final_col(data_arr, xdim, data_files)
        if 3 in data_files and data_files[3] is not None:
            data_arr = self.append_final_cell(data_arr, data_files)

        if data_arr[-1,:-1].sum() == 0:
            data_arr[-1,:-1] = data_arr[-2,:-1]
        if data_arr[:-1, -1].sum() == 0:
            data_arr[:-1, -1] = data_arr[:-1, -2]
        if data_arr[-1, -1] == 0:
            data_arr[-1, -1] = data_arr[-2, -2]

        assert isinstance(self.data, dict)
        self.data[zoom][key] = data_arr
            
        return True
    
    def get_elevation(self, lat: CoordVal, lon: CoordVal, elevations: ElevArr | None = None) -> ElevInt | None:
        if elevations is None:
            for zoom in self.zoom_lvls:
                if self.can_load_tile(lat, lon, zoom) or self.try_download:
                    elevations = self.get_data_arr_with_zoom(lat, lon, zoom, self.with_key)
                    bbox = self.get_tile_bbox(lat, lon, zoom)
                    #elevation = get_elevation_from_tile(lat, lon, bbox, elevations)
                    elevation = get_elevation_from_tile_numba(lat, lon, bbox, elevations)
                    break
                else:
                    elevation = None
        else:
            elevation = get_elevation_from_tile_numba(lat, lon, 
                                                      (int(lat), int(lon), int(lat)+1, int(lon)+1), 
                                                      elevations)
                
        return elevation

class BayDataFromTiles(ElevationDataFromTiles):
    def get_path(self) -> str:
        if self.path is None:
            self.path = 'C:/05_Python/by_elev_tiles'
        return self.path
    
    def get_remote_tile_url(self, x: int, y: int, zoom: ZoomInt) -> Url:
        raise NotImplementedError("Cannot download tiles from remote sever on demand", zoom, x, y)

    def load_tile_by_zoom(self, x: int, y: int, zoom: ZoomInt) -> ElevArr:
        subfolders = self.get_tile_subfolders(x, zoom)
        fname = self.get_tile_fname(y)
        gdem_file_path = self.get_fullpath(fname, subfolders)
        elevations = do_ungzip_pkl(gdem_file_path)
        #print(self.get_tile_bbox(lat, lon, zoom))
        assert elevations is not None

        return elevations 

class MapzenDataFromTiles(ElevationDataFromTiles):
    def get_path(self) -> str:
        if self.path is None:
            self.path = 'C:/05_Python/awstiles/terrarium'
        return self.path

    def get_remote_tile_url(self, x: int, y: int, zoom: ZoomInt) -> Url:
        url = f'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{zoom}/{x}/{y}.png'
        return url
    
    def load_tile_by_zoom(self, x: int, y: int, zoom: ZoomInt) -> ElevArr:
        subfolders = self.get_tile_subfolders(x, zoom)
        #print("mapzen - subfolders", subfolders)
        fname = self.get_tile_fname(y)
        #print("mapzen - fname     ", fname)
        gdem_file_path = self.get_fullpath(fname, subfolders)
        #print("mapzen - file_path ", gdem_file_path)
        elevations = np.clip(get_elevation_data(gdem_file_path), a_min=-10, a_max=10000)
        #print("mapzen - elevations", elevations.shape, np.sum(elevations))
        #print(self.get_tile_bbox(lat, lon, zoom))
        assert elevations is not None

        return elevations

class MapterhornDataFromTiles(ElevationDataFromTiles):
    def get_path(self) -> str:
        if self.path is None:
            for p in ['C:/05_Python/awstiles/mapterhorn', '/kaggle/working', '/kaggle/input/mapterhorn', '/kaggle/input']:
                #print("mpth", os.path.exists(p), p)
                if os.path.exists(p):
                    self.path = p
                    break
        if self.path is None:
            raise ValueError("No valid path specified")
        return self.path

    def get_remote_tile_url(self, x: int, y: int, zoom: ZoomInt) -> Url:
        url = f'https://tiles.mapterhorn.com/{zoom}/{x}/{y}.webp'
        return url
    
    def load_tile_by_zoom(self, x: int, y: int, zoom: ZoomInt) -> ElevArr:
        subfolders = self.get_tile_subfolders(x, zoom)
        fname = self.get_tile_fname(y)
        gdem_file_path = self.get_fullpath(fname, subfolders)
        elevations = np.clip(get_elevation_data(gdem_file_path), a_min=-10, a_max=10000)
        #print(self.get_tile_bbox(lat, lon, zoom))
        assert elevations is not None

        return elevations


class AsterGdemData(ElevationDataSource):    
    def __init__(self, readonly: bool = True, filefmt: str | None = None, 
                 res: int = 30, dim: int = 3601, 
                 preprocessor: Callable[[Any], ElevArr] | None = None, 
                 path: str | None = None, with_key: bool = True, 
                 fallbackDemSrc: ElevationDataSource | None = None, 
                 adj_offset: bool = True) -> None:
        self.srctype: str = 'cell'
        self.filefmt = filefmt
        self.res = res
        self.dim = dim
        self.preprocessor = preprocessor
        self.readonly = readonly
        self.path = path
        self.with_key = with_key
        self.adj_offset = adj_offset

        if with_key:
            self.data: dict[KeyedDataKey, ElevArr] | None = {}
        else:
            self.data = None
            
        if self.path is not None:
            self.check_path()

        self.fallbackDemSrc = fallbackDemSrc
        if fallbackDemSrc is None:
            if res==90:
                self.fallbackDemSrc = SrtmDataDict(filefmt='pkl', res=90, dim=dim+1, 
                                                   readonly=False, preload_data=False)
            else:
                self.fallbackDemSrc = CopernicusData(filefmt='tif', res=30, dim=dim-1, 
                                                     readonly=True, 
                                                     fallbackDemSrc=CopernicusData(filefmt='tif', 
                                                                                   res=90, dim=1200, 
                                                                                   readonly=True))
                
    def get_path(self) -> str:
        if self.path is None:
            self.path = 'C:/ASTER_GDEM'
        return self.path

    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        if lat >= 0:
            ns = 'N'
        else:
            ns = 'S'
            lat = np.abs(lat) + 1

        if lon >= 0:
            ew = 'E'
        else:
            ew = 'W'
            lon = np.abs(lon) + 1

        return f"ASTGTMV003_{ns}{int(lat):02d}{ew}{int(lon):03d}_dem"
        
    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr:
        gdem_file_path = self.get_fullpath(self.get_fname(lat, lon), [])
        #elevations = gdal.Open(gdem_file_path).ReadAsArray()
        with rasterio.open(gdem_file_path) as src:
            # Read the data
            elevations = src.read(1)
        
        return elevations

    def get_data_arr(self, lat: CoordVal, lon: CoordVal, with_key: bool = True) -> ElevArr | None:
        overlap = 1
        assert isinstance(self.data, dict)
        assert with_key
        if self.has_data_loaded(lat, lon):
            return self.data[self.get_data_key(lat, lon)]
        elif self.can_load_data(lat, lon):
            dim = cast(int, self.get_dim())
            self.data[self.get_data_key(lat, lon)] = np.zeros((dim + overlap, 
                                                               dim + overlap), 
                                                               dtype=np.int64)
            for getlat, getlon, y, x in [(lat, lon, 0, 0), (lat-1, lon, -1, 0), 
                                         (lat, lon+1, 0, -1), (lat-1, lon+1, -1, -1)]:
                _ = self.load_keyed_data_with_stitch(self.get_data_key(lat, lon), 
                                                     getlat, getlon, x, y)
                
            dem_data = self.data[self.get_data_key(lat, lon)]
            self.data[self.get_data_key(lat, lon)] = np.round(np.mean(np.dstack([
                dem_data[:-1,:-1], dem_data[:-1,1:], dem_data[1:,:-1], dem_data[1:,1:]
            ]), axis=2)).astype(int)
                    
            return self.data[self.get_data_key(lat, lon)]

        return None

    def load_keyed_data(self, key: KeyedDataKey, lat: CoordVal, lon: CoordVal) -> bool:
        return self.load_keyed_data_with_stitch(key, lat, lon)

    def load_keyed_data_with_stitch(self, key: KeyedDataKey, lat: CoordVal, lon: CoordVal, x: int = 0, y: int = 0) -> bool:
        #print("AGD load_keyed_data_with_stitch", self.can_load_data(lat, lon), lat, lon, x, y, "|", key, self.data[key].shape)
        assert isinstance(self.data, dict)

        if self.can_load_data(lat, lon):
            data_file = self.load_file(lat, lon)
            if self.preprocessor is None:
                pass
            else:
                data_file = self.preprocessor(data_file)
            
            dim = cast(int, self.get_dim())
            YDIM, XDIM = self.data[key].shape if key in self.data else (dim, dim)
            ydim, xdim = data_file.shape
            if x == 0 and y == 0:
                if xdim < XDIM:
                    self.data[key] = np.zeros((ydim+1, xdim+1), dtype=np.int64)
                #self.data[key][y:ydim, x:xdim] = data_file
                self.data[key] = data_file
            elif x == 0:
                if xdim > XDIM:
                    self.data[key][y, x:xdim] = downsample_1d_arr(data_file[0, :], XDIM)
                elif xdim < XDIM - 1:
                    self.data[key][y, x:xdim] = upsample_1d_arr(data_file[0, :], XDIM)
                else:
                    self.data[key][y, x:xdim] = data_file[0, :]
            elif y == 0:
                self.data[key][y:ydim, x] = data_file[:, 0]
            else:
                self.data[key][y, x] = data_file[0, 0]

        elif self.fallbackDemSrc is not None and self.fallbackDemSrc.can_load_data(lat, lon):
            assert isinstance(self.fallbackDemSrc, ElevationDataSource)
            assert isinstance(self.fallbackDemSrc.data, dict)

            fbkey = self.fallbackDemSrc.get_data_key(lat, lon)
            if self.fallbackDemSrc.has_data_loaded(lat, lon):
                fbkData = self.fallbackDemSrc.data[fbkey]
            elif isinstance(self.fallbackDemSrc, CopernicusData):
                _ = self.fallbackDemSrc.load_keyed_data_with_overlap(fbkey, lat, lon, 0, 0)
                assert isinstance(self.fallbackDemSrc.data, dict)
                fbkData = self.fallbackDemSrc.data[fbkey]
            else:
                fbkData = self.fallbackDemSrc.load_keyed_data(fbkey, lat, lon) # type: ignore
                
            YDIM, XDIM = self.data[key].shape
            
            assert isinstance(fbkData, np.ndarray)
            ydim, xdim = fbkData.shape
            if x == 0 and y == 0:
                pass
            elif x == 0:
                if xdim > XDIM:
                    self.data[key][y, x:xdim] = downsample_1d_arr(fbkData[0, :], XDIM)
                elif xdim < XDIM - 1:
                    self.data[key][y, x:xdim] = upsample_1d_arr(fbkData[0, :], XDIM)
                else:
                    self.data[key][y, x:xdim] = fbkData[0, :]
            elif y == 0:
                self.data[key][y:ydim, x] = fbkData[:, 0]
            else:
                self.data[key][y, x] = fbkData[0, 0]
            
        else:
            xdim, ydim = self.data[key].shape
            if x == 0 and y== 0:
                pass
            elif x == 0:
                self.data[key][y, :xdim] = self.data[key][y-1, :xdim]
            elif y == 0:
                self.data[key][:ydim, x] = self.data[key][:ydim, x-1]
            else:
                self.data[key][y, x] = self.data[key][ydim-1, xdim-1]
                            
        return True

    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        return []

    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ElevInt | None = None) -> Url:
        raise NotImplementedError("Cannot download Aster GDEM data on demand", lat, lon, zoom)

class SrtmDataDict(ElevationDataSource):
    def __init__(self, readonly: bool = True, filefmt: str | None = None, 
                 res: int = 90, dim: int = 1201, 
                 preprocessor: Callable[[Any], ElevArr] | None = None, 
                 path: str | None = None, preload_data: bool = False) -> None:

        self.srctype = 'cell'
        self.filefmt = filefmt
        self.res = res
        self.path = path
        self.preprocessor = preprocessor
        self.dim = dim
        self.preload_data = preload_data
        self.readonly = readonly
        self.with_key = True
        self.try_download = False
        self.adj_offset = False

        if self.path is not None:
            self.check_path()

        if preload_data:
            try:
                self.data = self.pre_load_file()
            except Exception as e:
                print("Could not preload SRTM DEM data", e)
                self.data = {}
        else:
            self.data = {}

    def pre_load_file(self) -> dict[KeyedDataKey, ElevArr]:
        return cast(dict[KeyedDataKey, ElevArr], do_unpickle(self.get_path()))
    
    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr:
        hgt_file = self.get_fullpath(self.get_fname(lat, lon), [])
        assert isinstance(self.dim, int)
        with open(hgt_file, 'rb') as hgt_data:
            # Each data is 16bit signed integer(i2) - big endian(>)
            elevations = np.fromfile(hgt_data, np.dtype('>i2'), self.dim*self.dim).reshape((self.dim, self.dim))
        
        return elevations

    def get_elevation(self, lat: CoordVal, lon: CoordVal, elevations: ElevArr | None = None) -> ElevInt | None:
        if elevations is None:
            if self.can_load_data(lat, lon):
                elevations = self.get_data_arr(lat, lon, self.with_key)
                #return get_srtm_elevation(lat, lon, elevations=elevations, srtm_data_dict=self.data, DIMS=self.get_dim())
                return get_srtm_elevation_numba(lat, lon, elevations=elevations, 
                                                srtm_data_dict=self.data, DIMS=self.get_dim())   # type: ignore
            else:
                return None
        else:
            #return get_srtm_elevation(lat, lon, elevations=elevations, srtm_data_dict=self.data, DIMS=self.get_dim())
            return get_srtm_elevation_numba(lat, lon, elevations=elevations, 
                                            srtm_data_dict=self.data, DIMS=self.get_dim())     # type: ignore
    
    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        if lat >= 0:
            ns = 'N'
        else:
            ns = 'S'

        if lon >= 0:
            ew = 'E'
        else:
            ew = 'W'
            lon = np.abs(lon) + 1

        return os.path.join("C:/SRTM2", f"{ns}{int(lat):02d}{ew}{int(lon):03d}.hgt")

    def get_path(self) -> str:
        if self.path is None:
            self.path = 'C:/01_AnacondaProjects/bikesite/SRTM_DATA_DICT.pkl'
        return self.path
    
    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        return []

    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ElevInt | None = None) -> Url:
        raise NotImplementedError("Cannot download SRTM data on demand", lat, lon, zoom)

    def can_load_data(self, lat: CoordVal, lon: CoordVal) -> bool:
        return os.path.exists(self.get_fname(lat, lon))
    
    def get_data_arr(self, lat: CoordVal, lon: CoordVal, with_key: bool = True) -> ElevArr | None:
        if with_key and isinstance(self.data, dict):            
            if self.has_data_loaded(lat, lon):
                return self.data[self.get_data_key(lat, lon)]
            elif self.can_load_data(lat, lon):
                done = self.load_keyed_data(self.get_data_key(lat, lon), lat, lon)
                if not self.readonly and done:
                    do_pickle(self.data, self.get_path())
                return self.data[self.get_data_key(lat, lon)]
        else:
            if isinstance(self.data, np.ndarray) and len(self.data) > 0:
                return self.data
            elif self.can_load_data(lat, lon):
                done = self.load_data(lat, lon)
                assert done and isinstance(self.data, np.ndarray)
                if not self.readonly and done:
                    do_pickle(self.data, self.get_path())
                return self.data
            else:
                return None
        
        return None
     
class SrtmRawData(ElevationDataSource):
    def get_path(self) -> str:
        if self.path is None:
            self.path = 'C:/SRTM2'
        return self.path

    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        if lat >= 0:
            ns = 'N'
        else:
            ns = 'S'

        if lon >= 0:
            ew = 'E'
        else:
            ew = 'W'
            lon = np.abs(lon) + 1

        return f"{ns}{int(lat):02d}{ew}{int(lon):03d}"
        
    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr:
        hgt_file = self.get_fullpath(self.get_fname(lat, lon), [])
        with open(hgt_file, 'rb') as hgt_data:
            # Each data is 16bit signed integer(i2) - big endian(>)
            dim = cast(int, self.get_dim())
            elevations = np.fromfile(hgt_data, np.dtype('>i2'), dim*dim).reshape((dim, dim))
        
        return elevations
    
class CopernicusData(ElevationDataSource):
    def __init__(self, readonly: bool = True, filefmt: str | None = None, 
                 res: int = 90, dim: int = 1200, 
                 preprocessor: Callable[[Any], ElevArr] | None = None, 
                 path: str | None = None, with_key: bool = True, 
                 fallbackDemSrc: ElevationDataSource | None = None, 
                 adj_offset: bool = True, try_download: bool = True) -> None:
        self.srctype = 'cell'
        self.filefmt = filefmt
        self.res = res
        self.dim = dim
        self.preprocessor = preprocessor
        self.readonly = readonly
        self.path = path
        self.with_key = with_key
        self.adj_offset = adj_offset
        self.try_download = try_download

        if with_key:
            self.data = {}
        else:
            self.data = None
            
        if self.path is not None:
            self.check_path()

        self.fallbackDemSrc = fallbackDemSrc
        if fallbackDemSrc is None:
            if res==90:
                self.fallbackDemSrc = SrtmDataDict(filefmt='pkl', res=90, dim=dim+1, 
                                                   readonly=False, preload_data=False)
            else:
                self.fallbackDemSrc = AsterGdemData(filefmt='tif', res=30, dim=dim+1, 
                                                    readonly=True)

    def get_dim(self) -> int:
        assert isinstance(self.dim, int)
        return self.dim

    def get_path(self) -> str:
        if self.path is None:
            self.path = f'C:/05_Python/awstiles/copernicus/{self.get_res()}'
        return self.path
    
    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> Url:
        base_url = f'https://copernicus-dem-{self.res}m.s3.amazonaws.com/'
        file_path_comp = self.get_fname(lat, lon)
        
        return base_url + file_path_comp + "/" + file_path_comp + ".tif"
    
    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        res_code = self.get_res() // 3
        if lat >= 0:
            ns = 'N'
        else:
            ns = 'S'
            raise NotImplementedError("No data for Southern Hemisphere available", lat, lon)

        if lon >= 0:
            ew = 'E'
        else:
            ew = 'W'
            lon = np.abs(lon) + 1 if lon != int(lon) else np.abs(lon)
        
        return f"Copernicus_DSM_COG_{res_code}_{ns}{int(lat):02d}_00_{ew}{int(lon):03d}_00_DEM"  # {self.filefmt}

    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr:
        fpath = self.get_fullpath(self.get_fname(lat, lon), [])
        with rasterio.open(fpath) as src:
            # Read the data
            dem_dataset = src.read(1)
        return dem_dataset
    
    def get_elevation(self, lat: CoordVal, lon: CoordVal, elevations: ElevArr | None = None) -> ElevInt | None:
        if elevations is None:
            if self.can_load_data(lat, lon) or self.try_download:
                elevations = self.get_data_arr(lat, lon, self.with_key)
                if elevations is None:
                    return 0
                #return get_srtm_elevation(lat, lon, elevations=elevations, DIMS="auto")
                return get_srtm_elevation_numba(lat, lon, elevations=elevations, DIMS="auto")
            else:
                return None
        else:
            #return get_srtm_elevation(lat, lon, elevations=elevations, DIMS="auto")
            return get_srtm_elevation_numba(lat, lon, elevations=elevations, DIMS="auto")
    
    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        return []

    def get_data_arr(self, lat: CoordVal, lon: CoordVal, with_key: bool = True) -> ElevArr | None:
        overlap = 1 if not self.adj_offset else 2
        if self.has_data_loaded(lat, lon):
            if isinstance(self.data, dict):
                return self.data[self.get_data_key(lat, lon)]
            else:
                return None
        elif isinstance(self.data, dict) and (self.can_load_data(lat, lon) or self.try_download):
            self.data[self.get_data_key(lat, lon)] = np.zeros((self.get_dim()+overlap, 
                                                               self.get_dim()+overlap), 
                                                               dtype=np.int64)
            for getlat, getlon, y, x in [(lat, lon, 0, 0), (lat-1, lon, -1, 0), 
                                         (lat, lon+1, 0, -1), (lat-1, lon+1, -1, -1)]:
                if not self.can_load_data(getlat, getlon) and self.try_download:
                    url = self.get_remote_url(getlat, getlon)
                    fpath = self.get_fullpath(self.get_fname(getlat, getlon), 
                                              self.get_subfolders(getlat, getlon))
                    file_content = download_remote_file(url)
                    file_ok = write_file(file_content, fpath)
                    if file_ok:
                        _ = self.load_keyed_data_with_overlap(self.get_data_key(lat, lon), 
                                                              getlat, getlon, x, y, overlap)
                    elif x == 0 and y == 0:
                        return None
                    else:
                        continue
                elif self.can_load_data(getlat, getlon):
                    _ = self.load_keyed_data_with_overlap(self.get_data_key(lat, lon), 
                                                          getlat, getlon, x, y, overlap)
                else:
                    continue
                
            if overlap == 2:
                dem_data = self.data[self.get_data_key(lat, lon)]
                self.data[self.get_data_key(lat, lon)] = np.round(np.mean(np.dstack([
                    dem_data[:-1,:-1], dem_data[:-1,1:], dem_data[1:,:-1], dem_data[1:,1:]
                ]), axis=2)).astype(int)
                    
            return self.data[self.get_data_key(lat, lon)]
        
        elif self.fallbackDemSrc is not None and self.fallbackDemSrc.can_load_data(lat, lon):
            #fbkey = self.fallbackDemSrc.get_data_key(lat, lon)
            print("from fbkData", lat, lon, self.get_data_key(lat, lon))
            fallbackDataArr = self.fallbackDemSrc.get_data_arr(lat, lon)
            assert isinstance(self.data, dict) and isinstance(fallbackDataArr, np.ndarray)
            self.data[self.get_data_key(lat, lon)] = fallbackDataArr
            return self.data[self.get_data_key(lat, lon)]
        else:
            return None

    def load_keyed_data(self, key: KeyedDataKey, lat: CoordVal, lon: CoordVal) -> bool:
        raise NotImplementedError("Function load_keyed_data is not implemented. Use load_keyed_data_with_overlap() instead", 
                                  key, lat, lon)

    def load_keyed_data_with_overlap(self, key: KeyedDataKey, lat: CoordVal, lon: CoordVal, 
                                     x: int, y: int, overlap: int = 1) -> bool:
        #print("load_keyed_data_with_overlap", self.can_load_data(lat, lon), lat, lon, x, y, overlap, "|", key, self.data[key].shape)
        assert isinstance(self.data, dict)
        if self.can_load_data(lat, lon):
            data_file = self.load_file(lat, lon)
            if self.preprocessor is None:
                pass
            else:
                data_file = self.preprocessor(data_file)
            
            add_to_dim = 1 if isinstance(self, CopernicusData) else 0
            YDIM, XDIM = self.data[key].shape if key in self.data else (self.get_dim() + add_to_dim, 
                                                                        self.get_dim() + add_to_dim)
            ydim, xdim = data_file.shape
            if x == 0 and y == 0:
                if xdim < XDIM:
                    self.data[key] = np.zeros((ydim + overlap, xdim + overlap), 
                                              dtype=np.int64)
                self.data[key][y:ydim, x:xdim] = data_file
            elif x == 0:
                if xdim > XDIM:
                    #print("xdim, XDIM data_file", data_file.shape)
                    #print("xdim, XDIM", xdim, XDIM, "|", lat, lon, x, y, overlap)
                    self.data[key][-overlap:, x:xdim] = np.vstack([downsample_1d_arr(data_file[row, :], XDIM) \
                                                                     for row in range(overlap)])
                elif xdim < XDIM - overlap:
                    self.data[key][-overlap:, x:xdim] = np.vstack([upsample_1d_arr(data_file[row, :], XDIM) \
                                                                     for row in range(overlap)])
                else:
                    self.data[key][-overlap:, x:xdim] = data_file[0:overlap, :]
            elif y == 0:
                #self.data[key][y:ydim, x:x+overlap] = data_file[:, 0:overlap]
                self.data[key][y:ydim, -overlap:] = data_file[:, 0:overlap]
            else:
                #self.data[key][y, x:x+overlap] = data_file[0, 0:overlap]
                self.data[key][-overlap:, -overlap:] = data_file[0, 0]

        elif self.fallbackDemSrc is not None and self.fallbackDemSrc.can_load_data(lat, lon):
            fbkey = self.fallbackDemSrc.get_data_key(lat, lon)
            print("fbkData", lat, lon, fbkey, x, y, overlap)
            if self.fallbackDemSrc.has_data_loaded(lat, lon):
                assert isinstance(self.fallbackDemSrc.data, dict)
                fbkData = self.fallbackDemSrc.data[fbkey]
            elif isinstance(self.fallbackDemSrc, CopernicusData):
                _ = self.fallbackDemSrc.load_keyed_data_with_overlap(fbkey, lat, lon, 0, 0, overlap)
                assert isinstance(self.fallbackDemSrc.data, dict)
                fbkData = self.fallbackDemSrc.data[fbkey]
            elif isinstance(self.fallbackDemSrc, AsterGdemData):
                _ = self.fallbackDemSrc.load_keyed_data_with_stitch(fbkey, lat, lon, 0, 0)
                assert isinstance(self.fallbackDemSrc.data, dict)
                fbkData = self.fallbackDemSrc.data[fbkey]
            else:
                _ = self.fallbackDemSrc.load_keyed_data(fbkey, lat, lon)
                assert isinstance(self.fallbackDemSrc.data, dict)
                fbkData = self.fallbackDemSrc.data[fbkey]
                
            YDIM, XDIM = self.data[key].shape
            ydim, xdim = fbkData.shape

            if x == 0 and y == 0:
                pass
            elif x == 0:
                if xdim > XDIM:
                    #self.data[key][y:y+overlap, x:xdim] = np.vstack([downsample_1d_arr(fbkData[row, :], XDIM) \
                    #                                                 for row in range(overlap)])
                    self.data[key][y*overlap:, x:xdim] = np.vstack([downsample_1d_arr(fbkData[row, :], XDIM) \
                                                                     for row in range(overlap)])
                elif xdim < XDIM - 1:
                    #self.data[key][y:y+overlap, x:xdim] = np.vstack([upsample_1d_arr(fbkData[row, :], XDIM) \
                    #                                                 for row in range(overlap)])
                    self.data[key][y*overlap:, x:xdim+overlap] = np.vstack([upsample_1d_arr(fbkData[row, :], XDIM) \
                                                                     for row in range(overlap)])
                else:
                    #print("fbk 783-1", x, xdim, y, ydim, overlap)
                    #print(f"fbk 783-2     [{y}:{y+overlap}, {x}:{xdim}], [0:{overlap}, :]         ", 
                    #      fbkData[0:overlap, :].shape)
                    #self.data[key][y:y+overlap, x:xdim] = fbkData[0:overlap, :]
                    self.data[key][y*overlap:, x:xdim] = fbkData[0:overlap, :]
            elif y == 0:
                self.data[key][y:ydim, x*overlap:] = fbkData[:, 0:overlap]
            else:
                #self.data[key][y:y+overlap, x:x+overlap] = fbkData[0:overlap, 0:overlap]
                self.data[key][-overlap:, -overlap:] = fbkData[0:overlap, 0:overlap]
            
        else:
            print("No fallback available:", lat, lon, " | ", key, x, y, overlap)

            add_to_dim = 1 if isinstance(self, CopernicusData) else 0
            YDIM, XDIM = (self.get_dim() + add_to_dim, self.get_dim() + add_to_dim)
            fbkData = self.data[key]
            ydim, xdim = fbkData.shape

            #print("No fallback available:", lat, lon, fbkey, x, y, overlap)
            #data_file = self.load_file(lat, lon)
            #if self.preprocessor is None:
            #    pass
            #else:
            #    data_file = self.preprocessor(data_file)

            #print("No fallback available:", key, lat, lon, " | ", x, y, overlap)
            print("                      ", key, xdim, ydim, " | ", XDIM, YDIM, " | ", self.get_dim())
            if x == 0 and y== 0:
                pass
            elif x == 0:
                #print("   X == 0 - DATA", self.data[key][-6:, -6:])
                for i in range(overlap):
                    #print("   X == 0:      [i+ydim, :xdim] ", i, f"[{i+ydim}, :{xdim}]    ", i+ydim-1)
                    self.data[key][-overlap+i, :xdim] = self.data[key][-overlap-1, :xdim]
            elif y == 0:
                #print("   Y == 0 - DATA", self.data[key][-6:, -6:])
                for i in range(overlap):
                    #print("   Y == 0:      [:ydim, -overlap+i] ", i, f"[:{ydim}, {-overlap+i}]    ", -overlap-1)
                    self.data[key][:ydim, -overlap+i] = self.data[key][:ydim, -overlap-1]
            else:
                #self.data[key][y, x] = self.data[key][ydim-1, xdim-1]
                #print("   ELSE - DATA", self.data[key][-6:, -6:])
                #print("   ELSE:      [y-overlap+1:, x-overlap+1:]   --- ",  f"[:{y-overlap+1}, {x-overlap+1:}]    ", 
                #      y-overlap, x-overlap)
                self.data[key][y-overlap+1:, x-overlap+1:] = self.data[key][y-overlap, x-overlap]

        return True



class SuperDem(ElevationDataSource):    
    def get_path(self) -> str:
        if self.path is None:
            self.path = 'C:/SUPER_DEM'
        return self.path

    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        if lat >= 0:
            ns = 'N'
        else:
            ns = 'S'
            raise NotImplementedError("No data for Southern Hemisphere available", lat, lon)

        if lon >= 0:
            ew = 'E'
        else:
            ew = 'W'
            lon = np.abs(lon) + 1

        #return f"SUPER_DEM_{ns}{int(lat):02d}_{ew}{int(lon):03d}"
        return f"SUPERDEM_{ns}{int(lat):02d}{ew}{int(lon):03d}"
        
    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr | None:
        gdem_file_path = self.get_fullpath(self.get_fname(lat, lon), [])
    
        if gdem_file_path.endswith(".npz"):
            elevations = np.load(gdem_file_path)['arr_0']
        elif gdem_file_path.endswith(".gzip"):
            elevations = do_ungzip_pkl(gdem_file_path)
        elif gdem_file_path.endswith(".tif"):
        #    elevations = gdal.Open(gdem_file_path).ReadAsArray()
            with rasterio.open(gdem_file_path) as src:
                # Read the data
                elevations = src.read(1)
        else:
            return None
        
        return elevations

    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> Url:
        raise NotImplementedError("Cannot download data on demand", lat, lon, zoom)
        
    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        return []


class SonnyDtm(ElevationDataSource):    
    def get_path(self) -> str:
        if self.path is None:
            self.path = '/kaggle/input/elevtiles-splitsonny'
        return self.path

    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        if lat >= 0:
            ns = 'N'
        else:
            ns = 'S'
            raise NotImplementedError("No data for Southern Hemisphere available", lat, lon)

        if lon >= 0:
            ew = 'E'
        else:
            ew = 'W'
            lon = np.abs(lon) + 1

        return f"tile_{ns}{int(lat):02d}_{ew}{int(lon):03d}"
        
    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr | None:
        gdem_file_path = self.get_fullpath(self.get_fname(lat, lon), [])
    
        elevations = np.load(gdem_file_path)['arr_0']
        
        return elevations

    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> Url:
        raise NotImplementedError("Cannot download data on demand", lat, lon, zoom)
        
    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        return []


class GeDtm30(ElevationDataSource):    
    def get_dim(self) -> int:
        return 4001

    def get_path(self) -> str:
        if self.path is None:
            self.path = '/kaggle/input/elev-gedtm30-cog'
        return self.path

    def get_fname(self, lat: CoordVal, lon: CoordVal) -> str:
        if lat >= 0:
            ns = 'N'
        else:
            ns = 'S'
            raise NotImplementedError("No data for Southern Hemisphere available", lat, lon)

        if lon >= 0:
            ew = 'E'
        else:
            ew = 'W'
            lon = np.abs(lon) + 1

        return f"GEDTM30_{ns}{int(lat):02d}_{ew}{int(lon):03d}"
        
    def load_file(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> ElevArr:
        gdem_file_path = self.get_fullpath(self.get_fname(lat, lon), [])
    
        elevations = np.load(gdem_file_path)['elevation']
        return elevations
    
    def get_remote_url(self, lat: CoordVal, lon: CoordVal, zoom: ZoomInt | None = None) -> Url:
        raise NotImplementedError("Cannot download data on demand", lat, lon, zoom)
        
    def get_subfolders(self, lat: CoordVal, lon: CoordVal) -> list[str]:
        return []
