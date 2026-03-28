# type: ignore
import unittest
import math
import numpy as np
from math import cos, radians, sqrt
from shapely.geometry import Point, LineString, MultiLineString, Polygon


# python -m unittest test_distance_helper.py

from geom_helpers.distance_helper import (get_dist, get_dist_circle_coords, get_dist_point_to_line, 
                             convert_lonlat_to_m, get_dist_from_linestring, 
                             get_dist_from_multilinestring)


class TestGetDistCircleCoords(unittest.TestCase):
    def test_get_dist_circle_coords(self):
        lat, lon = (48, 12)
        D = 111320
        dD = 0.00001 * D
        coords = get_dist_circle_coords(D, lat, lon, num_points=9)
        self.assertTrue(isinstance(coords, list), 
                        "ERROR - wrong type returned data - should be a list")
        self.assertTrue(len(coords) == 9, 
                        "ERROR - invalid length of returned list")
        self.assertTrue(coords[2][1] == lat+1, 
                        "ERROR - invalid lat at pos 2")
        self.assertTrue(coords[4][1] == lat, 
                        "ERROR - invalid lat at pos 4")
        self.assertTrue(coords[6][1] == lat-1, 
                        "ERROR - invalid lat at pos 6")
        self.assertTrue(coords[8][1] == lat, 
                        "ERROR - invalid lat at pos 8")
        for i in range(len(coords)):
            d = get_dist((lat, lon), coords[i][::-1])
            self.assertAlmostEqual(d, D, delta = dD,
                                   msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        

    def test_get_dist_circle_coords2(self):
        lat, lon = (42, 12)
        num_points = 9
        D = 111320
        dD = 0.00001 * D
        coords = get_dist_circle_coords(D, lat, lon, num_points=num_points)
        self.assertTrue(len(coords) == 9, 
                        "ERROR - invalid length of returned list")
        self.assertTrue(coords[int((num_points-1)/4)][1] == lat+D/111320, 
                        f"ERROR - invalid lat at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[2*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {2*int((num_points-1)/4)}")
        self.assertAlmostEqual(coords[2*int((num_points-1)/4)][0], 
                               lon-D/111320*1/cos(radians(lat)), 5, 
                               f"ERROR - invalid lon at pos {2*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][1] == lat-D/111320, 
                        f"ERROR - invalid lat at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[4*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {4*int((num_points-1)/4)}")
        self.assertAlmostEqual(coords[4*int((num_points-1)/4)][0], 
                               lon+D/111320*1/cos(radians(lat)), 5, 
                               f"ERROR - invalid lon at pos {4*int((num_points-1)/4)}")
        for i in range(len(coords)):
            d = get_dist((lat, lon), coords[i][::-1])
            #print(i, coords[i])
            #print(d)
            self.assertAlmostEqual(d, D, delta = dD,
                                   msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        

    def test_get_dist_circle_coords3(self):
        lat, lon = (42, 12)
        num_points = 9
        n = 3
        D = n * 111320
        dD = 0.00001 * n * D
        coords = get_dist_circle_coords(D, lat, lon, num_points=9)

        for i in range(len(coords)):
            d = get_dist((lat, lon), coords[i][::-1])
            #print(i, coords[i])
            #print(d)
            self.assertAlmostEqual(d, D, delta = dD,
                                   msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        

        self.assertTrue(len(coords) == 9, 
                        "ERROR - invalid length of returned list")
        self.assertTrue(coords[int((num_points-1)/4)][1] == lat+D/111320, 
                        f"ERROR - invalid lat at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[2*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {2*int((num_points-1)/4)}")
        self.assertAlmostEqual(coords[2*int((num_points-1)/4)][0], 
                               lon-D/111320*1/cos(radians(lat)), 5, 
                               f"ERROR - invalid lon at pos {2*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][1] == lat-D/111320, 
                        f"ERROR - invalid lat at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[4*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {4*int((num_points-1)/4)}")
        self.assertAlmostEqual(coords[4*int((num_points-1)/4)][0], 
                               lon+D/111320*1/cos(radians(lat)), 5, 
                               f"ERROR - invalid lon at pos {4*int((num_points-1)/4)}")

    def test_get_dist_circle_coords4(self):
        lat, lon = (42, 12)
        n = 0.1
        num_points = 9
        D = n * 111320
        dD = 0.00001 * D
        coords = get_dist_circle_coords(D, lat, lon, num_points=num_points)
        self.assertTrue(len(coords) == 9, 
                        "ERROR - invalid length of returned list")
        self.assertTrue(coords[int((num_points-1)/4)][1] == lat+D/111320, 
                        f"ERROR - invalid lat at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[2*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {2*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][1] == lat-D/111320, 
                        f"ERROR - invalid lat at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[4*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {4*int((num_points-1)/4)}")
        for i in range(len(coords)):
            d = get_dist((lat, lon), coords[i][::-1])
            #print(i, coords[i])
            #print(d)
            self.assertAlmostEqual(d, D, delta = dD,
                                   msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        


    def test_get_dist_circle_coords5(self):
        lat, lon = (50, 12)
        n = 0.1
        D = n * 1000
        dD = 0.00001 * D
        num_points = 13
        coords = get_dist_circle_coords(D, lat, lon, num_points=num_points)
        
        for i in range(len(coords)):
            d = get_dist((lat, lon), coords[i][::-1])
            #print(i, coords[i])
            #print(d)
            self.assertAlmostEqual(d, D, delta = dD,
                                   msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        
        
        self.assertTrue(len(coords) == num_points, 
                        "ERROR - invalid length of returned list")
        self.assertTrue(coords[int((num_points-1)/4)][1] == lat+D/111320, 
                        f"ERROR - invalid lat at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[2*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {2*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][1] == lat-D/111320, 
                        f"ERROR - invalid lat at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[4*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {4*int((num_points-1)/4)}")


    def test_get_dist_circle_coords6(self):
        lat, lon = (50, 12)
        n = 0.1
        D = n * 200
        dD = 0.00001 * D
        num_points = 21
        coords = get_dist_circle_coords(D, lat, lon, num_points=num_points)
        
        for i in range(len(coords)):
            d = get_dist((lat, lon), coords[i][::-1])
            #print(i, coords[i])
            #print(d)
            self.assertAlmostEqual(d, D, delta = dD,
                                   msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        
        
        self.assertTrue(len(coords) == num_points, 
                        "ERROR - invalid length of returned list")
        self.assertTrue(coords[int((num_points-1)/4)][1] == lat+D/111320, 
                        f"ERROR - invalid lat at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {int((num_points-1)/4)}")
        self.assertTrue(coords[2*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {2*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][1] == lat-D/111320, 
                        f"ERROR - invalid lat at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[3*int((num_points-1)/4)][0] == lon, 
                        f"ERROR - invalid lon at pos {3*int((num_points-1)/4)}")
        self.assertTrue(coords[4*int((num_points-1)/4)][1] == lat, 
                        f"ERROR - invalid lat at pos {4*int((num_points-1)/4)}")

    def test_get_dist_circle_coords7(self):
        for num_points in [17, 37, 101, 121]:
            lat, lon = (66, 12)
            n = 0.1
            D = n * 100000
            dD = 0.00001 * D
            coords = get_dist_circle_coords(D, lat, lon, num_points=num_points)
            
            for i in range(len(coords)):
                d = get_dist((lat, lon), coords[i][::-1])
                #print(i, coords[i])
                #print(d)
                self.assertAlmostEqual(d, D, delta = dD,
                                    msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        
            
            self.assertTrue(len(coords) == num_points, 
                            "ERROR - invalid length of returned list")
            self.assertTrue(coords[int((num_points-1)/4)][1] == lat+D/111320, 
                            f"ERROR - invalid lat at pos {int((num_points-1)/4)}")
            self.assertTrue(coords[int((num_points-1)/4)][0] == lon, 
                            f"ERROR - invalid lon at pos {int((num_points-1)/4)}")
            self.assertTrue(coords[2*int((num_points-1)/4)][1] == lat, 
                            f"ERROR - invalid lat at pos {2*int((num_points-1)/4)}")
            self.assertTrue(coords[3*int((num_points-1)/4)][1] == lat-D/111320, 
                            f"ERROR - invalid lat at pos {3*int((num_points-1)/4)}")
            self.assertTrue(coords[3*int((num_points-1)/4)][0] == lon, 
                            f"ERROR - invalid lon at pos {3*int((num_points-1)/4)}")
            self.assertTrue(coords[4*int((num_points-1)/4)][1] == lat, 
                            f"ERROR - invalid lat at pos {4*int((num_points-1)/4)}")

    def test_get_dist_circle_coords8(self):
        for num_points in [17, 37, 101, 121]:
            lat, lon = (66, 12)
            n = 0.1
            D = n * 100000
            dD = 0.00001 * D
            coords = get_dist_circle_coords(D, lat, lon, num_points=num_points, geo_json=False)
            
            for i in range(len(coords)):
                d = get_dist((lat, lon), coords[i])
                #print(i, coords[i], d)
                self.assertAlmostEqual(d, D, delta = dD,
                                    msg=f"ERROR - invalid distance of coord {i}: {d} vs {D}")        
            
            self.assertTrue(len(coords) == num_points, 
                            "ERROR - invalid length of returned list")
            self.assertTrue(coords[int((num_points-1)/4)][0] == lat+D/111320, 
                            f"ERROR - invalid lat at pos {int((num_points-1)/4)}")
            self.assertTrue(coords[int((num_points-1)/4)][1] == lon, 
                            f"ERROR - invalid lon at pos {int((num_points-1)/4)}")
            self.assertTrue(coords[2*int((num_points-1)/4)][0] == lat, 
                            f"ERROR - invalid lat at pos {2*int((num_points-1)/4)}")
            self.assertTrue(coords[3*int((num_points-1)/4)][0] == lat-D/111320, 
                            f"ERROR - invalid lat at pos {3*int((num_points-1)/4)}")
            self.assertTrue(coords[3*int((num_points-1)/4)][1] == lon, 
                            f"ERROR - invalid lon at pos {3*int((num_points-1)/4)}")
            self.assertTrue(coords[4*int((num_points-1)/4)][0] == lat, 
                            f"ERROR - invalid lat at pos {4*int((num_points-1)/4)}")


class TestGetDist(unittest.TestCase):
    def setUp(self):
        self.M = 111320  # Approx. meters per degree latitude
    
    def test_get_dist(self):
        self.assertAlmostEqual(get_dist((0, 0), (1, 0)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0), (3, 0)), 3*self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0), (0, 1)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((1, 0), (0, 1)), 
                               math.sqrt(self.M ** 2 + (self.M * math.cos(math.radians(0.5))) ** 2), 
                               msg = f"ERR: {(self.M * math.cos(math.radians(0.5))) ** 2}", delta=1)
        self.assertAlmostEqual(get_dist((0, 45), (0, 46)), self.M)
        self.assertAlmostEqual(get_dist((41, 10), (40, 11)), 
                               math.sqrt(self.M ** 2 + (self.M * math.cos(math.radians(40.5))) ** 2), 
                               msg = f"ERR: {(self.M * math.cos(math.radians(40.5))) ** 2}", delta=1)
        self.assertAlmostEqual(get_dist((1, -1), (0, 1)), 
                               math.sqrt(self.M ** 2 + (2 * self.M * math.cos(math.radians(0.5))) ** 2), 
                               msg = f"ERR: {(self.M * math.cos(math.radians(0.5))) ** 2}", delta=1)

    def test_get_dist_with_elev(self):
        self.assertAlmostEqual(get_dist((0, 0, 10), (0, 0, 0)), 10, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 0), (1, 0, 10)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 0), (1, 0, 100)), self.M, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 0), (1, 0, 1000)), self.M + 5, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 10000), (1, 0, 1000)), self.M + 363, delta=1)
        self.assertAlmostEqual(get_dist((0, 0, 10000), (1, 0, 0)), self.M + 448, delta=1)

class TestGetDistPointToLine(unittest.TestCase):
    def setUp(self):
        self.M = 111320  # Approx. meters per degree latitude
    
    def test_get_dist_point_to_line(self):
        lat, lon = (48, 12)
        pt = Point(lon, lat)
        line1 = LineString([[lon-1,lat-1], [lon+1,lat-1]])
        d1 = get_dist_point_to_line(pt, line1)
        
        
        line2 = LineString([[lon-1,lat+0.5], [lon+1,lat+0.5]])
        d2 = get_dist_point_to_line(pt, line2)

        line3 = LineString([[lon-1,lat+1], [lon, lat], [lon+1,lat-1]])
        d3 = get_dist_point_to_line(pt, line3)

        line4 = LineString([[lon-1,lat+1], [lon-1, lat-1]])
        d4 = get_dist_point_to_line(pt, line4)

        line5 = LineString([[lon+0.2,lat-1], [lon+0.2, lat+1]])
        d5 = get_dist_point_to_line(pt, line5)

        line6 = LineString([[lon+0.2,lat-1], [lon+0.3, lat+1]])
        d6 = get_dist_point_to_line(pt, line6)

        line7 = LineString([[lon+0.2,lat-1], [lon+0.22,lat-0.6], [lon+0.24,lat-0.2], 
                            [lon+0.25,lat], [lon+0.26,lat+0.1], [lon+0.3, lat+1]])
        d7 = get_dist_point_to_line(pt, line7)

        mls1 = MultiLineString([line1, line2])
        mls2 = MultiLineString([line1, line2, line3])
        mls3 = MultiLineString([line1, line2, LineString([[lon-1,lat+1], [lon+1,lat-1]])])
        d8 = get_dist_point_to_line(pt, mls1)
        d9 = get_dist_point_to_line(pt, mls2)
        d10 = get_dist_point_to_line(pt, mls3)

        # Satisfy type-checker - make sure d1 is not None
        assert d1
        assert d2
        assert d4
        assert d5
        assert d6
        assert d7

        self.assertAlmostEqual(d1, self.M, delta=1, msg=f"Invalid dist d1: {d1:.2f}")
        self.assertAlmostEqual(d2, self.M/2, delta=1, msg=f"Invalid dist d2: {d2:.2f}")
        self.assertTrue(d3 == 0, f"Invalid dist d3: {d3:.2f}")
        self.assertAlmostEqual(d4, self.M*cos(radians(lat)), delta=d4/200, msg=f"Invalid dist d4: {d4:.2f}")
        self.assertAlmostEqual(d5, self.M/5*cos(radians(lat)), delta=d5/200, msg=f"Invalid dist d5: {d5:.2f}")
        self.assertAlmostEqual(d6, self.M/4*cos(radians(lat)), delta=d6/200, msg=f"Invalid dist d6: {d6:.2f}")
        self.assertAlmostEqual(d7, self.M/4*cos(radians(lat)), delta=d7/200, msg=f"Invalid dist d7: {d6:.2f}")

        assert d8
        assert d10
        self.assertAlmostEqual(d8, self.M/2, delta=1, msg=f"Invalid dist d8: {d8:.2f}")
        self.assertTrue(d9 == 0, f"Invalid dist d9: {d9:.2f}")
        self.assertAlmostEqual(d10, 1200, delta=1, msg=f"Invalid dist d10: {d10:.2f}")

        self.assertIsNone(get_dist_point_to_line(pt, Point(13,44)), "Error: Must be None #1")
        poly = Polygon([[lon-1, lat-1], [lon-1, lat+1], [lon+1, lat+1], [lon+1, lat-1], [lon-1, lat-1]])
        self.assertIsNone(get_dist_point_to_line(pt, poly), "Error: Must be None #2")

class TestGetDistPointToLine2(unittest.TestCase):
    def setUp(self):
        self.M = 111320  # Approx. meters per degree latitude
    
    def test_get_dist_point_to_line(self):
        lat, lon = (58, -2)
        pt = Point(lon, lat)
        line1 = LineString([[lon-1,lat-1], [lon+1,lat-1]])
        d1 = get_dist_point_to_line(pt, line1)
        
        line2 = LineString([[lon-1,lat+0.5], [lon+1,lat+0.5]])
        d2 = get_dist_point_to_line(pt, line2)

        line3 = LineString([[lon-1,lat+1], [lon, lat], [lon+1,lat-1]])
        d3 = get_dist_point_to_line(pt, line3)

        line4 = LineString([[lon-1,lat+1], [lon-1, lat-1]])
        d4 = get_dist_point_to_line(pt, line4)

        line5 = LineString([[lon+0.2,lat-1], [lon+0.2, lat+1]])
        d5 = get_dist_point_to_line(pt, line5)

        line6 = LineString([[lon+0.2,lat-1], [lon+0.3, lat+1]])
        d6 = get_dist_point_to_line(pt, line6)

        line7 = LineString([[lon+0.2,lat-1], [lon+0.22,lat-0.6], [lon+0.24,lat-0.2], 
                            [lon+0.25,lat], [lon+0.26,lat+0.1], [lon+0.3, lat+1]])
        d7 = get_dist_point_to_line(pt, line7)

        mls1 = MultiLineString([line1, line2])
        mls2 = MultiLineString([line1, line2, line3])
        mls3 = MultiLineString([line1, line2, LineString([[lon-1,lat+1], [lon+1,lat-1]])])
        d8 = get_dist_point_to_line(pt, mls1)
        d9 = get_dist_point_to_line(pt, mls2)
        d10 = get_dist_point_to_line(pt, mls3)

        # Satisfy type-checker - make sure d1 is not None
        assert d1
        assert d2
        assert d4
        assert d5
        assert d6
        assert d7
        assert d8
        assert d10

        self.assertAlmostEqual(d1, self.M, delta=1, msg=f"Invalid dist d1: {d1:.2f}")
        self.assertAlmostEqual(d2, self.M/2, delta=1, msg=f"Invalid dist d2: {d2:.2f}")
        self.assertTrue(d3 == 0, f"Invalid dist d3: {d3:.2f}")
        self.assertAlmostEqual(d4, self.M*cos(radians(lat)), delta=d4/200, msg=f"Invalid dist d4: {d4:.2f}")
        self.assertAlmostEqual(d5, self.M/5*cos(radians(lat)), delta=d5/200, msg=f"Invalid dist d5: {d5:.2f}")
        self.assertAlmostEqual(d6, self.M/4*cos(radians(lat)), delta=d6/150, msg=f"Invalid dist d6: {d6:.2f}")
        self.assertAlmostEqual(d7, self.M/4*cos(radians(lat)), delta=d7/200, msg=f"Invalid dist d7: {d7:.2f}")

        self.assertAlmostEqual(d8, self.M/2, delta=1, msg=f"Invalid dist d8: {d8:.2f}")
        self.assertTrue(d9 == 0, f"Invalid dist d9: {d9:.2f}")
        self.assertAlmostEqual(d10, 1455, delta=1, msg=f"Invalid dist d10: {d10:.2f}")

        self.assertIsNone(get_dist_point_to_line(pt, Point(13,44)), "Error: Must be None #1")
        poly = Polygon([[lon-1, lat-1], [lon-1, lat+1], [lon+1, lat+1], [lon+1, lat-1], [lon-1, lat-1]])
        self.assertIsNone(get_dist_point_to_line(pt, poly), "Error: Must be None #2")


class TestConvertLonlatToM(unittest.TestCase):
    def test_convert_lonlat_to_m(self):
        self.M = 111320  # Approx. meters per degree latitude
        res = convert_lonlat_to_m(47,12,47,12) 
        print("res", res)
        self.assertTrue(res == (0, 0), f"Error - convert_lonlat_to_m - #1: {str(res)}")
        
        res = convert_lonlat_to_m(12,47.1,12,47)
        self.assertTrue(res == (0, 11132), f"Error - convert_lonlat_to_m - #2: {str(res)}")
        
        res = convert_lonlat_to_m(12,47.5,12,47)
        self.assertTrue(res == (0, 55500 + 160), f"Error - convert_lonlat_to_m - #3: {str(res)}")

        res = convert_lonlat_to_m(12,46.5,12,47)
        self.assertTrue(res == (0, -(55500 + 160)), f"Error - convert_lonlat_to_m - #3: {str(res)}")

        res = convert_lonlat_to_m(12.8,46.5,12,47)
        #self.assertTrue(res == [np.float32(0.8 * self.M * cos(radians(0.5*(46.5+47)))), -(55500 + 160)], 
        self.assertTrue(res == (np.float32(0.8 * self.M * cos(radians(46.5))), -(55500 + 160)), 
                        f"Error - convert_lonlat_to_m - #4: {str(res)}")

        res = convert_lonlat_to_m(11.8,46.9,12,47)
        self.assertAlmostEqual(res[0], np.float32(-0.2 * self.M * cos(radians(46.9))), 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #5.1: {str(res)}")
        self.assertAlmostEqual(res[1], -11132, 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #5.2: {str(res)}")

        res = convert_lonlat_to_m(-1.8,46.9,-2,47)
        #print("expected #6", [np.float32(0.2 * self.M * cos(radians(0.5*(46.9+47)))), -11132])
        self.assertAlmostEqual(res[0], np.float32(0.2 * self.M * cos(radians(46.9))), 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #6.1: {str(res)}")
        self.assertAlmostEqual(res[1], -11132, 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #6.2: {str(res)}")

        res = convert_lonlat_to_m(-2.25,46.9,-2,47)
        #print("expected #7", [np.float32(-0.25 * self.M * cos(radians(0.5*(46.9+47)))), -11132])
        self.assertAlmostEqual(res[0], np.float32(-0.25 * self.M * cos(radians(46.9))), 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #7.1: {str(res)}")
        self.assertAlmostEqual(res[1], -11132, 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #7.2: {str(res)}")

        res = convert_lonlat_to_m(-2.25,36.6,-2,37)
        #print("expected #8", [np.float32(-0.25 * self.M * cos(radians(0.5*(36.6+37)))), -4*11132])
        self.assertAlmostEqual(res[0], np.float32(-0.25 * self.M * cos(radians(36.6))), 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #8.1: {str(res)}")
        self.assertAlmostEqual(res[1], -4*11132, 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #8.2: {str(res)}")

        res = convert_lonlat_to_m(-1.6,37.6,-2,37)
        #print("expected #9", [np.float32(0.4 * self.M * cos(radians(0.5*(37.6+37)))), 6*11132])
        self.assertAlmostEqual(res[0], np.float32(0.4 * self.M * cos(radians(37.6))), 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #8.1: {str(res)}")
        self.assertAlmostEqual(res[1], 6*11132, 
                               delta=0.1, msg=f"Error - convert_lonlat_to_m - #8.2: {str(res)}")


class TestGetDistFromLineString(unittest.TestCase):
    def test_get_dist_from_linestring(self):
        M = 111320  # Approx. meters per degree latitude

        line1 = LineString([[12,47], [12,47.1], [12,47.2], [12,47.4]])
        #print(str(line1))
        d1 = get_dist_from_linestring(str(line1))
        self.assertAlmostEqual(d1, 0.4*M, delta=1, 
                               msg=f"Error - get_dist_from_linestring - #1: {d1:.2f}")
        
        line2 = 'LINESTRING (12.5 47, 12.6 47, 12.8 47, 12.9 47)'        
        d2 = get_dist_from_linestring(str(line2))
        self.assertAlmostEqual(d2, 0.4*M*cos(radians(47)), delta=1, 
                               msg=f"Error - get_dist_from_linestring - #2: {d2:.2f}")

        line3 = 'LINESTRING (12.3 37, 12.6 37, 12.8 37, 12.9 37)'        
        d3 = get_dist_from_linestring(str(line3))
        self.assertAlmostEqual(d3, 0.6*M*cos(radians(37)), delta=1, 
                               msg=f"Error - get_dist_from_linestring - #3: {d3:.2f}")

        line4 = 'LINESTRING (12.3 37, 12.6 38, 12.8 37, 12.9 38)'        
        d4 = get_dist_from_linestring(str(line4))
        exp = sqrt((0.3*M*cos(radians(37.5)))**2 + M**2) + sqrt((0.2*M*cos(radians(37.5)))**2 + M**2) + sqrt((0.1*M*cos(radians(37.5)))**2 + M**2)
        self.assertAlmostEqual(d4, round(exp), delta=1, 
                               msg=f"Error - get_dist_from_linestring - #4: {d4:.2f}")

class TestGetDistFromMultiLineString(unittest.TestCase):
    def test_get_dist_from_multilinestring(self):
        lat, lon = 52, 15
        line1 = LineString([[lon-1,lat-1], [lon+1,lat-1]])
        line2 = LineString([[lon-0.5,lat+0.5], [lon+1,lat+0.5]])
        line3 = LineString([[lon-1,lat+1], [lon, lat], [lon+1,lat-1]])

        l1 = get_dist_from_linestring(line1)
        l2 = get_dist_from_linestring(line2)
        l3 = get_dist_from_linestring(line3)

        self.assertAlmostEqual(l1, 140111, delta=1, 
                               msg=f"Error - get_dist_from_ls #1 - {l1:.1f}")
        self.assertAlmostEqual(l2, 101651, delta=1, 
                               msg=f"Error - get_dist_from_ls #2 - {l2:.1f}")
        self.assertAlmostEqual(l3, 261452, delta=1, 
                               msg=f"Error - get_dist_from_ls #3 - {l3:.1f}")
        #print(f"l1  : {l1:.1f} - l2: {l2:.1f} - l3: {l3:.1f}")
        #print(f"mls1: {l1+l2+l3:.1f}")
        #print(f"mls2: {l1+l3:.1f}")
        #print(f"mls3: {l2+l3:.1f}")

        mls1 = MultiLineString([line1, line2, line3])
        mls2 = MultiLineString([line1, line3])
        mls3 = MultiLineString([line2, line3])

        d1 = get_dist_from_multilinestring(mls1)
        d1x = get_dist_from_linestring(mls1)
        self.assertAlmostEqual(d1, 503215, delta=1, 
                               msg=f"Error - get_dist_from_mls #1.1 - {d1:.1f} vs 503215")
        self.assertTrue(d1 == d1x, f"Error - get_dist_from_mls #1.2 - {d1:.1f} vs {d1x:.1f}")

        d2 = get_dist_from_multilinestring(mls2)
        d2x = get_dist_from_linestring(mls2)
        self.assertAlmostEqual(d2, 401564, delta=1, 
                               msg=f"Error - get_dist_from_mls #2.1 - {d2:.1f} vs 401564")
        self.assertTrue(d2 == d2x, f"Error - get_dist_from_mls #2.2 - {d2:.1f} vs {d2x:.1f}")

        d3 = get_dist_from_multilinestring(mls3)
        d3x = get_dist_from_linestring(mls3)
        self.assertAlmostEqual(d3, 363103, delta=1, 
                               msg=f"Error - get_dist_from_mls #3.1 - {d3:.1f} vs 363103")
        self.assertTrue(d3 == d3x, f"Error - get_dist_from_mls #3.2 - {d3:.1f} vs {d3x:.1f}")
