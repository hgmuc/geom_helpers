import unittest
# python -m unittest test_utm_helpers.py

from geom_helpers.utm_helper import get_utm_zone, get_utm_crs, get_utm_crs_number

class TestUTMHelpers(unittest.TestCase):
    def test_get_utm_zone_north(self):
        lat, lon = 45.0, 7.0
        zone, hemi = get_utm_zone(lat, lon)
        self.assertEqual(zone, 32)
        self.assertEqual(hemi, 'north')

        lat, lon = 48.0, 13.0
        zone, hemi = get_utm_zone(lat, lon)
        self.assertEqual(zone, 33)
        self.assertEqual(hemi, 'north')

        lat, lon = 54.0, 4.0
        zone, hemi = get_utm_zone(lat, lon)
        self.assertEqual(zone, 31)
        self.assertEqual(hemi, 'north')

        lat, lon = 54.0, -4.0
        zone, hemi = get_utm_zone(lat, lon)
        self.assertEqual(zone, 30)
        self.assertEqual(hemi, 'north')

        lat, lon = 34.0, -9.0
        zone, hemi = get_utm_zone(lat, lon)
        self.assertEqual(zone, 29)
        self.assertEqual(hemi, 'north')

    def test_get_utm_zone_south(self):
        lat, lon = -23.5, 133.9
        zone, hemi = get_utm_zone(lat, lon)
        self.assertEqual(zone, 53)
        self.assertEqual(hemi, 'south')

    def test_get_utm_crs_north(self):
        lat, lon = 45.0, 7.0
        crs = get_utm_crs(lat, lon)
        self.assertEqual(crs, 'epsg:32632')

        lat, lon = 48.0, 13.0
        crs = get_utm_crs(lat, lon)
        self.assertEqual(crs, 'epsg:32633')

        lat, lon = 54.0, 4.0
        crs = get_utm_crs(lat, lon)
        self.assertEqual(crs, 'epsg:32631')

        lat, lon = 45.0, -4.0
        crs = get_utm_crs(lat, lon)
        self.assertEqual(crs, 'epsg:32630')

    def test_get_utm_crs_south(self):
        lat, lon = -23.5, 133.9
        crs = get_utm_crs(lat, lon)
        self.assertEqual(crs, 'epsg:32753')

    def test_get_utm_crs_number_north(self):
        lat, lon = 45.0, 7.0
        crs_num = get_utm_crs_number(lat, lon)
        self.assertEqual(crs_num, 32632)

    def test_get_utm_crs_number_south(self):
        lat, lon = -23.5, 133.9
        crs_num = get_utm_crs_number(lat, lon)
        self.assertEqual(crs_num, 32753)

#if __name__ == '__main__':
#    unittest.main()
