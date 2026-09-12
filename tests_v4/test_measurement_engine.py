import unittest

import numpy as np

from arborscan_v4.measurement_engine import (
    CalibrationRequest,
    compute_pixel_geometry,
    fuse_measurements,
    resolve_calibration,
)


class MeasurementEngineTests(unittest.TestCase):
    def make_mask(self):
        mask = np.zeros((240, 160), dtype=np.uint8)
        # crown
        mask[20:150, 30:130] = 255
        # trunk
        mask[150:221, 73:87] = 255
        return mask

    def test_pixel_geometry_has_no_metric_assumption(self):
        geom = compute_pixel_geometry(self.make_mask())
        self.assertEqual(geom.tree_height_px, 201.0)
        self.assertGreater(geom.crown_width_px, 90)
        self.assertIsNotNone(geom.trunk_width_px_estimate)

        req = CalibrationRequest()
        cal = resolve_calibration(geom, req)
        fused = fuse_measurements(geom, req, cal, 0.9)
        self.assertIsNone(fused.height.value_m)
        self.assertIsNone(fused.crown_width.value_m)
        self.assertIsNone(fused.trunk_diameter.value_m)
        self.assertEqual(fused.status, "absolute_scale_required")

    def test_manual_scale_derives_height_and_crown_but_not_dbh(self):
        geom = compute_pixel_geometry(self.make_mask())
        req = CalibrationRequest(manual_scale_px_to_m=0.01)
        cal = resolve_calibration(geom, req)
        fused = fuse_measurements(geom, req, cal, 0.9)
        self.assertAlmostEqual(fused.height.value_m, 2.01, places=3)
        self.assertIsNotNone(fused.crown_width.value_m)
        self.assertIsNone(fused.trunk_diameter.value_m)
        self.assertEqual(fused.status, "partial_measurement")

    def test_ar_height_calibrates_photo_and_keeps_height_direct(self):
        geom = compute_pixel_geometry(self.make_mask())
        req = CalibrationRequest(ar_height_m=20.1, ar_quality=0.9)
        cal = resolve_calibration(geom, req)
        self.assertTrue(cal.available)
        self.assertEqual(cal.source, "ar_height_calibration")
        fused = fuse_measurements(geom, req, cal, 0.92)
        self.assertAlmostEqual(fused.height.value_m, 20.1, places=3)
        self.assertEqual(fused.height.source, "ar")
        self.assertIsNotNone(fused.crown_width.value_m)

    def test_conflicting_ar_scales_do_not_invent_third_measurement(self):
        geom = compute_pixel_geometry(self.make_mask())
        req = CalibrationRequest(ar_height_m=20.0, ar_crown_width_m=30.0, ar_quality=0.9)
        cal = resolve_calibration(geom, req)
        self.assertTrue(cal.conflict)
        self.assertFalse(cal.available)
        fused = fuse_measurements(geom, req, cal, 0.9)
        self.assertEqual(fused.height.value_m, 20.0)
        self.assertEqual(fused.crown_width.value_m, 30.0)
        self.assertIsNone(fused.trunk_diameter.value_m)
        self.assertEqual(fused.status, "partial_measurement")

    def test_explicit_dbh_line_can_be_metric_only_with_scale(self):
        geom = compute_pixel_geometry(self.make_mask())
        req = CalibrationRequest(
            reference_length_m=1.0,
            reference_length_px=100.0,
            reference_same_plane=True,
            dbh_width_px=60.0,
            dbh_measurement_height_m=1.3,
        )
        cal = resolve_calibration(geom, req)
        fused = fuse_measurements(geom, req, cal, 0.9)
        self.assertAlmostEqual(fused.trunk_diameter.value_m, 0.6, places=4)
        self.assertEqual(fused.trunk_diameter.standard, "dbh_1_3m")


if __name__ == "__main__":
    unittest.main()
