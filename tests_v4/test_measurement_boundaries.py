from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from arborscan_v4.measurement_image import decode_oriented_rgb
from arborscan_v4.measurement_engine import CalibrationRequest,compute_pixel_geometry,resolve_calibration,fuse_measurements


def test_exif_rotation_and_decode_limit():
    image=Image.new('RGB',(40,20));exif=image.getexif();exif[274]=6
    stream=BytesIO();image.save(stream,format='JPEG',exif=exif)
    assert decode_oriented_rgb(stream.getvalue()).shape==(40,20,3)
    with pytest.raises(ValueError):decode_oriented_rgb(stream.getvalue(),max_pixels=100)


def test_unbound_ar_is_not_attached_to_another_tree():
    g=compute_pixel_geometry(np.pad(np.ones((100,20),dtype=np.uint8),10))
    req=CalibrationRequest(ar_height_m=10,ar_trunk_diameter_m=.3,ar_trunk_measurement_height_m=1.3)
    out=fuse_measurements(g,req,resolve_calibration(g,req),.9)
    assert out.height.value_m is None and out.trunk_diameter.value_m is None


def test_whole_tree_mask_width_is_not_a_measured_crown_and_ar_diameter_not_automatically_dbh():
    g=compute_pixel_geometry(np.pad(np.ones((100,20),dtype=np.uint8),10))
    req=CalibrationRequest(manual_scale_px_to_m=.02,ar_trunk_diameter_m=.3,
                           ar_trunk_measurement_height_m=1.4,ar_photo_matches=True)
    out=fuse_measurements(g,req,resolve_calibration(g,req),.9)
    assert out.crown_width.value_m is None
    assert out.trunk_diameter.value_m==.3 and out.trunk_diameter.standard is None


def test_reference_and_ar_sources_are_kept_distinct():
    g=compute_pixel_geometry(np.pad(np.ones((100,20),dtype=np.uint8),10))
    req=CalibrationRequest(reference_length_m=2,reference_length_px=100,reference_same_plane=True,
       ar_height_m=25,ar_photo_matches=True,crown_width_px=40)
    out=fuse_measurements(g,req,resolve_calibration(g,req),.9)
    assert out.height.value_m==25 and out.height.source=='ar'
    assert out.crown_width.value_m==.8 and out.crown_width.source=='reference+vision'


def test_reference_requires_depth_confirmation_not_a_confidence_discount():
    g=compute_pixel_geometry(np.ones((100,20),dtype=np.uint8))
    for confirmed in (False,None):
        req=CalibrationRequest(reference_length_m=2,reference_length_px=100,reference_same_plane=confirmed)
        assert not resolve_calibration(g,req).available
    # A manual scale is an explicit legacy measurement, not rejected by an
    # arbitrary metres-per-pixel threshold without a physical camera model.
    assert resolve_calibration(g,CalibrationRequest(manual_scale_px_to_m=2)).px_to_m==2
