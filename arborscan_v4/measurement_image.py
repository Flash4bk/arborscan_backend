"""Canonical uncropped EXIF-oriented RGB pixels, bounded before allocation."""
from io import BytesIO
import numpy as np
from PIL import Image, ImageOps


def decode_oriented_rgb(raw: bytes, max_pixels=25_000_000):
    with Image.open(BytesIO(raw)) as im:
        if im.width * im.height > max_pixels:
            raise ValueError('Image resolution exceeds limit')
        return np.array(ImageOps.exif_transpose(im).convert('RGB'))
