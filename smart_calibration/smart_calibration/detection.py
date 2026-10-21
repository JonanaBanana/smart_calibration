"""Chessboard detection on prepared 8-bit grayscale images."""

import cv2


def sb_flags(normalize=False, exhaustive=False, accuracy=False):
    """
    Combine findChessboardCornersSB flags.

    See explore.md for what they do and their measured effect: normalize (histogram
    equalization) helps on dark / low-contrast images, accuracy (internal 2x upsampling) lowers
    the corner error by 30-45 % at ~4x the runtime, exhaustive made no difference so far.
    """
    flags = 0
    if normalize:
        flags |= cv2.CALIB_CB_NORMALIZE_IMAGE
    if exhaustive:
        flags |= cv2.CALIB_CB_EXHAUSTIVE
    if accuracy:
        flags |= cv2.CALIB_CB_ACCURACY
    return flags


def detect_chessboard(gray, cols, rows, flags=0):
    """
    Detect a chessboard with cols x rows inner corners.

    Returns a (rows, cols, 2) float32 array of corner positions in pixels, row-major with pixel
    centres at integer coordinates (OpenCV convention), or None if no board was found.
    """
    found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), flags=flags)
    if not found:
        return None
    return corners.reshape(rows, cols, 2)
