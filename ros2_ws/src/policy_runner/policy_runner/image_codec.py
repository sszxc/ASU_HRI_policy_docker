"""Decode sensor_msgs Image/CompressedImage color frames to RGB ndarrays.

Raw-encoding table and the byte-layout handling (row stride via `step`,
big/little-endian dtype) are adapted from `trajectory_recorder/image.py`
(`/home/asu/Downloads/trajectory_recorder/`, verified working in
`asu_state_reward_ws` -- see repo README "Next phase"). That module targets
OpenCV (BGR) since it also re-encodes to JPEG for MP4s; here the only
consumer is `rerun.Image`, which wants RGB, so the conversions below end in
RGB instead.
"""

import cv2
import numpy as np

_RAW_COLOR_ENCODINGS = {
    # encoding -> (dtype, channels, cv2 conversion to RGB, or None if already RGB)
    "rgb8": (np.uint8, 3, None),
    "bgr8": (np.uint8, 3, cv2.COLOR_BGR2RGB),
    "rgba8": (np.uint8, 4, cv2.COLOR_RGBA2RGB),
    "bgra8": (np.uint8, 4, cv2.COLOR_BGRA2RGB),
    "mono8": (np.uint8, 1, cv2.COLOR_GRAY2RGB),
    "8uc1": (np.uint8, 1, cv2.COLOR_GRAY2RGB),
}


def decode_raw_color(msg):
    """sensor_msgs/Image -> HxWx3 uint8 RGB ndarray."""
    encoding = msg.encoding.lower()
    if encoding not in _RAW_COLOR_ENCODINGS:
        raise ValueError(f"unsupported raw color encoding: {msg.encoding}")
    dtype, channels, conversion = _RAW_COLOR_ENCODINGS[encoding]
    dtype = np.dtype(dtype).newbyteorder(">" if msg.is_bigendian else "<")
    row_values = msg.step // dtype.itemsize
    image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, row_values)
    image = image[:, : msg.width * channels]
    image = (
        image.reshape(msg.height, msg.width, channels)
        if channels > 1
        else image.reshape(msg.height, msg.width)
    )
    return cv2.cvtColor(image, conversion) if conversion is not None else image.copy()


def decode_compressed_color(data):
    """sensor_msgs/CompressedImage `.data` (JPEG/PNG bytes) -> HxWx3 uint8 RGB ndarray."""
    image_bgr = cv2.imdecode(np.frombuffer(bytes(data), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("OpenCV could not decode compressed color image")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
