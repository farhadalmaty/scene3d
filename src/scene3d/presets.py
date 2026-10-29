"""Approximate fields of view (degrees across the LONG side of a 16:9 phone video) for the iPhone cameras.
Only a fallback: the app should send the real intrinsics (AVFoundation can deliver them per frame), or run
`scene3d calibrate-fov` once per camera."""
CAMERA_FOV = {"main": 69.0, "ultrawide": 100.0, "tele": 38.0}
