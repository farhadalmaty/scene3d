"""Read the display rotation of an MP4/MOV file (iPhone portrait videos are stored landscape + a rotation flag).

Pure-Python parse of the track header ('tkhd') transformation matrix: no ffmpeg/ffprobe needed.
Only the small 'moov' box is read (it is at the END of iPhone files), never the media data.
Returns the clockwise rotation in degrees to apply for upright display: 0, 90, 180 or 270."""
from __future__ import annotations
import math
import os
import struct


def _boxes(buf: bytes, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        size, typ = struct.unpack(">I4s", buf[pos:pos + 8]); hdr = 8
        if size == 1 and pos + 16 <= end:
            size = struct.unpack(">Q", buf[pos + 8:pos + 16])[0]; hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr or pos + size > end:
            return
        yield typ.decode("latin1"), pos + hdr, pos + size
        pos += size


def _read_moov(path) -> bytes | None:
    with open(path, "rb") as f:
        total = os.fstat(f.fileno()).st_size; pos = 0
        while pos + 8 <= total:
            f.seek(pos); h = f.read(16)
            size, typ = struct.unpack(">I4s", h[:8]); hdr = 8
            if size == 1: size = struct.unpack(">Q", h[8:16])[0]; hdr = 16
            elif size == 0: size = total - pos
            if size < hdr: return None
            if typ == b"moov":
                f.seek(pos + hdr); return f.read(size - hdr)
            if pos == 0 and typ not in (b"ftyp", b"free", b"wide", b"skip", b"mdat", b"moov"):
                return None                                        # not an MP4/MOV container (AVI, MKV, ...)
            pos += size
    return None


def _rotation_from_tkhd(b: bytes, s: int) -> int | None:
    ver = b[s]
    off = s + 4 + (20 if ver == 0 else 32) + 16                    # start of the 3x3 matrix
    a, bb, c, d = (struct.unpack(">i", b[off + k:off + k + 4])[0] for k in (0, 4, 12, 16))
    if a == bb == c == d == 0:
        return None
    ang = math.degrees(math.atan2(bb / 65536.0, a / 65536.0))
    q = round(ang / 90.0)
    if abs(ang - 90 * q) > 5:
        return None                                                # skewed / unusual matrix: ignore
    return (int(q) % 4) * 90


def mp4_rotation(path) -> int:
    try:
        moov = _read_moov(path)
        if not moov: return 0
        for t2, s2, e2 in _boxes(moov, 0, len(moov)):
            if t2 != "trak": continue
            for t3, s3, e3 in _boxes(moov, s2, e2):
                if t3 == "tkhd":
                    r = _rotation_from_tkhd(moov, s3)
                    if r: return r                                 # first track with a non-trivial rotation
        return 0
    except (OSError, struct.error, IndexError):
        return 0
