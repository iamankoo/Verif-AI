"""Read individual members of a remote ZIP over HTTP range requests (no full download).

Used to fetch a *sampled subset* of the openly licensed DeepFakeFace dataset (openrail) from
Hugging Face without downloading its ~5 GB of archives. Standard library only.
"""
from __future__ import annotations

import io
import time
import urllib.request
import zipfile
from typing import Optional

UA = {"User-Agent": "verifai-research/1.0"}


class RangeFile(io.RawIOBase):
    """Seekable read-only file object backed by HTTP Range requests."""

    def __init__(self, url: str, block: int = 1 << 20, retries: int = 4):
        self.url, self.block, self.retries = url, block, retries
        self.pos = 0
        self._cache_start, self._cache = -1, b""
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range")          # "bytes 0-0/12345"
            self.size = int(cr.split("/")[-1])
            self.url = r.geturl()                         # resolved CDN URL

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, start: int, end: int) -> bytes:
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return b""

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        s, e = self.pos, self.pos + n
        if not (self._cache_start <= s and e <= self._cache_start + len(self._cache)):
            fetch_len = max(n, self.block)
            self._cache = self._fetch(s, min(self.size, s + fetch_len) - 1)
            self._cache_start = s
        out = self._cache[s - self._cache_start: e - self._cache_start]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b)); b[:len(data)] = data; return len(data)


def open_remote_zip(url: str, block: int = 1 << 20) -> zipfile.ZipFile:
    return zipfile.ZipFile(RangeFile(url, block=block))
