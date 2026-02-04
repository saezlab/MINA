from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from . import down, pl, up

__all__ = ["pl", "up", "down"]

try:
    __version__ = version(__name__)  # resolves to "mina"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0+unknown"