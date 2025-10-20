from importlib.metadata import version

from . import down, pl, up

__all__ = ["pl", "up", "down"]

__version__ = version("MINAtraining")
