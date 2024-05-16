import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = "{}/misc/lightkurve.mplstyle".format(PACKAGEDIR)

from .wavelet_transform import WaveletTransform

__version__ = "0.2.0"