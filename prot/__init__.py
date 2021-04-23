import os

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
MPLSTYLE = "{}/misc/lightkurve.mplstyle".format(PACKAGEDIR)

from .version import __version__
from .wavelet_transform import WaveletTransform