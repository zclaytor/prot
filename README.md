# prot
Python tools for time-series frequency analysis, particularly for stellar rotational light curves. 

## Getting Started
Install using pip:
```bash
pip install git+https://github.com/zclaytor/prot
```

To make a Wavelet Transform, start with a `lightkurve.LightCurve`:
```python
import lightkurve as lk
result = lk.search_lightcurve("TIC 149308317", author="tess-spoc", sector=range(14), cadence=1800)
lcs = result.download_all(flux_column="sap_flux").stitch()

from prot import WaveletTransform
wt = WaveletTransform.from_lightcurve(lcs)

wt.plot_all()
```
