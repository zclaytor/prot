import warnings
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker


class WaveletTransform(object):
    """
    The WaveletTransform class represents a two-dimensional, time-varying
    power spectrum. The x-axis represents time, and the y-axis represents
    frequency or period. Each point in x-y space is colored by the power.
    As of right now, the units of the wavelet transform are meaningless.
    
    Attributes
    ----------
    frequency : `~astropy.units.Quantity`
        Array of frequencies as an AstroPy Quantity object.
    power : `~astropy.units.Quantity`
        Array of power-spectral-densities. The Quantity must have units of
        `flux^2 / freq_unit`, where freq_unit is the unit of the frequency
        attribute.
    nyquist : float
        The Nyquist frequency of the lightcurve. In units of freq_unit, where
        freq_unit is the unit of the frequency attribute.
    #label : str
    #    Human-friendly object label, e.g. "KIC 123456789".
    #targetid : str
    #    Identifier of the target.
    default_view : "frequency" or "period"
        Should plots be shown in frequency space or period space by default?
    #meta : dict
    #    Free-form metadata associated with the Periodogram.
    """
    def __init__(self, time, flux, period, power, wavelet, w,
                 nyquist=None, label=None, 
                 targetid=None, meta={}):
        self.time = time
        self.flux = flux
        self.period = period
        self.power = power
        self.wavelet = wavelet
        self.w = w
        self.nyquist = nyquist
        self.label = label
        self.targetid = targetid
        self.meta = meta

    def __repr__(self):
        return("WaveletTransform(ID: {})".format(self.label))

    @property
    def frequency(self):
        """Returns the array of frequency, i.e. 1/period."""
        return 1. / self.period

    @property
    def gwps(self):
        """Returns the Global Wavelet Power Spectrum."""
        return self.power.sum(axis=1)

    @property
    def coi(self):
        """Returns Cone of Influence."""
        fourier_factor = (4 * np.pi) / (self.w + np.sqrt(2 + self.w**2))
        time = self.time - self.time[0]
        tt = np.minimum(time, time[-1]-time)
        coi = fourier_factor * tt / np.sqrt(2)
        return coi

    @property
    def frequency_at_max_power(self):
        """Returns the frequency corresponding to the highest peak in the periodogram."""
        return 1. / self.period_at_max_power

    @property
    def period_at_max_power(self):
        """Returns the period corresponding to the highest peak in the periodogram."""
        return self.period[np.nanargmax(self.gwps)]

    @staticmethod
    def from_lightcurve(lc,
                        wavelet=signal.morlet2,
                        w=6,
                        period=None,
                        minimum_period=None,
                        maximum_period=None,
                        period_samples=512):
        """Docstring
        """
        if np.isnan(lc.flux).any():
            lc = lc.fill_gaps()

        time = lc.time.copy()
        time -= time[0]
        flux = lc.flux.copy()
        flux -= flux.mean()

        nyquist = 0.5 * (1./(np.median(np.diff(time))))

        if period is None:
            if minimum_period is None:
                minimum_period = 1/nyquist
            if maximum_period is None:
                maximum_period = time[-1]
            period = np.geomspace(minimum_period, maximum_period, period_samples)
        else:
            if any(b is not None for b in [minimum_period, maximum_period]):
                warnings.warn(
                    "Both `period` and at least one of `minimum_period` or "
                    "`maximum_period` have been specified. Using constraints "
                    "from `period`.", RuntimeWarning)

        widths = w * nyquist * period / np.pi
        cwtm = signal.cwt(flux, wavelet, widths, w=w)
        power = np.abs(cwtm)**2 / widths[:, np.newaxis]
        
        return WaveletTransform(lc.time.copy(), lc.flux.copy(),
                                period, power,
                                wavelet=wavelet, w=w,
                                nyquist=nyquist,
                                targetid=lc.targetid,
                                label=lc.label,
                                meta=lc.meta)
        
    def plot(self, ax=None, xlabel=None, ylabel=None, title='',
             **kwargs):
        """Plots the WaveletTransform.
        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.pcolormesh`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """

        if ax is None:
            fig, ax = plt.subplots()

        # Plot wavelet power spectrum
        ax.pcolormesh(self.time, self.period, self.power, shading='auto', **kwargs)

        # Plot cone of influence
        ax.plot(self.time, self.coi, 'k', linewidth=1, rasterized=True)
        ax.plot(self.time, self.coi, 'w:', linewidth=1, rasterized=True)
        
        if xlabel is None:
            xlabel = "Time - 2457000 (BTJD days)"
        if ylabel is None:
            ylabel = "Period (days)"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_yscale('log', base=2)
        ax.set_ylim(self.period.max(), self.period.min())

        ax.set_title(title)
        return ax


def wavelet_transform(lc, 
                      wavelet=signal.morlet2,
                      w=6,
                      period=None,
                      minimum_period=None,
                      maximum_period=None,
                      period_samples=512):
    """Docstring
    """
    if np.isnan(lc.flux).any():
        lc = lc.fill_gaps()

    time = lc.time.copy()
    time -= time[0]
    flux = lc.flux.copy()
    flux -= flux.mean()

    nyquist = 0.5 * (1./(np.median(np.diff(time))))

    if period is None:
        if minimum_period is None:
            minimum_period = 1/nyquist
        if maximum_period is None:
            maximum_period = time[-1]
        period = np.geomspace(minimum_period, maximum_period, period_samples)
    else:
        if any(b is not None for b in [minimum_period, maximum_period]):
            warnings.warn(
                "Both `period` and at least one of `minimum_period` or "
                "`maximum_period` have been specified. Using constraints "
                "from `period`.", RuntimeWarning)

    widths = w * nyquist * period / np.pi
    cwtm = signal.cwt(flux, wavelet, widths, w=w)
    power = np.abs(cwtm)**2 / widths[:, np.newaxis]
        
    fourier_factor = (4 * np.pi) / (self.w + np.sqrt(2 + self.w**2))
    tt = np.minimum(time, time[-1]-time)
    coi = fourier_factor * tt / np.sqrt(2)

    return time, period, power, coi

def wavelet_plot(time, period, power, flux, coi=None,
                 xlabel='time', ylabel='period', flabel='flux', plabel='power', 
                 **kw):
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 0.2), height_ratios=(0.3, 1), hspace=0, wspace=0)
    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.pcolormesh(time, period, power, cmap='viridis', shading='auto', rasterized=True, **kw)
    
    if coi is not None:
        ax1.plot(time, coi, 'k', linewidth=1, rasterized=True)
        ax1.plot(time, coi, 'w:', linewidth=1, rasterized=True)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax1.set_yscale('log', base=2)
    ax1.set_ylim(period.max(), period.min())
    
    #ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax1.ticklabel_format(axis='y', style='scientific')
    
    ax2 = fig.add_subplot(gs[1, 1], sharey=ax1)
    global_ws = power.sum(axis=1)
    ax2.plot(global_ws, period, 'k', rasterized=True)
    ax2.set_xlabel(plabel)
    ax2.set_xlim(0, 1.25*global_ws.max())
    ax2.xaxis.set_ticks([])
    ax2.yaxis.set_visible(False)

    ax3 = fig.add_subplot(gs[0, 0], sharex=ax1)
    ax3.plot(time, flux, 'k', rasterized=True)
    ax3.set_ylabel(flabel)
    ax3.xaxis.set_visible(False)
    ax3.yaxis.set_ticks([])
    ax3.set_xlim(time.min(), time.max())
  
    for ax in ax1, ax2, ax3:
        ax.tick_params('both', direction='inout')
    
    return fig, (ax1, ax2, ax3)