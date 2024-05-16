import warnings
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

import lightkurve as lk

from . import MPLSTYLE


@FuncFormatter
def myFormatter(x, pos):
    if x < 1:
        n = np.ceil(-np.log10(x)).astype(int)
        return f"{x:.{n}f}"
    else:
        return f"{x:.0f}"


class WaveletTransform(object):
    """
    The WaveletTransform class represents a two-dimensional, time-varying
    power spectrum. The x-axis represents time, and the y-axis represents
    frequency or period. Each point in x-y space is colored by the power.
    As of right now, the units of the wavelet transform are meaningless.
    
    Attributes
    ----------
    lightcurve : `lightkurve.LightCurve`
        light curve from which the WaveletTransform is constructed.
    period : numpy array
        Array of periods.
    power : numpy array
        Array of power-spectral-densities.
    phase : numpy array
        Array of signal phase, computed using `numpy.angle`.
    wavelet : one of the wavelets from `scipy.signal`
        The wavelet with which the WaveletTransform is constructed.
    w : int
        The wavelet parameter.
    nyquist : float
        The Nyquist frequency of the lightcurve.
    """
    def __init__(self, lightcurve, period, power, phase, wavelet, w, nyquist=None):
        self.lightcurve = lightcurve
        self.period = period
        self.power = power
        self.phase = phase
        self.wavelet = wavelet
        self.w = w
        self.nyquist = nyquist

    def __repr__(self):
        return("WaveletTransform(ID: {})".format(self.label))

    @property
    def time(self):
        """Returns the array of time from the light curve."""
        return self.lightcurve.time.value

    @property
    def label(self):
        """Returns the label from the light curve."""
        return self.lightcurve.label

    @property
    def targetid(self):
        """Returns the targetid from the light curve."""
        return self.lightcurve.targetid

    @property
    def meta(self):
        """Returns meta dict from the light curve."""
        return self.lightcurve.meta

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
        """Computes the wavelet power spectrum from a `lightkurve.LightCurve`.

        Parameters
        ----------
        lc : `lightkurve.LightCurve`
            The light curve from which to compute the wavelet transform.
        wavelet : one of the wavelets from `scipy.signal`
            The wavelet with which the WaveletTransform is constructed.
        w : int
            The wavelet parameter.
        period : numpy array
            Array of periods at which to compute the power.
        minimum_period : float
            If specified, use this rather than the nyquist frequency.
        maximum_period : float
            If specified, use this rather than the time baseline.
        period_samples : int
            If `period` is not specified, use `minimum_period` and 
            `maximum_period` to define the period array, using `period_samples`
            points.
        """
        if np.isnan(lc.flux).any():
            lc = lc.remove_nans()

        time = lc.time.copy().value
        time -= time[0]
        flux = lc.flux.copy().value
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
        phase = np.angle(cwtm)
        
        return WaveletTransform(lc, period, power, phase,
                                wavelet=wavelet, w=w,
                                nyquist=nyquist)
        
    def plot(self, ax=None, xlabel=None, ylabel=None, title='', plot_coi=True,
             cmap='binary', style=None, **kwargs):
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
        plot_coi : bool
            Whether to plot the cone of influence (COI)
        cmap : str or matplotlib colormap object
            Colormap for wavelet transform heat map.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.pcolormesh`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if style is None or style == "lightkurve":
            style = MPLSTYLE
        
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot wavelet power spectrum
            ax.pcolormesh(self.time, self.period, self.power, shading='auto', 
                cmap=cmap, **kwargs)

            # Plot cone of influence
            if plot_coi:
                ax.plot(self.time, self.coi, 'k', linewidth=1, rasterized=True)
                ax.plot(self.time, self.coi, 'w:', linewidth=1, rasterized=True)
            
            if xlabel is None:
                if isinstance(self.lightcurve, lk.TessLightCurve):
                    xlabel = "Time - 2457000 [BTJD days]"
                elif isinstance(self.lightcurve, lk.KeplerLightCurve):
                    xlabel = "Time - 2454833 [BKJD days]"
                else:
                    xlabel = "Time (days)"
            if ylabel is None:
                ylabel = "Period (days)"

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.set_ylim(self.period.max(), self.period.min())
            ax.yaxis.set_major_formatter(myFormatter)
            ax.set_title(title)
        return ax

    def plot_gwps(self, ax=None, scale="linear", xlabel=None, ylabel=None, title='', style=None,
                  **kwargs):
        """Plots the Global Wavelet Power Spectrum

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        scale: str
            Set x,y axis to be "linear" or "log". Default is linear.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if style is None or style == "lightkurve":
            style = MPLSTYLE
        
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot global wavelet power spectrum
            ax.plot(self.period, self.gwps, 'k', **kwargs)
            
            if ylabel is None:
                ylabel = "Power (arbitrary units)"
            if xlabel is None:
                xlabel = "Period (days)"

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(scale)
            ax.set_yscale(scale)
            ax.set_title(title)
        return ax

    def _plot_gwps_vertical(self, ax=None, scale="linear", 
                            xlabel=None, ylabel=None, title='', 
                            style=None, **kwargs):
        """Plots the Global Wavelet Power Spectrum vertically.
        Intended for use with `WaveletTransform.plot_all`.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            A matplotlib axes object to plot into. If no axes is provided,
            a new one will be generated.
        scale: str
            Set x,y axis to be "linear" or "log". Default is linear.
        xlabel : str
            Plot x axis label
        ylabel : str
            Plot y axis label
        title : str
            Plot set_title
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.plot`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if style is None or style == "lightkurve":
            style = MPLSTYLE
        
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot global wavelet power spectrum
            ax.plot(self.gwps, self.period, 'k', **kwargs)
            
            if xlabel is None:
                xlabel = "Power (arbitrary units)"
            if ylabel is None:
                ylabel = "Period (days)"

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(scale)
            #ax.set_yscale(scale) plot shares yscale with wavelet decomposition
            ax.set_title(title)
        return ax

    def plot_lightcurve(self, *args, **kwargs):
        """Wrapper for `lightkurve.LightCurve.plot`."""
        return self.lightcurve.plot(*args, **kwargs)

    def plot_all(self, figsize=(10, 8), style=None,
        wavelet_kwargs={}, gwps_kwargs={}, lightcurve_kwargs={}, **kwargs):
        """Plots all 3 graphs, with the largest being the wavelet transform
        in the main panel, the light curve on top, and the global wavelet
        power spectrum to the right.

        Parameters
        ----------
        figsize : tuple
            The size of the figure.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        plot_coi : bool
            Whether to plot the cone of influence (COI) on the main panel.
        wavelet_kwargs : dict
            Keyword arguments to be passed to `WaveletTransform.plot`.
        gwps_kwargs : dict
            Keyword arguments to be passed to `WaveletTransform.plot_gwps`.
        lightcurve_kwargs : dict
            Keyword arguments to be passed to `WaveletTransform.plot_lightcurve`.
        """
        if style is None or style == "lightkurve":
            style = MPLSTYLE
        
        with plt.style.context(style):
            fig = plt.figure(figsize=figsize, tight_layout=True)
            gs = fig.add_gridspec(2, 2, 
                width_ratios=(1, 0.2), 
                height_ratios=(0.3, 1), 
                hspace=0, wspace=0)
            
            ax1 = fig.add_subplot(gs[1, 0])
            try:
                plot_coi = kwargs.pop("plot_coi")
            except KeyError:
                plot_coi = True

            ax1 = self.plot(ax=ax1, plot_coi=plot_coi, **wavelet_kwargs, **kwargs)
                    
            ax2 = fig.add_subplot(gs[1, 1])
            ax2 = self._plot_gwps_vertical(ax=ax2, 
                xlabel='Power', ylabel='', 
                **gwps_kwargs, **kwargs)
            ax2.set_yscale('log')
            ax2.set_ylim(self.period.max(), self.period.min())
            ax2.xaxis.set_ticks([])
            ax2.yaxis.set_visible(False)

            ax3 = fig.add_subplot(gs[0, 0], sharex=ax1)
            ax3 = self.plot_lightcurve(ax=ax3, 
                xlabel='', ylabel='Flux', label=None, 
                **lightcurve_kwargs, **kwargs)
            ax3.xaxis.set_visible(False)
            ax3.yaxis.set_ticks([])
        
        return fig, (ax1, ax2, ax3)

    def plot_phase(self, ax=None, xlabel=None, ylabel=None, title='', plot_coi=True,
             cmap='binary', style=None, **kwargs):
        """Plots the WaveletTransform phase.

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
        plot_coi : bool
            Whether to plot the cone of influence (COI)
        cmap : str or matplotlib colormap object
            Colormap for wavelet transform heat map.
        style : str
            Path or URL to a matplotlib style file, or name of one of
            matplotlib's built-in stylesheets (e.g. 'ggplot').
            Lightkurve's custom stylesheet is used by default.
        kwargs : dict
            Dictionary of arguments to be passed to `matplotlib.pyplot.pcolormesh`.
        Returns
        -------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes object.
        """
        if style is None or style == "lightkurve":
            style = MPLSTYLE
        
        with plt.style.context(style):
            if ax is None:
                fig, ax = plt.subplots()

            # Plot wavelet power spectrum
            ax.pcolormesh(self.time, self.period, self.phase, shading='auto', 
                cmap=cmap, **kwargs)

            # Plot cone of influence
            if plot_coi:
                ax.plot(self.time, self.coi, 'k', linewidth=1, rasterized=True)
                ax.plot(self.time, self.coi, 'w:', linewidth=1, rasterized=True)
           
            if xlabel is None:
                if isinstance(self.lightcurve, lk.TessLightCurve):
                    xlabel = "Time - 2457000 [BTJD days]"
                elif isinstance(self.lightcurve, lk.KeplerLightCurve):
                    xlabel = "Time - 2454833 [BKJD days]"
                else:
                    xlabel = "Time (days)"
            if ylabel is None:
                ylabel = "Period (days)"

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.set_ylim(self.period.max(), self.period.min())
            ax.yaxis.set_major_formatter(myFormatter)

            ax.set_title(title)
        return ax
    
def wavelet_transform(lc, 
                      wavelet=signal.morlet2,
                      w=6,
                      period=None,
                      minimum_period=None,
                      maximum_period=None,
                      period_samples=512):
    """Computes the wavelet power spectrum from a `lightkurve.LightCurve`.

    Parameters
    ----------
    lc : `lightkurve.LightCurve`
        The light curve from which to compute the wavelet transform.
    wavelet : one of the wavelets from `scipy.signal`
        The wavelet with which the WaveletTransform is constructed.
    w : int
        The wavelet parameter.
    period : numpy array
        Array of periods at which to compute the power.
    minimum_period : float
        If specified, use this rather than the nyquist frequency.
    maximum_period : float
        If specified, use this rather than the time baseline.
    period_samples : int
        If `period` is not specified, use `minimum_period` and 
        `maximum_period` to define the period array, using `period_samples`
        points.

    Returns
    -------
    time : numpy array
        The time axis for the wavelet transform.
    period : numpy array
        The period axis for the wavelet transform.
    power : numpy array
        The power spectral density.
    coi : numpy array
        The cone of influence, below which edge effects become worrisome.
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
        
    fourier_factor = (4 * np.pi) / (w + np.sqrt(2 + w**2))
    tt = np.minimum(time, time[-1]-time)
    coi = fourier_factor * tt / np.sqrt(2)

    return time, period, power, coi

def wavelet_plot(time, period, power, flux, coi=None,
                 xlabel='time', ylabel='period', flabel='flux', plabel='power',
                 **kw):
    """Plots all 3 graphs, with the largest being the wavelet transform
    in the main panel, the light curve on top, and the global wavelet
    power spectrum to the right.

    Parameters
    ----------
    time : numpy array
        The time axis for the wavelet transform.
    period : numpy array
        The period axis for the wavelet transform.
    power : numpy array
        The power spectral density.
    flux : numpy array
        The light curve flux.
    coi : numpy array
        The cone of influence, below which edge effects become worrisome.
    xlabel : str, optional
        The label for the time axis.
    ylabel : str, optional
        The label for the period axis.
    flabel : str, optional
        The label for the flux axis of the light curve panel.
    plabel : str, optional
        The label for the power axis of the global wavelet power spectrum panel.

    Returns
    -------
    fig : `matplotlib.Figure`
        The figure for the plot.
    (ax1, ax2, ax3) : tuple of `matplotlib.pyplot.Axes`
        Tuple containing the axes handles for the plot.
    """
    fig = plt.figure(figsize=(8, 6), tight_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=(1, 0.2), height_ratios=(0.3, 1), hspace=0, wspace=0)
    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.pcolormesh(time, period, power, cmap='binary', shading='auto', rasterized=True, **kw)
    
    if coi is not None:
        ax1.plot(time, coi, 'k', linewidth=1, rasterized=True)
        ax1.plot(time, coi, 'w:', linewidth=1, rasterized=True)
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    ax1.set_yscale('log')
    ax1.set_ylim(period.max(), period.min())
    ax1.yaxis.set_major_formatter(myFormatter)

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
    
    return fig, (ax1, ax2, ax3)
