import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker


def wavelet_transform(lc, 
    wavelet=signal.morlet2,
    minimum_period=None,
    maximum_period=None,
    period_samples=512,
    w=6):
    """
    Assume:
    - time begins at zero and is uniformly spaced
    - flux has zero mean and unit standard deviation
    """

    time = lc.time - lc.time[0]
    flux = lc.flux - lc.flux.mean()

    dt = np.median(time[1:] - time[:-1])
    sampling_rate = 1/dt

    if minimum_period is None:
        minimum_period = 2*dt
    if maximum_period is None:
        maximum_period = time[-1] / 2

    period = np.geomspace(minimum_period, maximum_period, period_samples)
    freq = 1/period
    widths = w * sampling_rate / (2*freq*np.pi)

    cwtm = signal.cwt(flux, wavelet, widths, w=w)
    power = np.abs(cwtm)**2 / widths[:, np.newaxis]
    
    fourier_factor = (4 * np.pi) / (w + np.sqrt(2 + w**2)) / np.sqrt(2)
    tt = np.minimum(time, time[-1]-time)
    coi = fourier_factor * tt
    
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