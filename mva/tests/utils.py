from rootpy.plotting import Hist
import numpy as np


def transform(x, scale=1):
    return 2.0 / (1.0 + np.exp(- scale * x)) - 1.0


def get_background_signal(nbins=100, scale=1, sig_events=1000, bkg_events=1000):

    bkg_scores = transform(np.random.normal(-.1, .2, size=bkg_events),
        scale=scale)
    sig_scores = transform(np.random.normal(.1, .2, size=sig_events),
        scale=scale)
    bkg_hist = Hist(nbins, -1, 1)
    sig_hist = Hist(nbins, -1, 1)
    bkg_hist.fill_array(bkg_scores)
    sig_hist.fill_array(sig_scores)
    return bkg_hist, sig_hist
