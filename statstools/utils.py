from . import log; log = log[__name__]
import numpy as np
from rootpy import asrootpy


def efficiency_cut(hist, effic):
    integral = hist.Integral()
    cumsum = 0.
    for bin in hist:
        cumsum += bin.value
        if cumsum / integral > effic:
            return bin.x.low
    return hist.xedges(-1)


def significance(signal, background, min_bkg=0, highstat=True):
    if isinstance(signal, (list, tuple)):
        signal = sum(signal)
    if isinstance(background, (list, tuple)):
        background = sum(background)
    sig_counts = np.array(list(signal.y()))
    bkg_counts = np.array(list(background.y()))
    # reverse cumsum
    S = sig_counts[::-1].cumsum()[::-1]
    B = bkg_counts[::-1].cumsum()[::-1]
    exclude = B < min_bkg
    with np.errstate(divide='ignore', invalid='ignore'):
        if highstat:
            # S / sqrt(S + B)
            sig = np.ma.fix_invalid(np.divide(S, np.sqrt(S + B)),
                fill_value=0.)
        else:
            # sqrt(2 * (S + B) * ln(1 + S / B) - S)
            sig = np.sqrt(2 * (S + B) * np.ln(1 + S / B) - S)
    bins = list(background.xedges())[:-1]
    max_bin = np.argmax(np.ma.masked_array(sig, mask=exclude))
    max_sig = sig[max_bin]
    max_cut = bins[max_bin]
    return sig, max_sig, max_cut


def get_bestfit_nll_workspace(workspace, return_nll=False):
    if return_nll:
        roo_min, nll_func = asrootpy(workspace).fit(return_nll=return_nll)
        fitres = roo_min.save()
        return fitres.minNll(), nll_func
    else:
        roo_min = asrootpy(workspace).fit(return_nll=return_nll)
        fitres = roo_min.save()
        return fitres.minNll()
