#!/usr/bin/env python
# Author: Noel Dawe noel@dawe.me
# License: GPLv3

import os
import math
from math import sqrt
from itertools import izip
from fnmatch import fnmatch

import ROOT
from rootpy.io import root_open
from rootpy.plotting import Hist, Hist2D, Hist3D
from rootpy.utils.silence import silence_sout
from rootpy.context import do_nothing

try:
    from collections import OrderedDict
except ImportError:
    from rootpy.extern.ordereddict import OrderedDict

from . import log; log = log[__name__]


def process_measurement(m,
                        remove_window=None,
                        remove_window_channels=None,

                        split_norm_shape=False,

                        zero_negs=False,

                        fill_empties=False,
                        fill_empties_samples=None,
                        fill_empties_channels=None,

                        rebin=1,
                        rebin_channels=None,

                        merge_bins=None,
                        merge_bins_channels=None,

                        flat_signal=None,

                        rename_names=None,
                        rename_types=None,
                        rename_samples=None,
                        rename_channels=None,
                        rename_func=None,

                        drop_np_names=None,
                        drop_np_types=None,
                        drop_np_samples=None,
                        drop_np_channels=None,

                        prune_samples=False,
                        prune_samples_threshold=1e-7,

                        symmetrize_names=None,
                        symmetrize_types=None,
                        symmetrize_samples=None,
                        symmetrize_channels=None,
                        symmetrize_partial=False,
                        asymmetry_threshold = 1,

                        smooth_histosys=False,
                        smooth_histosys_iterations=1,
                        smooth_histosys_samples=None,
                        smooth_histosys_channels=None,

                        prune_histosys=False,
                        prune_histosys_samples=None,
                        prune_histosys_channels=None,
                        prune_histosys_blacklist=None,
                        prune_histosys_method='max',
                        prune_histosys_threshold=0.1,

                        prune_overallsys=False,
                        prune_overallsys_samples=None,
                        prune_overallsys_channels=None,
                        prune_overallsys_blacklist=None,
                        prune_overallsys_threshold=0.5,

                        uniform_binning=False,

                        data_is_partially_blind=False,

                        hybrid_data=False,
                        hybrid_data_mu=1.0):

    for c in m.channels:
        log.info("processing measurement `{0}` channel `{1}` ...".format(
            m.name, c.name))

        # remove a range of bins (useful for sideband fit bias tests)
        if remove_window is not None and matched(
                c.name, remove_window_channels):
            log.info("removing window {0} from channel `{1}`".format(
                remove_window, c.name))
            c.data.hist = apply_remove_window(c.data.hist, remove_window)
            for s in c.samples:
                s.hist = apply_remove_window(s.hist, remove_window)
                for histosys in s.histo_sys:
                    histosys.high = apply_remove_window(
                        histosys.high, remove_window)
                    histosys.low = apply_remove_window(
                        histosys.low, remove_window)

        # flat signal binning
        if flat_signal is not None:
            sig = [s for s in c.samples if is_signal(s)]
            if sig:
                # get sum of signal
                total_sig = sum([s.hist.Clone(shallow=True) for s in sig])
                # get N quantiles
                quantiles = total_sig.quantiles(flat_signal, strict=True)
                # rebin all histos

                data_hist = c.data.hist
                if data_is_partially_blind:
                    # determine first blinded bin
                    for ibin in xrange(data_hist.nbins(0), 0, -1):
                        if data_hist.GetBinContent(ibin) > 0:
                            break
                    log.info("detected data blinding in bin {0:d} "
                             "and above".format(ibin + 1))
                    blind_cut = data_hist.GetBinCenter(ibin + 1)
                data_hist = apply_rebin(data_hist, quantiles)
                if data_is_partially_blind:
                    # blind from bin containing blind_cut
                    blind_bin = data_hist.FindBin(blind_cut)
                    for ibin in xrange(blind_bin, data_hist.nbins(0) + 2):
                        data_hist.SetBinContent(ibin + 1, 0)
                        data_hist.SetBinError(ibin + 1, 0)
                c.data.hist = data_hist

                for s in c.samples:
                    log.info("applying rebinning {0} on sample `{1}`".format(
                             quantiles, s.name))
                    s.hist = apply_rebin(s.hist, quantiles)
                    for histosys in s.histo_sys:
                        histosys.high = apply_rebin(histosys.high, quantiles)
                        histosys.low = apply_rebin(histosys.low, quantiles)
            else:
                log.warning(
                    "not rebinning to flatten signal in category `{0}` "
                    "since no signal is present".format(c.name))

        # rebin
        if rebin != 1 and matched(c.name, rebin_channels):
            data_hist = c.data.hist
            if data_is_partially_blind:
                # determine first blinded bin
                for ibin in xrange(data_hist.nbins(0), 0, -1):
                    if data_hist.GetBinContent(ibin) > 0:
                        break
                log.info("detected data blinding in bin {0:d} "
                         "and above".format(ibin + 1))
                blind_cut = data_hist.GetBinCenter(ibin + 1)
            data_hist = apply_rebin(data_hist, rebin)
            if data_is_partially_blind:
                # blind from bin containing blind_cut
                blind_bin = data_hist.FindBin(blind_cut)
                for ibin in xrange(blind_bin, data_hist.nbins(0) + 2):
                    data_hist.SetBinContent(ibin + 1, 0)
                    data_hist.SetBinError(ibin + 1, 0)
            c.data.hist = data_hist
            for s in c.samples:
                log.info("applying rebinning {0:d} on sample `{1}`".format(
                         rebin, s.name))
                s.hist = apply_rebin(s.hist, rebin)
                for histosys in s.histo_sys:
                    histosys.high = apply_rebin(histosys.high, rebin)
                    histosys.low = apply_rebin(histosys.low, rebin)

        # merge bins
        if merge_bins and matched(c.name, merge_bins_channels):
            c.data.hist = apply_merge_bins(c.data.hist, merge_bins)
            for s in c.samples:
                log.info("merging bins {0} in sample `{1}`".format(
                    ', '.join([':'.join(map(str, r)) for r in merge_bins]),
                    s.name))
                s.hist = apply_merge_bins(s.hist, merge_bins)
                for histosys in s.histo_sys:
                    histosys.high = apply_merge_bins(histosys.high, merge_bins)
                    histosys.low = apply_merge_bins(histosys.low, merge_bins)

        # split HistoSys into HistoSys and OverallSys components
        if split_norm_shape:
            for s in c.samples:
                apply_split_norm_shape(s)

        # rename specific NPs
        if rename_names and (rename_func is not None) and matched(
                c.name, rename_channels):
            for s in c.samples:
                if not matched(s.name, rename_samples):
                    continue
                if matched('histosys', rename_types, ignore_case=True):
                    for np in s.histo_sys:
                        if matched(np.name, rename_names, ignore_case=True):
                            np.name = rename_func(c.name, s.name, np.name)
                if matched('overallsys', rename_types, ignore_case=True):
                    for np in s.overall_sys:
                        if matched(np.name, rename_names, ignore_case=True):
                            np.name = rename_func(c.name, s.name, np.name)

        # drop NPs by name
        if drop_np_names and matched(c.name, drop_np_channels):
            for s in c.samples:
                if not matched(s.name, drop_np_samples):
                    continue
                if matched('histosys', drop_np_types, ignore_case=True):
                    names = []
                    for np in s.histo_sys:
                        if matched(np.name, drop_np_names, ignore_case=True):
                            names.append(np.name)

                    for name in names:
                        log.info("removing HistoSys `{0}` from sample "
                                 "`{1}`".format(name, s.name))
                        s.RemoveHistoSys(name)

                if matched('overallsys', drop_np_types, ignore_case=True):
                    names = []
                    for np in s.overall_sys:
                        if matched(np.name, drop_np_names, ignore_case=True):
                            names.append(np.name)

                    for name in names:
                        log.info("removing OverallSys `{0}` from sample "
                                 "`{1}`".format(name, s.name))
                        s.RemoveOverallSys(name)

        # remove samples with integral below threshold
        if prune_samples:
            for s in c.samples:
                if s.hist.Integral() < prune_samples_threshold:
                    log.info("removing sample {0} in channel {1}".format(
                             s.name, c.name))
                    c.RemoveSample(s.name)

        # apply fill_empties on nominal histograms
        if fill_empties and matched(c.name, fill_empties_channels):
            for s in c.samples:
                if not matched(s.name, fill_empties_samples):
                    continue
                if not is_signal(s):
                    log.info("applying fill_empties on sample `{0}`".format(
                             s.name))
                    s.hist = apply_fill_empties(s.hist)

        # smooth
        if smooth_histosys and matched(c.name, smooth_histosys_channels):
            for s in c.samples:
                if not matched(s.name, smooth_histosys_samples):
                    continue
                for histosys in s.histo_sys:
                    smooth_shape(s, s.hist, histosys,
                                 smooth_histosys_iterations)

        # prune shapes
        if prune_histosys and matched(c.name, prune_histosys_channels):

            # total bkg needed for max method
            if prune_histosys_method == 'max':
                total_bkg = sum([s.hist.Clone(shallow=True)
                                 for s in c.samples if not is_signal(s)])
                total_sig = sum([s.hist.Clone(shallow=True)
                                 for s in c.samples if is_signal(s)])

            for s in c.samples:
                if not matched(s.name, prune_histosys_samples):
                    continue
                names = []
                if prune_histosys_method == 'max':
                    for histosys in s.histo_sys:
                        if prune_histosys_blacklist and matched(
                                histosys.name,
                                prune_histosys_blacklist):
                            continue
                        if not shape_is_significant(
                                total_sig if is_signal(s) else total_bkg,
                                histosys.high, histosys.low,
                                prune_histosys_threshold):
                            names.append(histosys.name)

                elif prune_histosys_method == 'chi2':
                    for histosys in s.histo_sys:
                        if prune_histosys_blacklist and matched(
                                histosys.name,
                                prune_histosys_blacklist):
                            continue
                        if not shape_chi2_test(s.hist,
                                               histosys.high, histosys.low,
                                               prune_histosys_threshold):
                            names.append(histosys.name)

                elif prune_histosys_method == 'ks':
                    for histosys in s.histo_sys:
                        if prune_histosys_blacklist and matched(
                                histosys.name,
                                prune_histosys_blacklist):
                            continue
                        if (histosys.high.KolmogorovTest(s.hist) < prune_histosys_threshold and
                            histosys.low.KolmogorovTest(s.hist) < prune_histosys_threshold):
                            names.append(histosys.name)

                for name in names:
                    log.info("removing HistoSys `{0}` from sample `{1}` using method `{2}`".format(
                        name, s.name, prune_histosys_method))
                    s.RemoveHistoSys(name)

        # prune overallsys
        if prune_overallsys and matched(c.name, prune_overallsys_channels):
            for s in c.samples:
                if not matched(s.name, prune_overallsys_samples):
                    continue
                names = []
                for overallsys in s.overall_sys:
                    if prune_overallsys_blacklist and matched(
                            overallsys.name,
                            prune_overallsys_blacklist):
                        continue
                    if (abs(overallsys.high - 1.) * 100. < prune_overallsys_threshold and
                        abs(overallsys.low - 1.) * 100. < prune_overallsys_threshold):
                        names.append(overallsys.name)
                for name in names:
                    log.info("removing OverallSys `{0}` from sample `{1}`".format(name, s.name))
                    s.RemoveOverallSys(name)

        # symmetrize NPs
        if symmetrize_names and matched(c.name, symmetrize_channels):
            for s in c.samples:
                if not matched(s.name, symmetrize_samples):
                    continue
                if matched('histosys', symmetrize_types, ignore_case=True):
                    for np in s.histo_sys:
                        if matched(np.name, symmetrize_names, ignore_case=True):
                            if symmetrize_histosys(np, s.hist, partial=symmetrize_partial, asymmetry_threshold=asymmetry_threshold):
                                log.info("symmetrized HistoSys `{0}` in sample `{1}`".format(
                                    np.name, s.name))

                if matched('overallsys', symmetrize_types, ignore_case=True):
                    for np in s.overall_sys:
                        if matched(np.name, symmetrize_names, ignore_case=True):
                            if symmetrize_overallsys(np, nominal=1., partial=symmetrize_partial):
                                log.info("symmetrized OverallSys `{0}` in sample `{1}`".format(
                                    np.name, s.name))

        # zero out negatives
        if zero_negs:
            for s in c.samples:
                s.hist = apply_zero_negs(s.hist)
                for histosys in s.histo_sys:
                    histosys.high = apply_zero_negs(histosys.high)
                    histosys.low = apply_zero_negs(histosys.low)

        # convert to uniform binning
        if uniform_binning:
            c.data.hist = to_uniform_binning(c.data.hist)
            for s in c.samples:
                s.hist = to_uniform_binning(s.hist)
                for histosys in s.histo_sys:
                    histosys.high = to_uniform_binning(histosys.high)
                    histosys.low = to_uniform_binning(histosys.low)

        # construct hybrid data
        if hybrid_data:
            data_hist = c.data.hist
            if data_is_partially_blind:
                # get first bin to construct hybrid data in
                for blind_bin in xrange(data_hist.nbins(0), 0, -1):
                    if data_hist.GetBinContent(blind_bin) > 0:
                        break
                blind_bin += 1
            else:
                blind_bin = 0
            # get sum of background and sum of signal
            total_bkg = sum([s.hist.Clone(shallow=True) for s in c.samples if not is_signal(s)])
            sigs = [s.hist.Clone(shallow=True) for s in c.samples if is_signal(s)]
            if not sigs:
                log.warning(
                    "not constructing hybrid data in channel `{0}` "
                    "because no signal is present".format(c.name))
            else:
                log.info(
                    "constructing hybrid data with mu={0:f} in bin {1:d} and above".format(
                        hybrid_data_mu, blind_bin))
                total_sig = sum(sigs)
                hybrid_data = total_bkg + (total_sig * hybrid_data_mu)
                for ibin in xrange(blind_bin, data_hist.nbins(0) + 2):
                    data_hist.SetBinContent(ibin, hybrid_data.GetBinContent(ibin))
                    data_hist.SetBinError(ibin, hybrid_data.GetBinError(ibin))


def matched(name, patterns, ignore_case=False):
    if not patterns:
        return True
    if ignore_case:
        name = name.lower()
    for pattern in patterns:
        if ignore_case:
            pattern = pattern.lower()
        if fnmatch(name, pattern):
            return True
    return False


def apply_remove_window(hist, window):
    """
    Remove a window of bins from a histogram
    """
    low, high = window
    keep_bins = []
    for bin in hist.bins():
        if ((low < hist.xedgesl(bin.idx) < high) or
            (low < hist.xedgesh(bin.idx) < high) or
            (low < bin.x < high)):
            continue
        keep_bins.append(bin.idx)
    hist_window = Hist(len(keep_bins), 0, len(keep_bins),
                       type=hist.TYPE, name=hist.name + '_window')
    for idx_window, idx in enumerate(keep_bins):
        hist_window[idx_window + 1] = hist[idx]
    return hist_window


def apply_split_norm_shape(s):
    from rootpy.stats.histfactory import split_norm_shape
    for histosys in s.histo_sys:
        # skip histosys for which overallsys already exist
        if s.GetOverallSys(histosys.name) is not None:
            continue
        log.info("splitting HistoSys `{0}` in sample `{1}`".format(
            histosys.name, s.name))
        norm, shape = split_norm_shape(histosys, s.hist)
        histosys.high = shape.high
        histosys.low = shape.low
        s.AddOverallSys(norm)


def symmetrize_histosys(np, nominal, partial=False, asymmetry_threshold=1):
    """
    Full Symmetrization (default)
    -----------------------------
    If high and low are on one side of nominal in a given bin, then
    then set the side with the lower absolute deviation to have the
    same absolute variation as the other, but on the other side of nominal.

    Partial Symmetrization
    ----------------------
    Same as above but set the side with the lower absolute deviation
    to the nominal value.

    Asymmetry threshold
    -------------------
    threshold = 1 means no correction needed: min(up/do, do/u) is always smaller than 1
    threshold of 0.5 means correction needed if abs(do) < 0.5abs(up) or abs(up)<0.5abs(do)
    """
    high = np.high.Clone(name=np.high.name + '_symmetrized', shallow=True)
    low = np.low.Clone(name=np.low.name + '_symmetrized', shallow=True)
    symmetrized = False
    for high_bin, low_bin, nom_bin in izip(high.bins(), low.bins(),
                                           nominal.bins()):
        high_value = high_bin.value
        low_value = low_bin.value
        nom_value = nom_bin.value
        up = high_value - nom_value
        dn = low_value - nom_value
        if up * dn > 0:
            symmetrized = True
            # same side variation
            if abs(up) > abs(dn):
                if partial:
                    low_bin.value = nom_value
                else:
                    low_bin.value = nom_value - up
            else:
                if partial:
                    high_bin.value = nom_value
                else:
                    high_bin.value = nom_value - dn
        elif up!=0 and dn!=0 and min(abs(up/dn), abs(dn/up))<asymmetry_threshold:
            symmetrized = True
            if abs(up) > abs(dn):
                low_bin.value = nom_value - up
            else:
                high_bin.value = nom_value - dn

    if symmetrized:
        np.high = high
        np.low = low
    return symmetrized


def symmetrize_overallsys(np, nominal=1., partial=False):
    """
    Full Symmetrization (default)
    -----------------------------
    If high and low are on one side of nominal, then
    then set the side with the lower absolute deviation to have the
    same absolute variation as the other, but on the other side of nominal.

    Partial Symmetrization
    ----------------------
    Same as above but set the side with the lower absolute deviation
    to the nominal value.
    """
    up = np.high - nominal
    dn = np.low - nominal
    if up * dn > 0:
        symmetrized = True
        # same side variation
        if abs(up) > abs(dn):
            if partial:
                np.low = nominal
            else:
                np.low = nominal - up
        else:
            if partial:
                np.high = nominal
            else:
                np.high = nominal - dn
        return True
    return False


def shape_chi2_test(nom, up, down, threshold):
    """
    Calculate the Chi^2 of the up and down variations and return True the
    variations are significant given a threshold on the minimum Chi^2 value.
    """
    chi2up = 0.
    chi2dn = 0.
    nbin = 0
    for nom_bin, up_bin, down_bin in izip(
            nom.bins(overflow=False),
            up.bins(overflow=False),
            down.bins(overflow=False)):
        n = nom_bin.value
        u = up_bin.value
        d = down_bin.value
        eup = max(nom_bin.error, up_bin.error)
        edn = max(nom_bin.error, down_bin.error)
        if not (n > 0. and eup > 0. and edn > 0.):
            continue
        chi2up += ((u - n) / eup) ** 2.
        chi2dn += ((d - n) / edn) ** 2.
        nbin += 1
    chi2up = ROOT.TMath.Prob(chi2up, nbin)
    chi2dn = ROOT.TMath.Prob(chi2dn, nbin)
    # return True if the shape should be kept
    return min(chi2up, chi2dn) <= threshold


def shape_is_significant(total, high, low, threshold=0.1):
    """
    For a given shape systematic calculate ``s_i=|up_i-down_i|/stat_total_i``,
    where ``up_i`` is up variation in ``bin_i``, ``down_i`` is down variation
    in ``bin_i``, ``stat_total_i`` is the statistical uncertainty for total
    background prediction in ``bin_i``. If ``max(s_i)<0.1``, then drop this
    shape systematic.
    """
    for bin_total, bin_high, bin_low in zip(
            total.bins(), high.bins(), low.bins()):
        diff = abs(bin_high.value - bin_low.value)
        if bin_total.error == 0:
            if diff != 0:
                return True
            continue
        sig = abs(bin_high.value - bin_low.value) / bin_total.error
        if sig > threshold:
            return True
    return False


def smooth_shape(sample, nominal, histosys, iterations=1):
    """
    Smooth shape systematics with respect to the nominal histogram by applying
    TH1::Smooth() on the ratio of systematic / nominal.
    """
    log.info("smoothing HistoSys `{0}` in sample `{1}`".format(
        histosys.name, sample.name))
    high = histosys.high
    low = histosys.low

    high_name = high.name + '_smoothed_{0}'.format(iterations)
    low_name = low.name + '_smoothed_{0}'.format(iterations)

    nominal_high = nominal.Clone(shallow=True)
    nominal_low = nominal.Clone(shallow=True)

    for bin_high, bin_low, bin_nom in izip(
            high.bins(overflow=True),
            low.bins(overflow=True),
            nominal_high.bins(overflow=True)):
        if bin_nom.value < 1E-3:
            bin_nom.value = (bin_high.value + bin_low.value) / 2.

    ratio_high = high / nominal_high
    ratio_low = low / nominal_low

    ratio_high.Smooth(iterations)
    ratio_low.Smooth(iterations)

    histosys.high = ratio_high * nominal_high
    histosys.low = ratio_low * nominal_low

    histosys.high.name = high_name
    histosys.low.name = low_name


def uniform_channel(c):
    c.data.hist = to_uniform_binning(c.data.hist)
    for s in c.samples:
        s.hist = to_uniform_binning(s.hist)
        for histosys in s.histo_sys:
            histosys.high = to_uniform_binning(histosys.high)
            histosys.low = to_uniform_binning(histosys.low)


def to_uniform_binning(hist):
    """
    For some obscure technical reason, HistFactory can't handle histograms with
    variable width bins. This function takes any histogram and outputs a new
    histogram with constant width bins along all axes by using the bin indices
    of the input histogram as the x-axis of the new histogram.
    """
    # is this histogram already uniform?
    if hist.uniform(axis=None):
        return hist
    log.info("converting histogram `{0}` to uniform binning".format(hist.name))
    new_hist = hist.uniform_binned(name=hist.name + '_uniform_binning')
    return new_hist


def apply_zero_negs(hist):
    """
    Return a clone of this histogram with all negative bins set to zero. The
    errors of these bins are left untouched.
    """
    new_hist = hist.Clone(name=hist.name + '_nonegs', shallow=True)
    applied = False
    for bin in new_hist.bins():
        if bin.value < 0:
            applied = True
            log.warning(
                "zeroing negative bin {0:d} in `{1}`".format(
                    bin.idx, hist.name))
            bin.value = 0.
    if applied:
        return new_hist
    return hist


def apply_merge_bins(hist, bin_ranges, axis=0):
    """
    Merge ranges of bins by bin indices.
    `bin_ranges` is a list of 2-tuples of bin index ranges.
    `axis` is the axis along which to merge the bins (0, 1, or 2)
    """
    new_hist = hist.merge_bins(bin_ranges, axis=axis)
    new_hist.name = hist.name + '_merged'
    return new_hist


def apply_rebin(hist, bins, axis=0):
    """
    Rebin the histogram
    """
    if isinstance(bins, int) and bins < 2:
        return hist
    new_hist = hist.rebinned(bins, axis=0)
    if isinstance(bins, int):
        new_hist.name = hist.name + '_rebin_{0:d}'.format(bins)
    else:
        new_hist.name = hist.name + '_rebinned'
    return new_hist


def apply_fill_empties(hist):
    """
    Return a clone of the input histogram where the empty bins have been filled
    with the average weight and the errors of these bins set to sqrt(<w^2>).
    If no bins were altered, then return the original histogram
    """
    fixed_hist = hist.Clone(name=hist.name + '_fill_empties', shallow=True)

    # value
    avWeightBin = hist.GetSumOfWeights() / hist.GetEntries()
    # error
    sumW2TotBin = sum([bin.error**2 for bin in hist.bins()])
    sqrt_avW2Bin = sqrt(sumW2TotBin / hist.GetEntries())

    applied = False
    for bin in fixed_hist.bins():
        if bin.value < 1E-6:
            log.warning(
                "filling bin {0:d} containing {1:f}+/-{2:f} in "
                "`{3}` with average weight {4:f}+/-{5:f}".format(
                    bin.idx, bin.value, bin.error,
                    hist.name,
                    avWeightBin, sqrt_avW2Bin))
            bin.value = avWeightBin
            bin.error = sqrt_avW2Bin
            applied = True

    if applied:
        return fixed_hist
    return hist


def is_signal(sample, poi='SigXsecOverSM'):
    for norm in sample.GetNormFactorList():
        if norm.name == poi:
            return True
    return False


def equal(left, right, precision=1E-7):
    return abs(left - right) < precision


def _get_named(sequence, name):
    for i, thing in enumerate(sequence):
        if thing.GetName() == name:
            return i, thing
    return None, None


def partitioning(left, right):
    """
    Partition each collection into three sets containing the elements that
    overlap between the collections, elements that are only found in the left
    collection and elements that are only found in the right collection.
    """
    overlap = []
    left_only  = []
    right_only = right[:]
    for thing in left:
        idx, other = _get_named(right_only, thing.name)
        if idx is not None:
            overlap.append((thing, other))
            right_only.pop(idx)
        else:
            left_only.append(thing)
    return overlap, left_only, right_only


def _diff_sequence_helper(left, right, diff_func, parent=None, **kwargs):
    differ = False
    overlap, left_only, right_only = partitioning(
        left, right)
    for left_o, right_o in overlap:
        _differ = diff_func(left_o, right_o, **kwargs)
        if _differ:
            if parent is not None:
                log.warning(
                    "{0} differs in {1}".format(
                        left_o, parent))
            else:
                log.warning(
                    "{0} differs".format(
                        left_o))
            differ = _differ
    for coll, sym in ((left_only, '<'), (right_only, '>')):
        for thing in coll:
            log.warning("{0} {1} {2}".format(parent, sym, thing))
            differ = True
    return differ


def diff_histograms(left, right, compare_edges=False, precision=1E-7):
    # compare dimensionality
    if left.GetDimension() != right.GetDimension():
        log.warning(
            "histgrams {0} and {1} differ in dimensionality: {2:d} and {3:d}".format(
                left.name, right.name,
                left.GetDimension(), right.GetDimension()))
        return True
    # compare axis
    axis_names = 'xyz'
    for axis in xrange(left.GetDimension()):
        if left.nbins(axis) != right.nbins(axis):
            log.warning(
                "histograms {0} and {1} differ in the number of bins "
                "along the {2}-axis: {3:d} {4:d}".format(
                    left.name, right.name, axis_names[axis],
                    left.nbins(axis),
                    right.nbins(axis)))
            return True
    differ = False
    # compare contents and errors
    for left_bin, right_bin in izip(left.bins(), right.bins()):
        if not equal(left_bin.value, right_bin.value, precision=precision):
            log.warning(
                "histgrams {0} and {1} differ in content in bin {2:d}: {3:f} {4:f}".format(
                    left.name, right.name, left_bin.idx,
                    left_bin.value, right_bin.value))
            differ = True
        if not equal(left_bin.error, right_bin.error, precision=precision):
            log.warning(
                "histgrams {0} and {1} differ in error in bin {2:d}: {3:f} {4:f}".format(
                    left.name, right.name, left_bin.idx,
                    left_bin.error, right_bin.error))
            differ = True
    return differ


def diff_overallsys(left, right, precision=1E-7):
    differ = False
    if not equal(left.low, right.low, precision=precision):
        differ = True
        log.warning("OverallSys {0} differs for low values: {1:f} {2:f}".format(
            left.name, left.low, right.low))
    if not equal(left.high, right.high, precision=precision):
        differ = True
        log.warning("OverallSys {0} differs for high values: {1:f} {2:f}".format(
            left.name, left.high, right.high))
    return differ


def diff_histosys(left, right, precision=1E-7):
    differ = False
    if diff_histograms(left.low, right.low, precision=precision):
        differ = True
    if diff_histograms(left.high, right.high, precision=precision):
        differ = True
    return differ


def diff_samples(left, right, precision=1E-7):
    # compare nominal
    differ = diff_histograms(left.hist, right.hist, precision=precision)
    # compare OverallSys
    _differ = _diff_sequence_helper(left.overall_sys, right.overall_sys,
        diff_func=diff_overallsys, parent=left, precision=precision)
    if _differ:
        differ = differ
    # compare HistoSys
    _differ = _diff_sequence_helper(left.histo_sys, right.histo_sys,
        diff_func=diff_histosys, parent=left, precision=precision)
    if _differ:
        differ = differ
    return differ


def diff_channels(left, right, precision=1E-7):
    return _diff_sequence_helper(left.samples, right.samples,
        diff_func=diff_samples, parent=left, precision=precision)


def diff_measurements(left, right, precision=1E-7):
    return _diff_sequence_helper(left.channels, right.channels,
        diff_func=diff_channels, parent=left, precision=precision)


def parse_names(names):
    _names = OrderedDict()
    if not names:
        return _names
    for name in names:
        in_name, _, out_name = name.partition('::')
        _names[in_name] = out_name or in_name
    return _names


def yields(m,
           channels=None,
           channel_names=None,
           sample_names=None,
           unblind=False,
           explode=False,
           xbin1=1, xbin2=-2):
    channel_names = parse_names(channel_names)
    sample_names = parse_names(sample_names)
    yields = {}
    data = {}
    # get yields for each sample in each channel
    for c in m.channels:
        if channels and c.name not in channels:
            continue
        if c.name not in channel_names:
            channel_names[c.name] = c.name
        if unblind and c.data.hist:
            if explode:
                nbinsx = c.data.hist.nbins(0, overflow=True)
                first = xbin1 % nbinsx
                last = xbin2 % nbinsx
                contents = []
                # get contents for each bin
                for i in xrange(first, last + 1):
                    contents.append(ufloat(*c.data.total(
                        xbin1=i, xbin2=i)))
                data[c.name] = contents
            else:
                data[c.name] = ufloat(*c.data.total(
                    xbin1=xbin1, xbin2=xbin2))
        for s in c.samples:
            if s.name not in sample_names:
                sample_names[s.name] = s.name
            if s.name not in yields:
                yields[s.name] = {}
            if explode:
                nbinsx = s.hist.nbins(0, overflow=True)
                first = xbin1 % nbinsx
                last = xbin2 % nbinsx
                contents = []
                # get contents for each bin
                for i in xrange(first, last + 1):
                    contents.append(ufloat(*s.total(
                        xbin1=i, xbin2=i)))
                num_bin_cols = len(contents)
                yields[s.name][c.name] = contents
            else:
                value = ufloat(*s.total(
                    xbin1=xbin1, xbin2=xbin2))
                yields[s.name][c.name] = value
    # get total background and signal
    total_background = {}
    total_signal = {}
    for c in m.channels:
        if channels and c.name not in channels:
            continue
        if explode:
            contents = []
            # get contents for each bin
            for i in xrange(first, last + 1):
                contents.append(ufloat(
                    *c.total(where=lambda s: not is_signal(s),
                    xbin1=i, xbin2=i)))
            total_background[c.name] = contents
        else:
            value = ufloat(*c.total(where=lambda s: not is_signal(s),
                xbin1=xbin1, xbin2=xbin2))
            total_background[c.name] = value
        if not c.has_sample_where(is_signal):
            # don't include signal yields for channels without signal
            continue
        if explode:
            contents = []
            # get contents for each bin
            for i in xrange(first, last + 1):
                contents.append(ufloat(
                    *c.total(where=lambda s: is_signal(s),
                    xbin1=i, xbin2=i)))
            total_signal[c.name] = contents
        else:
            value = ufloat(*c.total(where=is_signal,
                xbin1=xbin1, xbin2=xbin2))
            total_signal[c.name] = value
    # print a LaTeX table
    if explode:
        print r"\begin{tabular}{*{%d}{c|}c}" % (len(channel_names) * num_bin_cols)
        print r"\hline" * 2
        print " & ".join(["\multirow{2}{*}{Process/Category}"] + ["\multicolumn{%d}{c|}{%s}" % (num_bin_cols, c) for c in channel_names.values()]) + r"\\"
        print r"\hline"
        print " & ".join([" "] + map(str, range(first, last + 1)) * len(channel_names.values()) ) + r"\\"
        print r"\hline"
        for sample_name, out_sample_name in sample_names.items():
            channel_yield = yields[sample_name]
            print " & ".join([out_sample_name] +
                [" & ".join(map(repr, channel_yield[c])) if c in channel_yield else " & " * (num_bin_cols - 1)
                    for c in channel_names.keys()]) + r"\\"
        if total_background:
            print r"\hline"
            print " & ".join(["Total Background"] +
                [" & ".join(map(repr, total_background[c])) if c in total_background else " & " * (num_bin_cols - 1)
                    for c in channel_names.keys()]) + r"\\"
        if total_signal:
            print r"\hline"
            print " & ".join(["Total Signal"] +
                [" & ".join(map(repr, total_signal[c])) if c in total_signal else " & " * (num_bin_cols - 1)
                    for c in channel_names.keys()]) + r"\\"
        if unblind:
            print r"\hline"
            print " & ".join(["Data"] +
                [" & ".join(map(repr, data[c])) if c in data else " & " * (num_bin_cols - 1)
                    for c in channel_names.keys()]) + r"\\"
        print r"\hline" * 2
        print r"\end{tabular}"
    else:
        print r"\begin{tabular}{%s}" % '|'.join('c' * (len(channel_names) + 1))
        print r"\hline" * 2
        print " & ".join(["Process/Category"] + channel_names.values()) + r"\\"
        print r"\hline"
        for sample_name, out_sample_name in sample_names.items():
            channel_yield = yields[sample_name]
            print " & ".join([out_sample_name] +
                [repr(channel_yield[c]) if c in channel_yield else " "
                    for c in channel_names.keys()]) + r"\\"
        if total_background:
            print r"\hline"
            print " & ".join(["Total Background"] +
                [repr(total_background[c]) if c in total_background else " "
                    for c in channel_names.keys()]) + r"\\"
        if total_signal:
            print r"\hline"
            print " & ".join(["Total Signal"] +
                [repr(total_signal[c]) if c in total_signal else " "
                    for c in channel_names.keys()]) + r"\\"
        if unblind:
            print r"\hline"
            print " & ".join(["Data"] +
                [repr(data[c]) if c in data else " "
                    for c in channel_names.keys()]) + r"\\"
        print r"\hline" * 2
        print r"\end{tabular}"
