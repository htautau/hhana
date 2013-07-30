from . import log; log = log[__name__]

import math, array
from collections import namedtuple

from matplotlib import pyplot as plt
import numpy as np

from rootpy import asrootpy
from rootpy.plotting import Hist
from rootpy.fit import histfactory

from .asymptotics import AsymptoticsCLs
from .significance import runSig
from ..samples import Higgs
from ..plotting import significance, efficiency_cut
from ..classify import histogram_scores
from .utils import get_safe_template
from ..utils import hist_to_dict


def get_stat_w2(hist, x, y=0, z=0):
    """
    Obtain the true number of entries in the bin weighted by w^2
    """
    xl = hist.GetNbinsX() + 2
    yl = hist.GetNbinsY() + 2
    zl = hist.GetNbinsZ() + 2
    assert x >= 0 and x < xl
    assert y >= 0 and y < yl
    assert z >= 0 and z < zl

    if hist.DIM < 3:
        z = 0
    if hist.DIM < 2:
        y = 0
    return hist.GetSumw2().At(xl*yl*z + xl*y + x)


def set_stat_w2(hist, n, x, y=0, z=0):
    """
    Sets the true number of entries in the bin weighted by w^2
    """
    xl = hist.GetNbinsX() + 2
    yl = hist.GetNbinsY() + 2
    zl = hist.GetNbinsZ() + 2
    assert x >= 0 and x < xl
    assert y >= 0 and y < yl
    assert z >= 0 and z < zl

    if hist.DIM < 3:
        z = 0
    if hist.DIM < 2:
        y = 0
    hist.GetSumw2().SetAt(n, xl*yl*z + xl*y + x)


def add_stat_w2(hist1, bins1, hist2, bins2):
    """
    Returns the added (w^2 * N) from specified hists and bins

    bins1 and bins2 should be in the form of (x, y, z) where y and z can be left as 0
    """
    assert type(bins1) is tuple and len(bins1) is 3
    assert type(bins2) is tuple and len(bins2) is 3
    stat1 = get_stat_w2(hist1, *bins1)
    stat2 = get_stat_w2(hist2, *bins2)
    return stat1 + stat2


def rebin_hist(hist, new_binning, axis='x'):
    """
    Redo the binning of the hist and returns: rebinned_hist, hist_template

    new_binning = list of bin edges

    WARNING: doesn't assert that the edges of the new binning matches the old one.
    """
    assert axis in ['x', 'y', 'z']

    x_binning = [ hist.GetBinLowEdge(i) for i in range(1, hist.GetNbinsX() + 2) ]
    y_binning = [ hist.GetYaxis().GetBinLowEdge(i) for i in range(1, hist.GetNbinsY() + 2) ]
    z_binning = [ hist.GetZaxis().GetBinLowEdge(i) for i in range(1, hist.GetNbinsZ() + 2) ]
    if axis is 'x':
        x_binning = new_binning
    elif axis is 'y':
        y_binning = new_binning
    elif axis is 'z':
        z_binning = new_binning
    else:
        print "ERROR! Can only rebin x, y or z axis"

    if hist.DIM == 1:
        new_hist_template = Hist(x_binning)
    elif hist.DIM == 2:
        new_hist_template = Hist2D(x_binning, y_binning)
    elif hist.DIM == 3:
        new_hist_template = Hist3D(x_binning, y_binning, z_binning)

    # Save the stats of the histogram
    stat_array = array.array('d', [0.] * 10)
    hist.GetStats(stat_array)
    entries = hist.GetEntries()

    new_hist = new_hist_template.Clone()
    new_hist.systematics = {}
    for sys_term in hist.systematics:
        new_hist.systematics[sys_term] = new_hist_template.Clone()

    # Use TH1.FindBin to find out where the bins should be merged into
    for x in range(1, hist.GetNbinsX()+1):
        new_x = new_hist.FindBin(hist.GetBinCenter(x))
        for y in range(1, hist.GetNbinsY()+1):
            new_y = new_hist.GetYaxis().FindBin(hist.GetYaxis().GetBinCenter(y))
            for z in range(1, hist.GetNbinsZ()+1):
                new_z = new_hist.GetZaxis().FindBin(hist.GetZaxis().GetBinCenter(z))
                v = hist.GetBinContent(x, y, z)
                new_v = new_hist.GetBinContent(new_x, new_y, new_z)
                new_hist.SetBinContent(new_x, new_y, new_z, v + new_v)
                comb_w2N = add_stat_w2( hist, (x, y, z), new_hist, (new_x, new_y, new_z) )
                set_stat_w2(new_hist, comb_w2N, new_x, new_y, new_z)

                # Rebin the systematics histograms, too
                for sys_term in hist.systematics:
                    v = hist.systematics[sys_term].GetBinContent(x, y, z)
                    new_v = new_hist.systematics[sys_term].GetBinContent(new_x, new_y, new_z)
                    new_hist.systematics[sys_term].SetBinContent(new_x, new_y, new_z, v + new_v)
                    # WARNING: stats completely ignored in systematics histogram for now.

    # Restores the stats of the NOMINAL histogram
    new_hist.SetEntries(entries)
    new_hist.PutStats(stat_array)
    return new_hist, new_hist_template


def significance(sig_hist, bkg_hist, xbinrange=None, ybinrange=None, zbinrange=None):
    """
    Calculates the significance based on this:
    https://indico.cern.ch/getFile.py/access?contribId=0&resId=0&materialId=slides&confId=250453

    xbinrange, ybinrange, zbinrange = list of bin indices to calculate significance
        - if None or empty list, then run over all bins
        - if index is a tuple, then treat bins in tuple as combined. For e.g.
            xbinrange = [ 0, (1, 2, 3, 4), 5, 6, 7, 8 ]
                --> only run on bins 0 to 8, however, merge bins 1-4

    IMPORTANT: the binranges are bin indices, not bin edges. This is potentially
    confusing, as in the function "rebin_hist()", the input is the latter. Will probably
    need to decide eventually which system to stick to.

    Note that bkg_hist.systematics must be set to at least {}!
    """
    assert hasattr(bkg_hist, 'systematics') and type(bkg_hist.systematics) is dict
    if xbinrange:
        assert type(xbinrange) is list
        x_axis = xbinrange
    else:
        x_axis = range(1, sig_hist.GetNbinsX()+1)
    if ybinrange:
        assert type(ybinrange) is list
        y_axis = ybinrange
    else:
        y_axis = range(1, sig_hist.GetNbinsY()+1)
    if zbinrange:
        assert type(zbinrange) is list
        z_axis = zbinrange
    else:
        z_axis = range(1, sig_hist.GetNbinsZ()+1)

    s = 0.
    for i in x_axis:
        for j in y_axis:
            for k in z_axis:
                if type(i) is int:
                    i = (i,)
                if type(j) is int:
                    j = (j,)
                if type(k) is int:
                    k = (k,)

                sig = 0.
                bkg = 0.
                syst = 0.
                # Merge bins as necessary
                for x in i:
                    for y in j:
                        for z in k:
                            sig += sig_hist.GetBinContent(x, y, z)
                            this_bkg = bkg_hist.GetBinContent(x, y, z)
                            bkg += this_bkg
                            if sig > 0 and bkg > 0:
                                syst += get_stat_w2(sig_hist, x, y, z)
                                syst += get_stat_w2(bkg_hist, x, y, z)
                                for sys_term in bkg_hist.systematics:
                                    syst += (this_bkg - bkg_hist.systematics[sys_term].GetBinContent(x, y, z)) ** 2.

                if sig > 0 and bkg > 0:
                    s += sig**2. / (bkg + syst)
    return math.sqrt(s)


def optimize_binning(sig_hist, bkg_hist, starting_point='fine'):
    """
    Searches for the best uneven binning. starting_point can be 'fine' or 'merged'

    Starting point = 'fine':
        1. For each adjacent bin-pair in all axes, find the bin-pair merge that gives the
           best improvement in significance.
        2. If this 'improvement' is not negative, then merge it and repeat from (1).

    Starting point = 'merged':
        1. Start from a single partition (all bins merged)
        2. Try splitting the partition by exactly half, in each axes individually, and
           also in all combinations.
        3. Do the split if improvement in significance is observed
        4. Repeat from step (2) for the two newly split partitions

    Note that bkg_hist.systematics must be set to at least {}!

    Returns the optimized sig_hist, bkg_hist and hist_template. hist_template is None if nothing is changed.
    """
    assert hasattr(bkg_hist, 'systematics') and type(bkg_hist.systematics) is dict

    original_s = significance(sig_hist, bkg_hist)
    current_template = None
    count = 0

    if starting_point == 'fine':
        current_binning = None
        current_binning_axis = None
        while count < 1e6:
            count += 1
            best_s = -999.
            best_binning_index = None
            best_binning_axis = None

            for x in range(1, sig_hist.GetNbinsX()):
                before_s = significance(sig_hist, bkg_hist, [x, x+1])
                after_s = significance(sig_hist, bkg_hist, [(x, x+1,)])
                if after_s - before_s > best_s:
                    best_binning_index = x
                    best_binning_axis = 'x'
                    best_s = after_s - before_s

            for y in range(1, sig_hist.GetNbinsY()):
                before_s = significance(sig_hist, bkg_hist, None, [y, y+1])
                after_s = significance(sig_hist, bkg_hist, None, [(y, y+1,)])
                if after_s - before_s > best_s:
                    best_binning_index = y
                    best_binning_axis = 'y'
                    best_s = after_s - before_s

            for z in range(1, sig_hist.GetNbinsZ()):
                before_s = significance(sig_hist, bkg_hist, None, None, [z, z+1])
                after_s = significance(sig_hist, bkg_hist, None, None, [(z, z+1,)])
                if after_s - before_s > best_s:
                    best_binning_index = z
                    best_binning_axis = 'z'
                    best_s = after_s - before_s

            if best_s < 0.:
                break
            current_binning_axis = best_binning_axis
            if current_binning_axis:
                if current_binning_axis == 'x':
                    current_binning = [ sig_hist.GetBinLowEdge(i) for i in range(1, sig_hist.GetNbinsX() + 2) if not i-1==best_binning_index ]
                elif current_binning_axis == 'y':
                    current_binning = [ sig_hist.GetBinLowEdge(i) for i in range(1, sig_hist.GetNbinsY() + 2) if not i-1==best_binning_index ]
                elif current_binning_axis == 'z':
                    current_binning = [ sig_hist.GetBinLowEdge(i) for i in range(1, sig_hist.GetNbinsZ() + 2) if not i-1==best_binning_index ]

                sig_hist, current_template = rebin_hist(sig_hist, current_binning, current_binning_axis)
                bkg_hist, _ = rebin_hist(bkg_hist, current_binning, current_binning_axis)
                new_s = significance(sig_hist, bkg_hist)

    elif starting_point == 'merged':
        def partition_binning(binrange, i):
            if i is None:
                return binrange
            partition = binrange[i]
            mid = len(partition) / 2
            if not mid:
                return None
            new_partition = [ tuple(partition[:mid]), tuple(partition[mid:]) ]
            return binrange[:i] + new_partition + binrange[i+1:]

        xbinrange = [ tuple(range(1, bkg_hist.GetNbinsX()+1)) ]
        ybinrange = [ tuple(range(1, bkg_hist.GetNbinsY()+1)) ]
        zbinrange = [ tuple(range(1, bkg_hist.GetNbinsZ()+1)) ]
        new_s = -999.
        while count < 1e6:
            count += 1
            improved = False
            for x in [None] + range(len(xbinrange)):
                test_xbinrange = partition_binning(xbinrange, x)
                for y in [None] + range(len(ybinrange)):
                    test_ybinrange = partition_binning(ybinrange, y)
                    for z in [None] + range(len(zbinrange)):
                        test_zbinrange = partition_binning(zbinrange, z)
                        if not test_xbinrange or not test_ybinrange or not test_zbinrange:
                            continue
                        test_sig = significance(sig_hist, bkg_hist, test_xbinrange, test_ybinrange, test_zbinrange)
                        if test_sig > new_s:
                            improved = True
                            new_s = test_sig
                            best_xbinrange = test_xbinrange
                            best_ybinrange = test_ybinrange
                            best_zbinrange = test_zbinrange
            if improved:
                xbinrange = best_xbinrange
                ybinrange = best_ybinrange
                zbinrange = best_zbinrange
            else:
                break
        if not len(xbinrange) == bkg_hist.GetNbinsX():
            xbinning = [ bkg_hist.GetBinLowEdge(partition[0]) for partition in xbinrange + [(bkg_hist.GetNbinsX()+1,)] ]
            sig_hist, current_template = rebin_hist(sig_hist, xbinning, 'x')
            bkg_hist, _ = rebin_hist(bkg_hist, xbinning, 'x')
        if not len(ybinrange) == bkg_hist.GetNbinsY():
            ybinning = [ bkg_hist.GetBinLowEdge(partition[0]) for partition in ybinrange + [(bkg_hist.GetNbinsY()+1,)] ]
            sig_hist, current_template = rebin_hist(sig_hist, ybinning, 'y')
            bkg_hist, _ = rebin_hist(bkg_hist, ybinning, 'y')
        if not len(zbinrange) == bkg_hist.GetNbinsZ():
            zbinning = [ bkg_hist.GetBinLowEdge(partition[0]) for partition in zbinrange + [(bkg_hist.GetNbinsZ()+1,)] ]
            sig_hist, current_template = rebin_hist(sig_hist, zbinning, 'z')
            bkg_hist, _ = rebin_hist(bkg_hist, zbinning, 'z')
    else:
        print "No such starting_point", starting_point, "for optimisation algo."
        exit(1)

    if current_template:
        xbinning = [ current_template.GetBinLowEdge(i) for i in range(1, current_template.GetNbinsX() + 2) ]
        ybinning = [ current_template.GetBinLowEdge(i) for i in range(1, current_template.GetNbinsY() + 2) ]
        zbinning = [ current_template.GetBinLowEdge(i) for i in range(1, current_template.GetNbinsZ() + 2) ]
        print "Optimised after", count, "iterations"
        print "Significance:", original_s, "\t-->\t", new_s
        print "x| Number of bins", len(xbinning)-1, "Bins:", xbinning
        print "y| Number of bins", len(ybinning)-1, "Bins:", ybinning
        print "z| Number of bins", len(zbinning)-1, "Bins:", zbinning
    else:
        print "Binning already optimized --- nothing is changed"
    return sig_hist, bkg_hist, current_template


Scores = namedtuple('Scores', [
    'data',
    'data_scores',
    'bkg_scores',
    'all_sig_scores',
    'min_score',
    'max_score',])


def get_scores(clf, category, region, backgrounds,
               data=None, cuts=None,
               mass_points=None, mu=1.,
               systematics=True):

    log.info("getting scores")
    year = backgrounds[0].year

    min_score = float('inf')
    max_score = float('-inf')

    # data scores
    data_scores = None
    if data is not None:
        data_scores, _ = data.scores(
            clf,
            category=category,
            region=region,
            cuts=cuts)
        _min = data_scores.min()
        _max = data_scores.max()
        if _min < min_score:
            min_score = _min
        if _max > max_score:
            max_score = _max

    # background model scores
    bkg_scores = []
    for bkg in backgrounds:
        scores_dict = bkg.scores(
            clf,
            category=category,
            region=region,
            cuts=cuts,
            systematics=systematics,
            systematics_components=bkg.WORKSPACE_SYSTEMATICS)

        for sys_term, (scores, weights) in scores_dict.items():
            if len(scores) == 0:
                continue
            _min = np.min(scores)
            _max = np.max(scores)
            if _min < min_score:
                min_score = _min
            if _max > max_score:
                max_score = _max

        bkg_scores.append((bkg, scores_dict))

    # signal scores
    all_sig_scores = {}
    for mass in Higgs.MASS_POINTS:
        if mass_points is not None and mass not in mass_points:
            continue
        sig_scores = []
        # signal scores
        for mode in Higgs.MODES:
            sig = Higgs(year=year, mode=mode, mass=mass, scale=mu,
                        systematics=systematics)

            scores_dict = sig.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts,
                systematics=systematics,
                systematics_components=sig.WORKSPACE_SYSTEMATICS)

            for sys_term, (scores, weights) in scores_dict.items():
                if len(scores) == 0:
                    continue
                _min = np.min(scores)
                _max = np.max(scores)
                if _min < min_score:
                    min_score = _min
                if _max > max_score:
                    max_score = _max

            sig_scores.append((sig, scores_dict))
        all_sig_scores[mass] = sig_scores

    min_score -= 1e-5
    max_score += 1e-5

    log.info("min score: {0} max score: {1}".format(min_score, max_score))
    return Scores(
        data=data,
        data_scores=data_scores,
        bkg_scores=bkg_scores,
        all_sig_scores=all_sig_scores,
        min_score=min_score,
        max_score=max_score)


def channels(clf, category, region, backgrounds,
             data=None, cuts=None, hist_template=None,
             bins=10, binning='constant',
             mass_points=None, mu=1.,
             systematics=True,
             unblind=False,
             hybrid_data=False):
    """
    Return a HistFactory Channel for each mass hypothesis
    """
    log.info("constructing channels")
    channels = dict()

    scores_obj = get_scores(clf, category, region, backgrounds,
                            data=data, cuts=cuts, mass_points=mass_points,
                            mu=mu, systematics=systematics)

    data_scores = scores_obj.data_scores
    bkg_scores = scores_obj.bkg_scores
    all_sig_scores = scores_obj.all_sig_scores
    min_score = scores_obj.min_score
    max_score = scores_obj.max_score

    hist_template = Hist(bins, min_score, max_score)

    # signal scores
    for mass in Higgs.MASS_POINTS:
        if mass_points is not None and mass not in mass_points:
            continue
        log.info('=' * 20)
        log.info("%d GeV mass hypothesis" % mass)

        # create HistFactory samples
        sig_samples = []
        for s, scores in all_sig_scores[mass]:
            sample = s.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=scores,
                suffix='_%d' % mass)
            sig_samples.append(sample)

        bkg_samples = []
        for s, scores in bkg_scores:
            sample = s.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=scores,
                suffix='_%d' % mass)
            bkg_samples.append(sample)

        data_sample = None
        if data_scores is not None:
            max_score = None
            if not unblind:
                sig_hist = sum([histogram_scores(hist_template, scores)
                                for s, scores in sig_scores])
                max_score = efficiency_cut(sig_hist, 0.3)
            data_sample = data.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=data_scores,
                max_score=max_score,
                suffix='_%d' % mass)
            if not unblind and hybrid_data:
                # blinded bins filled with S+B, for limit/p0 plots
                # Swagato:
                # We have to make 2 kinds of expected sensitivity plots:
                # blinded sensitivity and unblinded sensitivity.
                # For the first one pure AsimovData is used, for second one I
                # suggest to use Hybrid, because the profiled NP's are not
                # always at 0 pull.
                pass

        # create channel for this mass point
        channel = histfactory.make_channel(
            "%s_%d" % (category.name, mass),
            bkg_samples + sig_samples,
            data=data_sample)
        channels[mass] = channel

    return channels


def optimized_channels(clf, category, region, backgrounds,
                       data=None, cuts=None, mass_points=None, mu=1.,
                       systematics=True, lumi_rel_error=0.,
                       algo='EvenBinningByLimit'):
    """
    Return optimally binned HistFactory Channels for each mass hypothesis

    Determine the number of bins that yields the best limit at the 125 GeV mass
    hypothesis. Then construct and return the channels for all requested mass
    hypotheses.

    algos: EvenBinningByLimit, UnevenBinningBySignificance
    """
    log.info("constructing optimized channels")

    scores_obj = get_scores(clf, category, region, backgrounds,
                            data=data, cuts=cuts, mass_points=mass_points,
                            mu=mu, systematics=systematics)

    data_scores = scores_obj.data_scores
    bkg_scores = scores_obj.bkg_scores
    all_sig_scores = scores_obj.all_sig_scores
    min_score = scores_obj.min_score
    max_score = scores_obj.max_score

    sig_scores = all_sig_scores[125]

    best_hist_template = None
    if algo == 'EvenBinningByLimit':
        limit_hists = []
        best_limit = float('inf')
        best_nbins = 0
        nbins_range = xrange(2, 50)

        for nbins in nbins_range:

            hist_template = Hist(nbins, min_score, max_score)

            # create HistFactory samples
            samples = []
            for s, scores in bkg_scores + sig_scores:
                sample = s.get_histfactory_sample(
                    hist_template, clf,
                    category, region,
                    cuts=cuts, scores=scores)
                samples.append(sample)

            data_sample = None
            if data is not None:
                data_sample = data.get_histfactory_sample(
                    hist_template, clf,
                    category, region,
                    cuts=cuts, scores=data_scores)

            # create channel for this mass point
            channel = histfactory.make_channel(
                "%s_%d" % (category.name, 125),
                samples, data=data_sample)

            # get limit
            limit_hist = get_limit(channel,
                lumi_rel_error=lumi_rel_error)
            limit_hist.SetName("%s_%d_%d" % (category, 125, nbins))

            # is this better than the best limit so far?
            hist_dict = hist_to_dict(limit_hist)
            limit_hists.append(hist_dict)
            if hist_dict['Expected'] < best_limit:
                best_limit = hist_dict['Expected']
                best_nbins = nbins

        # plot limit vs nbins
        fig = plt.figure()
        ax = fig.add_subplot(111)
        central_values = np.array([h['Expected'] for h in limit_hists])
        high_values_1sig = np.array([h['+1sigma'] for h in limit_hists])
        low_values_1sig = np.array([h['-1sigma'] for h in limit_hists])
        high_values_2sig = np.array([h['+2sigma'] for h in limit_hists])
        low_values_2sig = np.array([h['-2sigma'] for h in limit_hists])
        plt.plot(nbins_range, central_values, 'k-')
        plt.fill_between(nbins_range, low_values_2sig, high_values_2sig,
            linewidth=0, facecolor='yellow')
        plt.fill_between(nbins_range, low_values_1sig, high_values_1sig,
            linewidth=0, facecolor='green')
        plt.xlim(nbins_range[0], nbins_range[-1])
        plt.xlabel("Number of Bins")
        plt.ylabel("Limit")
        plt.grid(True)
        plt.text(.5, .8, "Best limit of %.2f at %d bins" % (best_limit, best_nbins),
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform = ax.transAxes,
                 fontsize=20)
        plt.savefig('category_%s_limit_vs_nbins.png' % category.name)

    elif algo == 'UnevenBinningBySignificance':
        #hist_template = Hist(200, min_score, max_score)
        hist_template = Hist(200, -1.0, 1.0)

        sig_hist = hist_template.Clone(title='Signal')
        sig_hist.systematics = {}
        for sig, scores_dict in sig_scores:
            scores, weight = scores_dict['NOMINAL']
            sig_hist.fill_array(scores, weight)
            for sys_term in scores_dict.keys():
                if sys_term == 'NOMINAL':
                    continue
                if not sys_term in sig_hist.systematics:
                    sys_hist = hist_template.Clone()
                    sig_hist.systematics[sys_term] = sys_hist
                else:
                    sys_hist = sig_hist.systematics[sys_term]
                scores, weight = scores_dict[sys_term]
                sys_hist.fill_array(scores, weight)

        bkg_hist = hist_template.Clone(title='Background')
        bkg_hist.systematics = {}
        for bkg, scores_dict in bkg_scores:
            scores, weight = scores_dict['NOMINAL']
            bkg_hist.fill_array(scores, weight)
            for sys_term in scores_dict.keys():
                if sys_term == 'NOMINAL':
                    continue
                if not sys_term in bkg_hist.systematics:
                    sys_hist = hist_template.Clone()
                    bkg_hist.systematics[sys_term] = sys_hist
                else:
                    sys_hist = bkg_hist.systematics[sys_term]
                scores, weight = scores_dict[sys_term]
                sys_hist.fill_array(scores, weight)

        print "SIG entries:", sig_hist.GetEntries()
        print "BKG entries:", bkg_hist.GetEntries()
        sig_hist, bkg_hist, best_hist_template = optimize_binning(sig_hist, bkg_hist,
                #starting_point='fine'
                starting_point='merged'
            )
        if best_hist_template is None:
            best_hist_template = hist_template
        #raw_input("Hit enter to continue...")
    else:
        print "ERROR: binning optimisation algo %s not in list!" % algo
        exit(1)

    hist_template = best_hist_template
    channels = dict()

    # now use the optimal binning and construct channels for all requested mass
    # hypotheses
    for mass in Higgs.MASS_POINTS:
        if mass_points is not None and mass not in mass_points:
            continue
        log.info('=' * 20)
        log.info("%d GeV mass hypothesis" % mass)

        sig_scores = all_sig_scores[mass]

        # create HistFactory samples
        samples = []
        for s, scores in bkg_scores + sig_scores:
            sample = s.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=scores,
                suffix='_%d' % mass)
            samples.append(sample)

        data_sample = None
        if data_scores is not None:
            data_sample = data.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=data_scores,
                suffix='_%d' % mass)

        # create channel for this mass point
        channel = histfactory.make_channel(
            "%s_%d" % (category.name, mass),
            samples, data=data_sample)

        channels[mass] = channel
    return channels


from rootpy.utils.silence import silence_sout_serr


def get_limit(channels,
          unblind=False,
          lumi=1.,
          lumi_rel_error=0.,
          POI='SigXsecOverSM'):

    with silence_sout_serr():
        workspace, _ = histfactory.make_workspace('higgs', channels,
            lumi=lumi,
            lumi_rel_error=lumi_rel_error,
            POI=POI,
            silence=True)
    return get_limit_workspace(workspace, unblind=unblind)


def get_limit_workspace(workspace, unblind=False, verbose=False):

    calculator = AsymptoticsCLs(workspace, verbose)
    hist = asrootpy(calculator.run('ModelConfig', 'obsData', 'asimovData'))
    hist.SetName('%s_limit' % workspace.GetName())
    return hist


def get_significance_workspace(workspace, unblind=False, verbose=False):

    hist = asrootpy(runSig(workspace))
    hist.SetName('%s_significance' % workspace.GetName())
    return hist
