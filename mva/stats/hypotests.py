from . import log; log = log[__name__]
import numpy as np
from rootpy import asrootpy
from rootpy.plotting import Hist
from .asymptotics import AsymptoticsCLs
from ..samples import Higgs
from ..plotting import significance
from . import histfactory
from .utils import get_safe_template
from ..utils import hist_to_dict


def channels(clf, category, region, backgrounds,
            data=None, cuts=None, hist_template=None,
            bins=10, binning='constant', mass_points=None,
            systematics=True):
    """
    Return a HistFactory Channel for each mass hypothesis
    """
    log.info("constructing channels")
    channels = dict()
    # TODO check for sample compatibility
    year = backgrounds[0].year

    # background model scores
    bkg_scores = []
    for bkg in backgrounds:
        scores_dict = bkg.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts)
        bkg_scores.append((bkg, scores_dict))

    # data scores
    if data is not None:
        data_scores, _ = data.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts)

    # signal scores
    for mass in Higgs.MASS_POINTS:
        if mass_points is not None and mass not in mass_points:
            continue
        log.info('=' * 20)
        log.info("%d GeV mass hypothesis" % mass)
        # create separate signal. background and data histograms for each
        # mass hypothesis since the binning is optimized for each mass
        # individually.
        # The binning is determined by first locating the BDT cut value at
        # which the signal significance is maximized (S / sqrt(B)).
        # Everything above that cut is put in one bin. Everything below that
        # cut is put into N variable width bins such that the background is
        # flat.
        sig_scores = []
        # signal scores
        for mode in Higgs.MODES:
            sig = Higgs(year=year, mode=mode, mass=mass,
                    systematics=systematics)

            scores_dict = sig.scores(
                    clf,
                    category=category,
                    region=region,
                    cuts=cuts)
            sig_scores.append((sig, scores_dict))

        # get templates that are safe for hypo tests
        hist_template = get_safe_template(binning, bins, bkg_scores, sig_scores)

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
        channel = histfactory.make_channel("%s_%d" % (category.name, mass),
                               samples, data=data_sample)
        channels[mass] = channel
    return channels


def optimized_channels(clf, category, region, backgrounds,
            data=None, cuts=None, mass_points=None,
            systematics=True, lumi_rel_error=0.):
    """
    Return optimally binned HistFactory Channels for each mass hypothesis

    Determine the number of bins that yields the best limit at the 125 GeV mass
    hypothesis. Then construct and return the channels for all requested mass
    hypotheses.
    """
    log.info("constructing optimized channels")
    # TODO check for sample compatibility
    year = backgrounds[0].year

    # background model scores
    bkg_scores = []
    for bkg in backgrounds:
        scores_dict = bkg.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts)
        bkg_scores.append((bkg, scores_dict))

    # 125 GeV signal scores
    sig_scores = []
    for mode in Higgs.MODES:
        sig = Higgs(year=year, mode=mode, mass=125,
                    systematics=systematics)
        scores_dict = sig.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts)
        sig_scores.append((sig, scores_dict))

    # data scores
    if data is not None:
        data_scores, _ = data.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts)
        min_score = data_scores.min()
        max_score = data_scores.max()
    else:
        min_score = float('inf')
        max_score = float('-inf')

    for s, scores_dict in bkg_scores + sig_scores:
        for sys_term, (scores, weights) in scores_dict.items():
            if len(scores) == 0:
                continue
            _min = np.min(scores)
            _max = np.max(scores)
            if _min < min_score:
                min_score = _min
            if _max > max_score:
                max_score = _max

    limit_hists = []
    best_limit = float('inf')
    best_hist_template = None
    for nbins in xrange(2, 21):
        hist_template = Hist(nbins, min_score, max_score)
        # create HistFactory samples
        samples = []
        for s, scores in bkg_scores:
            sample = s.get_histfactory_sample(
                    hist_template, clf,
                    category, region,
                    cuts=cuts, scores=scores,
                    apply_kylefix=True)
            samples.append(sample)

        for s, scores in sig_scores:
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
            best_hist_template = hist_template

    hist_template = best_hist_template
    channels = dict()

    # now use the optimal binning and construct channels for all requested mass
    # hypotheses
    for mass in Higgs.MASS_POINTS:
        if mass_points is not None and mass not in mass_points:
            continue
        log.info('=' * 20)
        log.info("%d GeV mass hypothesis" % mass)
        # create separate signal. background and data histograms for each
        # mass hypothesis since the binning is optimized for each mass
        # individually.
        # The binning is determined by first locating the BDT cut value at
        # which the signal significance is maximized (S / sqrt(B)).
        # Everything above that cut is put in one bin. Everything below that
        # cut is put into N variable width bins such that the background is
        # flat.
        sig_scores = []
        # signal scores
        for mode in Higgs.MODES:
            sig = Higgs(year=year, mode=mode, mass=mass,
                        systematics=systematics)
            scores_dict = sig.scores(
                    clf,
                    category=category,
                    region=region,
                    cuts=cuts)
            sig_scores.append((sig, scores_dict))

        # create HistFactory samples
        samples = []
        for s, scores in bkg_scores:
            sample = s.get_histfactory_sample(
                    hist_template, clf,
                    category, region,
                    cuts=cuts, scores=scores,
                    apply_kylefix=True)
            samples.append(sample)

        for s, scores in sig_scores:
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
                "%s_%d" % (category.name, mass),
                samples, data=data_sample)

        channels[mass] = channel
    return channels


def get_limit(channels,
          unblind=False,
          lumi_rel_error=0.,
          POI='SigXsecOverSM'):

    workspace = histfactory.make_workspace('higgs', channels,
            lumi_rel_error=lumi_rel_error,
            POI=POI)
    return get_limit_workspace(workspace, unblind=unblind)


def get_limit_workspace(workspace, unblind=False):

    calculator = AsymptoticsCLs(workspace)
    hist = asrootpy(calculator.run('ModelConfig', 'obsData', 'asimovData'))
    hist.SetName('%s_limit' % workspace.GetName())
    return hist
