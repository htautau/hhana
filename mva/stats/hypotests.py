from . import log; log = log[__name__]
import numpy as np
from rootpy.plotting import Hist
from .asymptotics import AsymptoticsCLs
from ..samples import Higgs
from ..plotting import significance
from . import histfactory
from .utils import get_safe_template


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

    if data is not None:
        data_scores, _ = data.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts)

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


def limit(channels,
          unblind=False,
          lumi_rel_error=0.028,
          POI='SigXsecOverSM'):

    if not isinstance(channels, (list, tuple)):
        channels = [channels]
    measurement = histfactory.make_measurement(
            'higgs', '',
            channels,
            lumi_rel_error=lumi_rel_error,
            POI=POI)
    workspace = histfactory.make_model(measurement)
    calculator = AsymptoticsCLs(workspace)
    limit_hist = calculator.run('ModelConfig', 'obsData', 'asimovData')

    return limit_hist
