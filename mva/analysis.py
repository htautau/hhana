# stdlib imports
import random
from collections import namedtuple

# numpy imports
import numpy as np

# rootpy imports
from rootpy.stats import histfactory
from rootpy.plotting import Hist

# root_numpy imports
from root_numpy import rec2array

# local imports
from . import samples, log; log = log[__name__]
from . import norm_cache
from .samples import Higgs, Data
from .categories import CATEGORIES
from .classify import histogram_scores, Classifier
from .defaults import (
    TRAIN_FAKES_REGION, FAKES_REGION, TARGET_REGION, NORM_FIELD)
from statstools.utils import efficiency_cut


Scores = namedtuple('Scores', [
    'data',
    'data_scores',
    'bkg_scores',
    'all_sig_scores',
    'min_score',
    'max_score',])


def get_analysis(args, **kwargs):
    if 'year' in kwargs:
        year = kwargs.pop('year')
    else:
        year = args.year
    for name, value in kwargs.items():
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            raise ValueError("invalid Analysis kwarg {0}".format(name))
    analysis = Analysis(
        year=year,
        systematics=args.systematics,
        use_embedding=args.embedding,
        target_region=args.target_region,
        fakes_region=args.fakes_region,
        decouple_qcd_shape=args.decouple_qcd_shape,
        constrain_norms=args.constrain_norms,
        qcd_shape_systematic=args.qcd_shape_systematic,
        random_mu=args.random_mu,
        mu=args.mu,
        suffix=args.suffix)
    return analysis


class Analysis(object):

    def __init__(self, year,
                 systematics=False,
                 use_embedding=True,
                 target_region=TARGET_REGION,
                 fakes_region=FAKES_REGION,
                 decouple_qcd_shape=True,
                 qcd_workspace_norm=None,
                 ztt_workspace_norm=None,
                 constrain_norms=False,
                 qcd_shape_systematic=True,
                 random_mu=False,
                 mu=1.,
                 suffix=None,
                 norm_field=NORM_FIELD):
        self.year = year
        self.systematics = systematics
        self.use_embedding = use_embedding
        self.target_region = target_region
        self.fakes_region = fakes_region
        self.suffix = suffix
        self.norm_field = norm_field

        if use_embedding:
            log.info("Using embedded Ztautau")
            self.ztautau = samples.Embedded_Ztautau(
                year=year,
                systematics=systematics,
                workspace_norm=ztt_workspace_norm,
                constrain_norm=constrain_norms,
                color='#00A3FF')
        else:
            log.info("Using ALPGEN Ztautau")
            self.ztautau = samples.MC_Ztautau(
                year=year,
                systematics=systematics,
                workspace_norm=ztt_workspace_norm,
                constrain_norm=constrain_norms,
                color='#00A3FF')

        self.others = samples.Others(
            year=year,
            systematics=systematics,
            color='#8A0F0F')

        if random_mu:
            log.info("using a random mu (signal strength)")
            self.mu = random.uniform(10, 1000)
        else:
            log.info("using a mu (signal strength) of {0:.1f}".format(mu))
            self.mu = mu

        self.data = samples.Data(year=year,
            markersize=1.2,
            linewidth=1)

        self.higgs_125 = samples.Higgs(
            year=year,
            mass=125,
            systematics=systematics,
            linecolor='red',
            linewidth=2,
            linestyle='dashed',
            scale=self.mu)

        # QCD shape region SS or !OS
        self.qcd = samples.QCD(
            data=self.data,
            mc=[self.ztautau, self.others],
            shape_region=fakes_region,
            decouple_shape=decouple_qcd_shape,
            workspace_norm=qcd_workspace_norm,
            constrain_norm=constrain_norms,
            shape_systematic=qcd_shape_systematic,
            color='#00FF00')

        self.qcd.scale = 1.
        self.ztautau.scale = 1.

        self.backgrounds = [
            self.qcd,
            self.others,
            self.ztautau,
        ]

        self.signals = self.get_signals(125)

    def get_signals(self, mass=125, mode=None, scale_125=False):
        signals = []
        if not isinstance(mass, list):
            mass = [mass]
        if scale_125:
            events_125 = self.higgs_125.events()[1].value
        if mode == 'combined':
            for m in mass:
                s = samples.Higgs(
                    year=self.year,
                    mass=m,
                    systematics=self.systematics,
                    scale=self.mu,
                    linecolor='red',
                    linewidth=2,
                    linestyle='solid')
                if m != 125 and scale_125:
                    log.warning("SCALING SIGNAL TO 125")
                    log.info(str(s.mass))
                    sf = events_125 / s.events()[1].value
                    log.info(str(sf))
                    s.scale *= sf
                signals.append(s)
            return signals
        elif mode == 'workspace':
            for m in mass:
                if m != 125 and scale_125:
                    curr_events = samples.Higgs(
                        year=self.year,
                        mass=m,
                        systematics=False,
                        scale=self.mu).events()[1].value
                    log.warning("SCALING SIGNAL TO 125")
                    sf = events_125 / curr_events
                    log.info(str(sf))
                for mode in samples.Higgs.MODES:
                    s = samples.Higgs(
                        year=self.year,
                        mode=mode,
                        mass=m,
                        systematics=self.systematics,
                        scale=self.mu)
                    if m != 125 and scale_125:
                        log.warning("SCALING SIGNAL TO 125")
                        log.info(str(s.mass))
                        s.scale *= sf
                    signals.append(s)
        elif mode is None:
            for m in mass:
                for modes in samples.Higgs.MODES_COMBINED:
                    signals.append(samples.Higgs(
                        year=self.year,
                        modes=modes,
                        mass=m,
                        systematics=self.systematics,
                        scale=self.mu))
        elif isinstance(mode, (list, tuple)):
            for _mass in mass:
                for _mode in mode:
                    signals.append(samples.Higgs(
                        year=self.year,
                        mass=_mass,
                        mode=_mode,
                        systematics=self.systematics,
                        scale=self.mu))
        else:
            for m in mass:
                signals.append(samples.Higgs(
                    year=self.year,
                    mass=m,
                    mode=mode,
                    systematics=self.systematics,
                    scale=self.mu))
        return signals

    def normalize(self, category):
        norm_cache.qcd_ztautau_norm(
            ztautau=self.ztautau,
            qcd=self.qcd,
            category=category,
            param=self.norm_field)
        return self

    def iter_categories(self, *definitions, **kwargs):
        names = kwargs.pop('names', None)
        for definition in definitions:
            for category in CATEGORIES[definition]:
                if names is not None and category.name not in names:
                    continue
                log.info("")
                log.info("=" * 40)
                log.info("%s category" % category.name)
                log.info("=" * 40)
                log.info("Cuts: %s" % self.ztautau.cuts(category, self.target_region))
                log.info("Weights: %s" % (', '.join(map(str, self.ztautau.weights('NOMINAL')))))
                self.normalize(category)
                yield category

    def get_suffix(self, clf=False, year=True):
        if clf:
            output_suffix = '_%s' % TRAIN_FAKES_REGION
        else:
            output_suffix = '_%s' % self.fakes_region
        if self.use_embedding:
            output_suffix += '_ebz'
        else:
            output_suffix += '_mcz'
        if self.suffix:
            output_suffix += '_%s' % self.suffix
        if self.year % 1E3 == 11 and clf:
            # force the use of 2012 clf on 2011
            output_suffix += '_12'
        elif year:
            output_suffix += '_%d' % (self.year % 1000)
        if not clf and not self.systematics:
            output_suffix += '_stat'
        return  output_suffix

    def get_channel(self, hist_template, expr_or_clf, category, region,
                    cuts=None,
                    include_signal=True,
                    mass=125,
                    mode=None,
                    clf=None,
                    min_score=None,
                    max_score=None,
                    systematics=True,
                    no_signal_fixes=False):

        # TODO: implement blinding
        log.info("constructing channels")
        samples = [self.data] + self.backgrounds
        channel_name = '{0}_{1}'.format(category.name, self.year % 1000)
        suffix = None
        if include_signal:
            if isinstance(mass, list):
                suffix = '_' + ('_'.join(map(str, mass)))
            else:
                suffix = '_%d' % mass
            channel_name += suffix
            samples += self.get_signals(mass, mode)

        # create HistFactory samples
        histfactory_samples = []
        for s in samples:
            sample = s.get_histfactory_sample(
                hist_template, expr_or_clf,
                category, region,
                cuts=cuts,
                clf=clf,
                min_score=min_score,
                max_score=max_score,
                suffix=suffix if not isinstance(s, Higgs) else None,
                no_signal_fixes=no_signal_fixes,
                systematics=systematics)
            histfactory_samples.append(sample)

        # create channel for this mass point
        return histfactory.make_channel(
            channel_name, histfactory_samples[1:], data=histfactory_samples[0])

    def get_channel_array(self, vars,
                          category, region,
                          cuts=None,
                          include_signal=True,
                          mass=125,
                          mode=None,
                          scale_125=False,
                          clf=None,
                          min_score=None,
                          max_score=None,
                          weighted=True,
                          templates=None,
                          field_scale=None,
                          weight_hist=None,
                          systematics=True,
                          no_signal_fixes=False,
                          bootstrap_data=False,
                          ravel=True,
                          uniform=False,
                          hybrid_data=None):
        """
        Return a dictionnary of histfactory channels for different variables
        (i.e. {'MMC_MASS':channel1, ...}).

        Parameters
        ----------
        vars: dict
            dictionary of histograms (i.e. {'MMC_MASS':hist_template, ...}
        category: Category
            analysis category (see mva/categories/*)
        region: str
            analysis region (i.e 'OS_ISOL', ...)
        cuts : str or Cut
            additional cuts that could be place when requesting the channel
            array (See mva/categories/common.py for examples)
        hybrid_data : dict
            if specified, it is a dictionary mapping the vars key to a tuple
            specifying the range to be replaced by s+b prediction.
        """
        # TODO: implement blinding
        log.info("constructing channels")
        samples = [self.data]+self.backgrounds
        channel_name = '{0}_{1}'.format(category.name, self.year % 1000)
        suffix = None
        if include_signal:
            if isinstance(mass, list):
                suffix = '_' + ('_'.join(map(str, mass)))
            else:
                suffix = '_%d' % mass
            channel_name += suffix
            samples += self.get_signals(mass, mode, scale_125=scale_125)

        # create HistFactory samples
        histfactory_samples = []
        for s in samples:
            field_hist, _ = s.get_field_hist(vars, category, templates=templates)
            field_sample = s.get_histfactory_sample_array(
                field_hist,
                category, region,
                cuts=cuts,
                clf=clf,
                min_score=min_score,
                max_score=max_score,
                weighted=weighted,
                field_scale=field_scale,
                weight_hist=weight_hist,
                systematics=systematics,
                suffix=suffix if not isinstance(s, Higgs) else None,
                no_signal_fixes=no_signal_fixes,
                bootstrap_data=bootstrap_data,
                ravel=ravel,
                uniform=uniform)
            histfactory_samples.append(field_sample)

        field_channels = {}
        for field in vars.keys():
            # create channel for this mass point
            channel = histfactory.make_channel(
                channel_name + '_{0}'.format(field),
                [s[field] for s in histfactory_samples[1:]],
                data=histfactory_samples[0][field])
            # implement hybrid data if requested
            if isinstance(hybrid_data, dict):
                log.info('Hybrid data has been requested')
                if field in hybrid_data.keys():
                    if isinstance(hybrid_data[field], (list, tuple)):
                        log.info('Hybrid data: replacing data by s+b prediction for {0} in range {1}'.format(field, hybrid_data[field]))
                        if len(hybrid_data[field])!=2:
                            log.error('Hybrid data: Need to specify a range with only two edged')
                        # Get the range of bins to be replaced (add 1 additional bin on both side for safety)
                        (replace_low, replace_high) = (hybrid_data[field][0], hybrid_data[field][1])
                        hist_data_template = self.data.get_field_hist(vars, category)
                        log.info('Hybrid data: template binning {0}'.format(list(hist_data_template[0][field].xedges())))
                        replace_bin = (hist_data_template[0][field].FindBin(float(replace_low))-1,
                                       hist_data_template[0][field].FindBin(float(replace_high))+1)
                        total_bkg_sig = sum([s.hist for s in channel.samples])
                        log.info( 'Hybrid data: before --> {0}'.format(list(channel.data.hist.y())))
                        channel.data.hist[replace_bin[0]:replace_bin[1]] = total_bkg_sig[replace_bin[0]:replace_bin[1]]
                        log.info( 'Hybrid data: after --> {0}'.format(list(channel.data.hist.y())))
            field_channels[field] = channel
        return field_channels

    def get_scores(self, clf, category, region, cuts=None,
                   masses=None, mode=None, unblind=False,
                   systematics=True):

        log.info("getting scores")

        min_score = float('inf')
        max_score = float('-inf')

        # data scores
        data_scores = None
        if unblind:
            data_scores, _ = self.data.scores(
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
        for bkg in self.backgrounds:
            scores_dict = bkg.scores(
                clf,
                category=category,
                region=region,
                cuts=cuts,
                systematics=systematics,
                systematics_components=bkg.systematics_components())

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
        if masses is not None:
            for mass in masses:
                # signal scores
                sigs = self.get_signals(mass=mass, mode=mode)
                sig_scores = []
                for sig in sigs:
                    scores_dict = sig.scores(
                        clf,
                        category=category,
                        region=region,
                        cuts=cuts,
                        systematics=systematics,
                        systematics_components=sig.systematics_components())

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

        min_score -= 1e-8
        max_score += 1e-8

        log.info("min score: {0} max score: {1}".format(min_score, max_score))
        return Scores(
            data=self.data,
            data_scores=data_scores,
            bkg_scores=bkg_scores,
            all_sig_scores=all_sig_scores,
            min_score=min_score,
            max_score=max_score)

    def clf_channels(self, clf,
                     category, region,
                     cuts=None,
                     bins=10,
                     mass=None,
                     mode=None,
                     systematics=True,
                     unblind=False,
                     hybrid_data=False,
                     no_signal_fixes=False,
                     uniform=False):
        """
        Return a HistFactory Channel for each mass hypothesis
        """
        log.info("constructing channels")
        channels = dict()

        scores_obj = self.get_scores(
            clf, category, region, cuts=cuts,
            masses=[mass], mode=mode,
            systematics=systematics,
            unblind=unblind)

        data_scores = scores_obj.data_scores
        bkg_scores = scores_obj.bkg_scores
        all_sig_scores = scores_obj.all_sig_scores
        min_score = scores_obj.min_score
        max_score = scores_obj.max_score

        if isinstance(bins, int):
            binning = Hist(bins, min_score, max_score, type='D')
        else: # iterable
            binning = Hist(bins, type='D')

        bkg_samples = []
        for s, scores in bkg_scores:
            hist_template = binning.Clone(
                title=s.label,
                **s.hist_decor)
            sample = s.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=scores,
                systematics=systematics,
                uniform=uniform)
            bkg_samples.append(sample)

        data_sample = None
        if data_scores is not None:
            hist_template = binning.Clone(
                title=self.data.label,
                **self.data.hist_decor)
            data_sample = self.data.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=data_scores,
                uniform=uniform)
            """
            if unblind is False:
                # blind full histogram
                data_sample.hist[:] = (0, 0)
            elif isinstance(unblind, int):
                # blind highest N bins
                data_sample.hist[-(unblind + 1):] = (0, 0)
            elif isinstance(unblind, float):
                # blind above a signal efficiency
                max_unblind_score = efficiency_cut(
                    sum([histogram_scores(hist_template, scores)
                        for s, scores in all_sig_scores[mass]]), unblind)
                blind_bin = hist_template.FindBin(max_unblind_score)
                data_sample.hist[blind_bin:] = (0, 0)
            """

        # create signal HistFactory samples
        sig_samples = []
        for s, scores in all_sig_scores[mass]:
            hist_template = binning.Clone(
                title=s.label,
                **s.hist_decor)
            sample = s.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=scores,
                no_signal_fixes=no_signal_fixes,
                systematics=systematics,
                uniform=uniform)
            sig_samples.append(sample)

        # replace data in blind bins with signal + background
        """
        if hybrid_data and unblind is not True:
            sum_sig_bkg = sum([s.hist for s in (bkg_samples + sig_samples)])
            if unblind is False:
                # replace full hist
                data_sample.hist[:] = sum_sig_bkg[:]
            elif isinstance(unblind, int):
                # replace highest N bins
                bin = -(unblind + 1)
                data_sample.hist[bin:] = sum_sig_bkg[bin:]
            elif isinstance(unblind, float):
                data_sample.hist[blind_bin:] = sum_sig_bkg[blind_bin:]
        """

        # create channel for this mass point
        channel = histfactory.make_channel(
            '{0}_{1}_{2}'.format(category.name, mass, self.year % 1000),
            bkg_samples + sig_samples,
            data=data_sample)

        return scores_obj, channel

    def get_clf(self, category, load=False, swap=False, **kwargs):
        output_suffix = self.get_suffix()
        clf_output_suffix = self.get_suffix(clf=True)
        clf = Classifier(
            fields=category.features,
            category=category,
            region=self.target_region,
            clf_output_suffix=clf_output_suffix,
            output_suffix=output_suffix,
            **kwargs)
        if load and not clf.load(swap=swap):
            raise RuntimeError("train BDTs before requesting scores")
        return clf

    def records(self, category, region, cuts=None, fields=None,
                clf=None,
                clf_name='classifier',
                include_weight=True,
                systematic='NOMINAL'):
        bkg_recs = {}
        for bkg in self.backgrounds:
            bkg_recs[bkg] = bkg.merged_records(
                category, region,
                cuts=cuts, fields=fields,
                include_weight=include_weight,
                clf=clf,
                clf_name=clf_name,
                systematic=systematic)
        sig_recs = {}
        sig_recs[self.higgs_125] = self.higgs_125.merged_records(
            category, region,
            cuts=cuts, fields=fields,
            include_weight=include_weight,
            clf=clf,
            clf_name=clf_name,
            systematic=systematic)
        return bkg_recs, sig_recs

    def arrays(self, category, region, cuts=None, fields=None,
               clf=None,
               clf_name='classifier',
               include_weight=True,
               systematic='NOMINAL'):
        bkg_recs, sig_recs = self.records(
            category, region, cuts=cuts, fields=fields,
            clf=clf,
            clf_name=clf_name,
            include_weight=include_weight,
            systematic=systematic)
        bkg_arrs = {}
        sig_arrs = {}
        for b, rec in bkg_recs.items():
            bkg_arrs[b] = rec2array(rec)
        for s, rec in sig_recs.items():
            sig_arrs[s] = rec2array(rec)
        return bkg_arrs, sig_arrs

    def make_var_channels(self, hist_template, expr, categories, region,
                          include_signal=False, masses=None,
                          systematics=False):
        if not include_signal:
            channels = []
            for category in categories:
                parent_category = category.get_parent()
                # apply normalization
                self.normalize(parent_category)
                # clf = analysis.get_clf(parent_category, load=True)
                contr = self.get_channel(hist_template, expr,
                    category=category,
                    region=region,
                    #clf=clf,
                    #cuts=signal_region,
                    include_signal=False,
                    systematics=systematics)
                channels.append(contr)
        else:
            channels = {}
            for category in categories:
                parent_category = category.get_parent()
                # apply normalization
                self.normalize(parent_category)
                # clf = analysis.get_clf(parent_category, load=True)
                for mass in masses:
                    contr = self.get_channel(hist_template, expr,
                        category=category,
                        region=region,
                        #clf=clf,
                        #cuts=signal_region,
                        include_signal=True,
                        mass=mass,
                        mode='workspace',
                        systematics=systematics)
                    if mass not in channels:
                        channels[mass] = {}
                    channels[mass][category.name] = contr
        return channels
