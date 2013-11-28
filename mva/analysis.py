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
from .samples import Higgs, Data
from .norm import cache as norm_cache
from .categories import CATEGORIES
from .stats.utils import efficiency_cut
from .classify import histogram_scores, Classifier
from .np_utils import rec2array


Scores = namedtuple('Scores', [
    'data',
    'data_scores',
    'bkg_scores',
    'all_sig_scores',
    'min_score',
    'max_score',])


def get_analysis(args):
    analysis = Analysis(
        year=args.year,
        systematics=args.systematics,
        use_embedding=args.embedding,
        target_region=args.target_region,
        qcd_shape_region=args.qcd_shape_region,
        decouple_qcd_shape=args.decouple_qcd_shape,
        constrain_norms=args.constrain_norms,
        random_mu=args.random_mu,
        mu=args.mu,
        partition_key='EventNumber', # 'MET_phi_original * 100', or None
        suffix=args.suffix,
        transform=not args.raw_scores,
        mmc=not args.no_mmc,
        mpl=args.mpl)
    return analysis


class Analysis(object):

    def __init__(self, year,
                 systematics=False,
                 use_embedding=True,
                 target_region='OS_TRK',
                 qcd_shape_region='nOS',
                 decouple_qcd_shape=True,
                 qcd_workspace_norm=None,
                 ztt_workspace_norm=None,
                 constrain_norms=False,
                 random_mu=False,
                 mu=1.,
                 partition_key='EventNumber',
                 transform=True,
                 suffix=None,
                 mmc=True,
                 mpl=False):

        self.year = year
        self.systematics = systematics
        self.use_embedding = use_embedding
        self.target_region = target_region
        self.qcd_shape_region = qcd_shape_region
        self.partition_key = partition_key
        self.transform = transform
        self.suffix = suffix
        self.mmc = mmc
        self.mpl = mpl

        if use_embedding:
            log.info("Using embedded Ztautau")
            self.ztautau = samples.Embedded_Ztautau(
                year=year,
                systematics=systematics,
                workspace_norm=ztt_workspace_norm,
                constrain_norm=constrain_norms,
                color='#00A3FF',
                mpl=mpl)
        else:
            log.info("Using ALPGEN Ztautau")
            self.ztautau = samples.MC_Ztautau(
                year=year,
                systematics=systematics,
                workspace_norm=ztt_workspace_norm,
                constrain_norm=constrain_norms,
                color='#00A3FF',
                mpl=mpl)

        self.others = samples.Others(
            year=year,
            systematics=systematics,
            color='#8A0F0F',
            mpl=mpl)

        if random_mu:
            log.info("using a random mu (signal strength)")
            self.mu = random.uniform(10, 1000)
        else:
            log.info("using a mu (signal strength) of {0:.1f}".format(mu))
            self.mu = mu

        self.data = samples.Data(year=year,
            markersize=1.2,
            linewidth=1,
            mpl=mpl)

        self.higgs_125 = samples.Higgs(
            year=year,
            mass=125,
            systematics=systematics,
            linecolor='red',
            linewidth=2,
            linestyle='dashed',
            mpl=mpl,
            scale=self.mu)

        # QCD shape region SS or !OS
        self.qcd = samples.QCD(
            data=self.data,
            mc=[self.ztautau, self.others],
            shape_region=qcd_shape_region,
            decouple_shape=decouple_qcd_shape,
            workspace_norm=qcd_workspace_norm,
            constrain_norm=constrain_norms,
            color='#00FF00',
            mpl=mpl)

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
                    mpl=self.mpl,
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
                        mpl=self.mpl,
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
                        mpl=self.mpl,
                        scale=self.mu))
        else:
            for m in mass:
                signals.append(samples.Higgs(
                    year=self.year,
                    mass=m,
                    mode=mode,
                    systematics=self.systematics,
                    mpl=self.mpl,
                    scale=self.mu))
        return signals

    def normalize(self, category):

        norm_cache.qcd_ztautau_norm(
            ztautau=self.ztautau,
            qcd=self.qcd,
            category=category,
            param='TRACK')

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
                log.info("Weights: %s" % (', '.join(map(str, self.ztautau.get_weight_branches('NOMINAL')))))
                self.normalize(category)
                yield category

    def get_suffix(self, clf=False):

        # "track" here only for historical reasons
        output_suffix = '_trackfit_%s' % self.qcd_shape_region
        if self.use_embedding:
            output_suffix += '_embedding'
        else:
            output_suffix += '_alpgen'
        if not self.mmc:
            output_suffix += '_no_mmc'
        if self.suffix:
            output_suffix += '_%s' % self.suffix
        output_suffix += '_%d' % (self.year % 1E3)
        if not clf and not self.systematics:
            output_suffix += '_statsonly'
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
        channel_name = category.name
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
                no_signal_fixes=no_signal_fixes)
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
                          bootstrap_data=False):

        # TODO: implement blinding
        log.info("constructing channels")
        samples = [self.data] + self.backgrounds
        channel_name = category.name
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
            field_hist = s.get_field_hist(vars, category, templates=templates)
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
                bootstrap_data=bootstrap_data)
            histfactory_samples.append(field_sample)

        field_channels = {}
        for field in vars.keys():
            # create channel for this mass point
            channel = histfactory.make_channel(
                channel_name + '_{0}'.format(field),
                [s[field] for s in histfactory_samples[1:]],
                data=histfactory_samples[0][field])
            field_channels[field] = channel
        return field_channels

    def get_scores(self, clf, category, region, cuts=None,
                   mass_points=None, mode=None, unblind=False,
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
        if mass_points is not None:
            for mass in samples.Higgs.MASS_POINTS:
                if mass not in mass_points:
                    continue
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
                     mass_points=None,
                     mode=None,
                     systematics=True,
                     unblind=False,
                     hybrid_data=False,
                     no_signal_fixes=False):
        """
        Return a HistFactory Channel for each mass hypothesis
        """
        log.info("constructing channels")
        channels = dict()

        scores_obj = self.get_scores(
            clf, category, region, cuts=cuts,
            mass_points=mass_points, mode=mode,
            systematics=systematics,
            unblind=unblind)

        data_scores = scores_obj.data_scores
        bkg_scores = scores_obj.bkg_scores
        all_sig_scores = scores_obj.all_sig_scores
        min_score = scores_obj.min_score
        max_score = scores_obj.max_score

        bkg_samples = []
        for s, scores in bkg_scores:
            hist_template = Hist(
                bins, min_score, max_score,
                title=s.label,
                type='D',
                **s.hist_decor)
            sample = s.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=scores)
            bkg_samples.append(sample)

        data_sample = None
        if data_scores is not None:
            max_unblind_score = None
            if isinstance(unblind, float):
                """
                max_unblind_score = min([
                    efficiency_cut(
                        sum([histogram_scores(hist_template, scores)
                             for s, scores in all_sig_scores[mass]]), 0.3)
                        for mass in mass_points])
                """
                max_unblind_score = efficiency_cut(
                    sum([histogram_scores(hist_template, scores)
                         for s, scores in all_sig_scores[125]]), unblind)
            hist_template = Hist(
                bins, min_score, max_score,
                title=self.data.label,
                type='D',
                **self.data.hist_decor)
            data_sample = self.data.get_histfactory_sample(
                hist_template, clf,
                category, region,
                cuts=cuts, scores=data_scores,
                max_score=max_unblind_score)
            if not unblind and hybrid_data:
                # blinded bins filled with S+B, for limit/p0 plots
                # Swagato:
                # We have to make 2 kinds of expected sensitivity plots:
                # blinded sensitivity and unblinded sensitivity.
                # For the first one pure AsimovData is used, for second one I
                # suggest to use Hybrid, because the profiled NP's are not
                # always at 0 pull.
                pass

        if mass_points is None:
            # create channel without signal
            channel = histfactory.make_channel(
                category.name,
                bkg_samples,
                data=data_sample)
            return scores_obj, channel

        # signal scores
        for mass in samples.Higgs.MASS_POINTS:
            if mass not in mass_points:
                continue
            log.info('=' * 20)
            log.info("%d GeV mass hypothesis" % mass)

            # create HistFactory samples
            sig_samples = []
            for s, scores in all_sig_scores[mass]:
                hist_template = Hist(
                    bins, min_score, max_score,
                    title=s.label,
                    type='D',
                    **s.hist_decor)
                sample = s.get_histfactory_sample(
                    hist_template, clf,
                    category, region,
                    cuts=cuts, scores=scores,
                    no_signal_fixes=no_signal_fixes)
                sig_samples.append(sample)

            # create channel for this mass point
            channel = histfactory.make_channel(
                "%s_%d" % (category.name, mass),
                bkg_samples + sig_samples,
                data=data_sample)
            channels[mass] = channel

        return scores_obj, channels

    def get_clf(self, category, load=False, swap=False):

        output_suffix = self.get_suffix()
        clf_output_suffix = self.get_suffix(clf=True)

        clf = Classifier(
            fields=category.features,
            category=category,
            region=self.target_region,
            clf_output_suffix=clf_output_suffix,
            output_suffix=output_suffix,
            partition_key=self.partition_key,
            transform=self.transform,
            mmc=self.mmc)

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
