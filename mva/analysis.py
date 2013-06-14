from rootpy.fit import histfactory

from . import samples, log; log = log[__name__]
from .norm import cache as norm_cache


class Analysis(object):

    def __init__(self, year,
                 systematics=False,
                 use_embedding=False,
                 qcd_shape_region='nOS',
                 root=False):

        self.year = year
        self.systematics = systematics
        self.use_embedding = use_embedding
        self.qcd_shape_region = qcd_shape_region
        self.root = root

        if use_embedding:
            log.info("Using embedded Ztautau")
            self.ztautau = samples.Embedded_Ztautau(
                year=year,
                systematics=systematics,
                root=root)
        else:
            log.info("Using ALPGEN Ztautau")
            self.ztautau = samples.MC_Ztautau(
                year=year,
                systematics=systematics,
                root=root)

        self.others = samples.Others(
            year=year,
            systematics=systematics,
            root=root)

        self.data = samples.Data(year=year, root=root)

        self.higgs_125 = samples.Higgs(
            year=year,
            mass=125,
            systematics=systematics,
            linecolor='red',
            linewidth=2,
            linestyle='dashed',
            root=root)

        # QCD shape region SS or !OS
        self.qcd = samples.QCD(
            data=self.data,
            mc=[self.ztautau, self.others],
            shape_region=qcd_shape_region,
            root=root)

        self.qcd.scale = 1.
        self.ztautau.scale = 1.

        self.backgrounds = [
            self.qcd,
            self.others,
            self.ztautau,
        ]

        self.signals = self.get_signals(125)

    def get_signals(self, mass):

        signals = []
        for mode in samples.Higgs.MODES:
            signals.append(samples.Higgs(
                year=self.year,
                mode=mode,
                mass=mass,
                systematics=self.systematics,
                root=self.root))
        return signals

    def normalize(self, category, fit_param='TRACK'):

        norm_cache.qcd_ztautau_norm(
            ztautau=self.ztautau,
            qcd=self.qcd,
            category=category,
            param=fit_param)

    def get_suffix(self, fit_param='TRACK', suffix=None):

        output_suffix = '_%sfit_%s' % (fit_param.lower(), self.qcd_shape_region)
        if self.use_embedding:
            output_suffix += '_embedding'
        else:
            output_suffix += '_alpgen'
        if suffix:
            output_suffix += '_%s' % suffix
        output_suffix += '_%d' % (self.year % 1E3)
        return  output_suffix
        #if not self.systematics:
        #    output_suffix += '_statsonly'

    def get_channel(self, hist_template, expr_or_clf, category, region,
                    cuts=None,
                    include_signal=True,
                    mass=125,
                    clf=None,
                    min_score=None,
                    max_score=None,
                    systematics=True):

        log.info("constructing channels")
        samples = [self.data] + self.backgrounds
        channel_name = category.name
        suffix = None
        if include_signal:
            suffix = '_%d' % mass
            channel_name += suffix
            samples += self.get_signals(mass)

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
                suffix=suffix)
            histfactory_samples.append(sample)

        # create channel for this mass point
        return histfactory.make_channel(
            channel_name, histfactory_samples[1:], data=histfactory_samples[0])
