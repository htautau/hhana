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

        self.signals = []
        for mode in samples.Higgs.MODES:
            self.signals.append(samples.Higgs(
                year=year,
                mode=mode,
                mass=125,
                systematics=systematics,
                root=root))

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
