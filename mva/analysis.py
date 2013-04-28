from . import samples
from .norm import cache as norm_cache


class Analysis(object):

    def __init__(self, year, category,
                 systematics=False,
                 use_embedding=False,
                 fit_param='TRACK',
                 qcd_shape_region='SS_TRK'):

        self.year = year
        self.category = category
        self.systematics = systematics
        self.use_embedding = use_embedding

        if use_embedding:
            self.ztautau = samples.Embedded_Ztautau(
                    year=year,
                    systematics=systematics)
        else:
            self.ztautau = samples.MC_Ztautau(
                    year=year,
                    systematics=systematics)

        self.others = samples.Others(
                year=year,
                systematics=systematics)

        self.data = samples.Data(
                year=year)

        self.higgs_125 = samples.Higgs(
                year=year,
                mass=125,
                systematics=systematics,
                linecolor='red')

        # QCD shape region SS or !OS
        self.qcd = samples.QCD(
            data=self.data,
            mc=[self.others, self.ztautau],
            shape_region=qcd_shape_region)

        self.qcd.scale = 1.
        self.ztautau.scale = 1.

        self.fit_param = fit_param
        norm_cache.qcd_ztautau_norm(
            ztautau=self.ztautau,
            qcd=self.qcd,
            category=category,
            param=fit_param,
            shape_region=qcd_shape_region)

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
                systematics=systematics))
