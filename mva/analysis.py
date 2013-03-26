from . import samples
from .norm import cache as norm_cache


class Analysis(object):

    def __init__(self, year, category,
                 systematics=False,
                 use_embedding=False,
                 fit_param='TRACK'):

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
                year=year,
                markersize=1.2)

        self.higgs_125 = samples.Higgs(
                year=year,
                mass=125,
                systematics=systematics,
                linecolor='red')

        # QCD shape region SS or !OS
        self.qcd = samples.QCD(
            data=self.data,
            mc=[self.others, self.ztautau],
            shape_region=category.qcd_shape_region)

        self.qcd.scale = 1.
        self.ztautau.scale = 1.

        self.fit_param = fit_param
        norm_cache.qcd_ztautau_norm(
            year=year,
            ztautau=self.ztautau,
            qcd=self.qcd,
            category=category.name,
            param=fit_param)

        self.backgrounds = [
            self.qcd,
            self.others,
            self.ztautau,
        ]
