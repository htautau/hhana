from .sample import Background
from .mc import MC
from . import log


class EWK(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True


class Top(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True


class Diboson(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True


class Others(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True

    def histfactory(self, sample, category, systematics=True):
        if not systematics:
            return
        # https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/HSG4Uncertainties
        # pdf uncertainty
        sample.AddOverallSys('pdf_qq', 0.96, 1.04)
        # QCD scale uncertainty
        sample.AddOverallSys('QCDscale_V', 0.99, 1.01)
