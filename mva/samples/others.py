# local imports
from .sample import MC, Background
from . import log

from rootpy.tree import Cut


class EWK(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True

    def __init__(self, *args, **kwargs):
        # self.matched = kwargs.pop('matched', True)
        self.matched = kwargs.pop('matched', False)
        super(EWK, self).__init__(*args, **kwargs)

    def cuts(self, *args, **kwargs):
        cut = super(EWK, self).cuts(*args, **kwargs)
        if self.matched:
            # require that at least one tau matches truth
            cut &= Cut('tau1_matched || tau2_matched')
        return cut


class Top(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True

    def __init__(self, *args, **kwargs):
        # self.matched = kwargs.pop('matched', True)
        self.matched = kwargs.pop('matched', False)
        super(Top, self).__init__(*args, **kwargs)

    def cuts(self, *args, **kwargs):
        cut = super(Top, self).cuts(*args, **kwargs)
        if self.matched:
            # require that at least one tau matches truth
            cut &= Cut('tau1_matched || tau2_matched')
        return cut

class Diboson(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True

    def __init__(self, *args, **kwargs):
        self.matched = kwargs.pop('matched', True)
        super(Diboson, self).__init__(*args, **kwargs)

    def cuts(self, *args, **kwargs):
        cut = super(Diboson, self).cuts(*args, **kwargs)
        if self.matched:
            # require that at least one tau matches truth
            cut &= Cut('tau1_matched || tau2_matched')
        return cut

class Others(MC, Background):
    NO_KYLEFIX = True
    NORM_BY_THEORY = True

    def __init__(self, *args, **kwargs):
        self.matched = kwargs.pop('matched', True)
        super(Others, self).__init__(*args, **kwargs)

    def histfactory(self, sample, category, systematics=False, **kwargs):
        if not systematics:
            return
        # https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/HSG4Uncertainties
        # pdf uncertainty
        sample.AddOverallSys('pdf_qq', 0.96, 1.04)
        # QCD scale uncertainty
        sample.AddOverallSys('QCDscale_V', 0.99, 1.01)

    def cuts(self, *args, **kwargs):
        cut = super(Others, self).cuts(*args, **kwargs)
        if self.matched:
            # require that at least one tau matches truth
            cut &= Cut('tau1_matched || tau2_matched')
        return cut


# INDIVIDUAL SAMPLES
class MC_Wtaunu(MC, Background):
    pass

class MC_Wmunu(MC, Background):
    pass

class MC_Wenu(MC, Background):
    pass


class MC_Zee_DY(MC, Background):
    pass

class MC_Zmumu_DY(MC, Background):
    pass
