# rootpy imports
from rootpy.tree import Cut

# local imports
from . import log
from .sample import SystematicsSample, Background, MC
from ..regions import REGIONS


class Ztautau(Background):
    NORM_BY_THEORY = False

    def histfactory(self, sample, category, systematics=False):
        if self.workspace_norm is not None:
            sample.AddNormFactor(
                'ATLAS_norm_HH_{0:d}_Ztt'.format(self.year),
                self.workspace_norm,
                self.workspace_norm,
                self.workspace_norm,
                True) # const
        elif self.constrain_norm:
            # overallsys
            error = self.scale_error / self.scale
            sample.AddOverallSys(
                'ATLAS_norm_HH_{0:d}_Ztt'.format(self.year),
                1. - error,
                1. + error)
        else:
            sample.AddNormFactor(
                'ATLAS_norm_HH_{0:d}_Ztt'.format(self.year),
                1., 0., 50., False) # floating

    def __init__(self, *args, **kwargs):
        """
        Instead of setting the k factor here
        the normalization is determined by a fit to the data
        """
        self.scale_error = 0.
        self.workspace_norm = kwargs.pop('workspace_norm', None)
        self.constrain_norm = kwargs.pop('constrain_norm', False)
        super(Ztautau, self).__init__(*args, **kwargs)


class MC_Ztautau(Ztautau, MC):
    pass

class MC_Ztautau_DY(MC_Ztautau):
    pass


class Embedded_Ztautau(Ztautau, SystematicsSample):

    def systematics_components(self):
        return super(Embedded_Ztautau, self).systematics_components() + [
            'MFS',
            'ISOL',
        ]

    def weight_fields(self):
        return super(Embedded_Ztautau, self).weight_fields() + [
            'mc_weight',
            'embedding_reco_unfold',
            'embedding_trigger_weight',
            'embedding_spin_weight',
        ]

    def weight_systematics(self):
        systematics = super(Embedded_Ztautau, self).weight_systematics()
        systematics.update({
            'TRIGGER': {
                'UP': [
                    'tau1_trigger_eff_high',
                    'tau2_trigger_eff_high'],
                'DOWN': [
                    'tau1_trigger_eff_low',
                    'tau2_trigger_eff_low'],
                'NOMINAL': [
                    'tau1_trigger_eff',
                    'tau2_trigger_eff']},
        })
        return systematics

    def cut_systematics(self):
        systematics = super(Embedded_Ztautau, self).cut_systematics()
        systematics.update({
            'ISOL': { # MUON ISOLATION
                'UP': Cut('(embedding_isolation == 2)'),
                'DOWN': Cut(),
                'NOMINAL': Cut('(embedding_isolation >= 1)')},
        })
        return systematics

    def xsec_kfact_effic(self, isample):
        return 1., 1., 1.
