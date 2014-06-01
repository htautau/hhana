# rootpy imports
from rootpy.tree import Cut

# local imports
from . import log
from .sample import SystematicsSample, Background, MC
from ..regions import REGIONS


class Ztautau(Background):
    NORM_BY_THEORY = False

    def histfactory(self, sample, category, systematics=False):
        if self.workspace_norm is False:
            return
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

class Pythia_Ztautau(MC_Ztautau):
    def xsec_kfact_effic(self, isample):
        return 1., 1., 1.


class Embedded_Ztautau(Ztautau, SystematicsSample):

    def systematics_components(self):
        # No FAKERATE for embedding since fakes are data
        return super(Embedded_Ztautau, self).systematics_components() + [
            'MFS',
            'ISOL',
        ]

    def weight_fields(self):
        weights = super(Embedded_Ztautau, self).weight_fields() + [
            'mc_weight',
            'embedding_reco_unfold',
            'embedding_trigger_weight',
        ]
        if self.tauspinner:
            weights.append('embedding_spin_weight')
        return weights

    def weight_systematics(self):
        systematics = super(Embedded_Ztautau, self).weight_systematics()
        if self.year == 2011:
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
        else:
            systematics.update({
                'TRIGGER': {
                    'UP': [
                        'tau1_trigger_eff_sys_high',
                        'tau2_trigger_eff_sys_high'],
                    'DOWN': [
                        'tau1_trigger_eff_sys_low',
                        'tau2_trigger_eff_sys_low'],
                    'NOMINAL': [
                        'tau1_trigger_eff',
                        'tau2_trigger_eff']},
                'TRIGGER_STAT': {
                    'PERIODA_UP': [
                        'tau1_trigger_eff_stat_scale_PeriodA_high',
                        'tau2_trigger_eff_stat_scale_PeriodA_high'],
                    'PERIODA_DOWN': [
                        'tau1_trigger_eff_stat_scale_PeriodA_low',
                        'tau2_trigger_eff_stat_scale_PeriodA_low'],
                    'PERIODBD_BARREL_UP': [
                        'tau1_trigger_eff_stat_scale_PeriodBD_Barrel_high',
                        'tau2_trigger_eff_stat_scale_PeriodBD_Barrel_high'],
                    'PERIODBD_BARREL_DOWN': [
                        'tau1_trigger_eff_stat_scale_PeriodBD_Barrel_low',
                        'tau2_trigger_eff_stat_scale_PeriodBD_Barrel_low'],
                    'PERIODBD_ENDCAP_UP': [
                        'tau1_trigger_eff_stat_scale_PeriodBD_EndCap_high',
                        'tau2_trigger_eff_stat_scale_PeriodBD_EndCap_high'],
                    'PERIODBD_ENDCAP_DOWN': [
                        'tau1_trigger_eff_stat_scale_PeriodBD_EndCap_low',
                        'tau2_trigger_eff_stat_scale_PeriodBD_EndCap_low'],
                    'PERIODEM_BARREL_UP': [
                        'tau1_trigger_eff_stat_scale_PeriodEM_Barrel_high',
                        'tau2_trigger_eff_stat_scale_PeriodEM_Barrel_high'],
                    'PERIODEM_BARREL_DOWN': [
                        'tau1_trigger_eff_stat_scale_PeriodEM_Barrel_low',
                        'tau2_trigger_eff_stat_scale_PeriodEM_Barrel_low'],
                    'PERIODEM_ENDCAP_UP': [
                        'tau1_trigger_eff_stat_scale_PeriodEM_EndCap_high',
                        'tau2_trigger_eff_stat_scale_PeriodEM_EndCap_high'],
                    'PERIODEM_ENDCAP_DOWN': [
                        'tau1_trigger_eff_stat_scale_PeriodEM_EndCap_low',
                        'tau2_trigger_eff_stat_scale_PeriodEM_EndCap_low'],
                    'NOMINAL': []}})
        return systematics

    def cut_systematics(self):
        systematics = super(Embedded_Ztautau, self).cut_systematics()
        if self.year == 2011:
            return systematics
        # isolation treatment in 2012 is different
        systematics.update({
            'ISOL': { # MUON ISOLATION
                'UP': Cut('(embedding_isolation == 2)'),
                'DOWN': Cut(),
                'NOMINAL': Cut('(embedding_isolation >= 1)')},
        })
        return systematics

    def xsec_kfact_effic(self, isample):
        return 1., 1., 1.

    def __init__(self, *args, **kwargs):
        self.tauspinner = kwargs.pop('tauspinner', True)
        self.posterior_trigger_correction = kwargs.pop('posterior_trigger_correction', True)
        super(Embedded_Ztautau, self).__init__(*args, **kwargs)


class MC_Embedded_Ztautau(Embedded_Ztautau):
    pass
