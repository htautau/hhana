# rootpy imports
from rootpy.tree import Cut

# local imports
from .sample import Sample, Background
from .mc import MC
from . import log
from ..systematics import EMBEDDING_SYSTEMATICS, WEIGHT_SYSTEMATICS
from ..regions import REGIONS


class Ztautau(Background):
    NORM_BY_THEORY = False

    def histfactory(self, sample, category, systematics=True):
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
    WORKSPACE_SYSTEMATICS = MC.WORKSPACE_SYSTEMATICS


class MC_Ztautau_DY(MC_Ztautau):
    pass


class Embedded_Ztautau(Ztautau, MC):
    WORKSPACE_SYSTEMATICS = Sample.WORKSPACE_SYSTEMATICS + [
        'MFS',
        'ISOL',
        #'TES',
        'TES_TRUE',
        'TES_FAKE',
        #'TES_EOP',
        #'TES_CTB',
        #'TES_Bias',
        #'TES_EM',
        #'TES_LCW',
        #'TES_PU',
        #'TES_OTHERS',
        'TAUID',
        'TRIGGER',
        'FAKERATE',
    ]

    def xsec_kfact_effic(self, isample):
        return 1., 1., 1.

    def get_weight_branches(self, systematic,
                            no_cuts=False, only_cuts=False,
                            weighted=True):
        if not weighted:
            return ["1.0"]
        systerm, variation = Sample.get_sys_term_variation(systematic)
        if not only_cuts:
            weight_branches = [
                'mc_weight',
                'pileup_weight',
                'ggf_weight',
                'embedding_reco_unfold',
                'embedding_trigger_weight',
                'embedding_spin_weight',
            ]
            for term, variations in WEIGHT_SYSTEMATICS.items():
                if term == systerm:
                    weight_branches += variations[variation]
                else:
                    weight_branches += variations['NOMINAL']
        else:
            weight_branches = []
        if not no_cuts:
            for term, variations in EMBEDDING_SYSTEMATICS.items():
                if term == systerm:
                    if variations[variation]:
                        weight_branches.append(variations[variation])
                else:
                    if variations['NOMINAL']:
                        weight_branches.append(variations['NOMINAL'])
        return weight_branches

    def iter_weight_branches(self):
        for type, variations in WEIGHT_SYSTEMATICS.items():
            for variation in variations:
                if variation == 'NOMINAL':
                    continue
                term = ('%s_%s' % (type, variation),)
                yield self.get_weight_branches(term), term
        for type, variations in EMBEDDING_SYSTEMATICS.items():
            for variation in variations:
                if variation == 'NOMINAL':
                    continue
                term = ('%s_%s' % (type, variation),)
                yield self.get_weight_branches(term), term

    def cuts(self, category, region, systematic='NOMINAL', **kwargs):
        sys_cut = Cut()
        systerm, variation = Sample.get_sys_term_variation(systematic)
        for term, variations in EMBEDDING_SYSTEMATICS.items():
            if term == systerm:
                sys_cut &= variations[variation]
            else:
                sys_cut &= variations['NOMINAL']
        return (category.get_cuts(self.year, **kwargs) &
                REGIONS[region] & self._cuts & sys_cut)
