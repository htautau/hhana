# numpy imports
import numpy as np
from numpy.lib import recfunctions
from matplotlib.mlab import rec_append_fields
# rootpy imports
from rootpy import asrootpy
from rootpy.plotting import Hist

# local imports
from . import log; log = log[__name__]
from .sample import Sample
from .db import TEMPFILE, get_file
from ..cachedtable import CachedTable
from ..lumi import LUMI
from moments import HCM
from root_numpy import rec2array

class DataInfo():
    """
    Class to hold lumi and collision energy info for plot labels
    """
    def __init__(self, lumi, energies):
        self.lumi = lumi
        if not isinstance(energies, (tuple, list)):
            self.energies = [energies]
        else:
            # defensive copy
            self.energies = energies[:]
        self.mode = 'root'

    def __add__(self, other):
        return DataInfo(self.lumi + other.lumi,
                        self.energies + other.energies)

    def __iadd__(self, other):
        self.lumi += other.lumi
        self.energies.extend(other.energies)

    def __str__(self):
        if self.mode == 'root':
            label = '#scale[0.7]{#int} L dt = %.1f fb^{-1}  ' % self.lumi
            label += '#sqrt{#font[52]{s}} = '
            label += '+'.join(map(lambda e: '%d TeV' % e,
                                  sorted(set(self.energies))))
        else:
            label = '$\int L dt = %.1f$ fb$^{-1}$ ' % self.lumi
            label += '$\sqrt{s} =$ '
            label += '$+$'.join(map(lambda e: '%d TeV' % e,
                                    sorted(set(self.energies))))
        return label


class Data(Sample):

    def __init__(self, year, name='Data', label='Data', **kwargs):
        super(Data, self).__init__(
            year=year, scale=1.,
            name=name, label=label,
            **kwargs)
        h5file = get_file(self.ntuple_path, self.student, hdf=True)
        if year == 2015:
            if self.channel == 'hadhad':
                stream_name = 'Main25'
            else:
                stream_name = 'Main'
        else:
            stream_name = 'JetTauEtMiss'
        dataname = 'data{0:1d}_{1}'.format(
            int(year % 1E3), stream_name)

        self.h5data = CachedTable.hook(getattr(h5file.root, dataname))
        self.info = DataInfo(LUMI[self.year] / 1e3, self.energy)

    def draw_array(self, field_hist, category, region,
                   cuts=None,
                   weighted=True,
                   field_scale=None,
                   weight_hist=None,
                   clf=None,
                   scores=None,
                   min_score=None,
                   max_score=None,
                   systematics=True,
                   systematics_components=None,
                   bootstrap_data=False):
        if bootstrap_data:
            scores = None
        elif scores is None and clf is not None:
            scores = self.scores(clf, category, region, cuts=cuts)
        elif isinstance(scores, dict):
            scores = scores['NOMINAL']
        if isinstance(scores, tuple):
            # ignore weights
            scores = scores[0]
        return self.draw_array_helper(field_hist, category, region,
            cuts=cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            clf=clf,
            scores=scores,
            min_score=min_score,
            max_score=max_score,
            bootstrap_data=bootstrap_data)

    def scores(self, clf, category, region,
               cuts=None,
               systematics=True,
               systematics_components=None):
        return clf.classify(self,
                category=category,
                region=region,
                cuts=cuts)

    def records(self,
                category=None,
                region=None,
                fields=None,
                cuts=None,
                include_weight=True,
                systematic='NOMINAL',
                return_idx=False,
                **kwargs):
        if include_weight and fields is not None:
            if 'weight' not in fields:
                fields = list(fields) + ['weight']

        selection = self.cuts(category, region) & cuts

        log.info("requesting table from Data %d" % self.year)
        log.debug("using selection: %s" % selection)

        # read the table with a selection
        if selection:
            rec = self.h5data.read_where(selection.where(), **kwargs)
        else:
            rec = self.h5data.read(**kwargs)

        # add weight field
        if include_weight:
            # data is not weighted
            weights = np.ones(rec.shape[0], dtype='f8')
#            rec = recfunctions.rec_append_fields(rec,
            rec = rec_append_fields(rec,
                names='weight',
                arrs=weights,
                dtypes=np.dtype('f8'))
        mom_branches = [
            'tau_0_pt', 'tau_0_eta', 'tau_0_phi', 'tau_0_m',
            'tau_1_pt', 'tau_1_eta', 'tau_1_phi', 'tau_1_m',
            'jet_0_pt', 'jet_0_eta', 'jet_0_phi', 'jet_0_m',
            'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_m',
        ]

        mom_arr = rec[mom_branches]
        mom_arr = rec2array( mom_arr ).reshape((mom_arr.shape[0], 4, 4))
        # convert array of pT, eta, phi, m
        # to array of p, px, py, pz, pT, eta, phi, m
        kin_arr = np.empty(shape=(mom_arr.shape[0], 4, 8))
        # |p| = pT cosh eta
        kin_arr[:,:,0] = mom_arr[:,:,0] * np.cosh(mom_arr[:,:,1])
        # px, py, pz
        kin_arr[:,:,1] = mom_arr[:,:,0] * np.cos(mom_arr[:,:,2])
        kin_arr[:,:,2] = mom_arr[:,:,0] * np.sin(mom_arr[:,:,2])
        kin_arr[:,:,3] = mom_arr[:,:,0] * np.sinh(mom_arr[:,:,1])
        # pT, eta, phi, m
        kin_arr[:,:,4] = mom_arr[:,:,0]
        kin_arr[:,:,5] = mom_arr[:,:,1]
        kin_arr[:,:,6] = mom_arr[:,:,2]
        kin_arr[:,:,7] = mom_arr[:,:,3]
        rec2jj_moments = HCM( 2, kin_arr[:,[2,3],:])#, kin_arr[:,:,:] )
        rec = rec_append_fields(rec,
            names='HCM2jj',
            arrs=rec2jj_moments,
            dtypes=np.dtype('f8'))
        rec2_moments = HCM( 2, kin_arr[:,:,:])#, kin_arr[:,:,:] )
        rec = rec_append_fields(rec,
            names='HCM2',
            arrs=rec2_moments,
            dtypes=np.dtype('f8'))
        rec3_moments = HCM( 3, kin_arr[:,:,:])#, kin_arr[:,:,:] )
        rec = rec_append_fields(rec,
            names='HCM3',
            arrs=rec3_moments,
            dtypes=np.dtype('f8'))
        rec1_moments = HCM( 1, kin_arr[:,:,:])#, kin_arr[:,:,:] )
        rec = rec_append_fields(rec,
            names='HCM1',
            arrs=rec1_moments,
            dtypes=np.dtype('f8'))

# VBF Variables
        rec = rec_append_fields(rec,
            names='dEta_jets',
            arrs=(np.absolute(kin_arr[:,2,5]-kin_arr[:,3,5])),
            dtypes=np.dtype('f8'))
        rec = rec_append_fields(rec,
            names='eta_product_jets',
            arrs=(kin_arr[:,2,5]*kin_arr[:,3,5]),
            dtypes=np.dtype('f8'))
        # M2 = 2 pT1 pT2 ( cosh(eta1-eta2) - cos(phi1-phi2)
        M2 = kin_arr[:,2,4] * kin_arr[:,3,4] * 2 * (np.cosh(kin_arr[:,2,5]-kin_arr[:,3,5])-np.cos(kin_arr[:,2,6]-kin_arr[:,3,6]))
        rec = rec_append_fields(rec,
            names='mass_jet1_jet2',
            arrs=np.absolute(M2)**0.5*10**3,
            dtypes=np.dtype('f8'))
        sum_vector_pt2 = np.sum(kin_arr[:,:,1], axis = 1)**2 \
                      + np.sum(kin_arr[:,:,2], axis = 1)**2
        rec = rec_append_fields(rec,
            names='vector_sum_pt',
            arrs=sum_vector_pt2**0.5,
            dtypes=np.dtype('f8'))

# centrality: exp( -4/(eta1-eta2)^2 (eta- eta1+eta2.2)^2)
        tau1_centrality = np.exp( \
                -4 / (kin_arr[:,2,5]-kin_arr[:,3,5])**2 \
                * (kin_arr[:,0,5]-(kin_arr[:,2,5]+kin_arr[:,3,5])/2)**2 \
                )
        tau2_centrality = np.exp( \
                -4 / (kin_arr[:,2,5]-kin_arr[:,3,5])**2 \
                * (kin_arr[:,1,5]-(kin_arr[:,2,5]+kin_arr[:,3,5])/2)**2 \
                )
# No MET centrality
        rec = rec_append_fields(rec,
            names='tau1_centrality',
            arrs=tau1_centrality,
            dtypes=np.dtype('f8'))
        rec = rec_append_fields(rec,
            names='tau2_centrality',
            arrs=tau2_centrality,
            dtypes=np.dtype('f8'))

# Tau PT ratio for boosted
        rec = rec_append_fields(rec,
            names='ditau_pt_ratio',
            arrs=(kin_arr[:,0,4]/kin_arr[:,1,4]),
            dtypes=np.dtype('f8'))

        if fields is not None:
            rec = rec[fields]

        if return_idx:
            # only valid if selection is non-empty
            idx = self.h5data.get_where_list(selection.where(), **kwargs)
            return [(rec, idx)]

        return [rec]
