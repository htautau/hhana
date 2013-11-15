from . import log
from .sample import Sample
from .db import TEMPFILE, get_file
from ..cachedtable import CachedTable
from ..lumi import LUMI

import numpy as np
from numpy.lib import recfunctions

from rootpy import asrootpy
from rootpy.plotting import Hist


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
            label = '#scale[0.7]{#int} L dt = %.1f fb^{-1} ' % self.lumi
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

    def __init__(self, year, markersize=2, **kwargs):

        super(Data, self).__init__(
            year=year, scale=1.,
            markersize=markersize, **kwargs)
        rfile = get_file(self.student)
        h5file = get_file(self.student, hdf=True)
        dataname = 'data%d_JetTauEtmiss' % (year % 1E3)
        self.data = getattr(rfile, dataname)
        self.h5data = CachedTable.hook(getattr(h5file.root, dataname))

        self.info = DataInfo(LUMI[self.year] / 1e3, self.energy)
        self._label = 'Data'
        self._label_root = 'Data'
        self.name = 'Data'

    def events(self, category, region, cuts=None, hist=None):

        if hist is None:
            hist = Hist(1, -100, 100)
        selection = self.cuts(category, region) & cuts
        log.debug("requesting number of events from %s using cuts: %s" %
                  (self.data.GetName(), selection))
        self.data.Draw('1', selection, hist=hist)
        return hist

    def draw_into(self, hist, expr, category, region,
                  cuts=None, weighted=True, systematics=True):

        self.data.draw(expr, self.cuts(category, region) & cuts, hist=hist)
        if not hasattr(hist, 'datainfo'):
            hist.datainfo = DataInfo(self.info.lumi, self.info.energies)
        else:
            hist.datainfo += self.info

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
            scores = self.scores(
                clf, category, region, cuts=cuts)

        self.draw_array_helper(field_hist, category, region,
            cuts=cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            scores=scores,
            clf=clf,
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

    def trees(self,
              category,
              region,
              cuts=None,
              systematic='NOMINAL'):

        TEMPFILE.cd()
        tree = asrootpy(self.data.CopyTree(self.cuts(category, region) & cuts))
        tree.userdata.weight_branches = []
        return [tree]

    def records(self,
                category,
                region,
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
        rec = self.h5data.read_where(selection.where(), **kwargs)

        # add weight field
        if include_weight:
            # data is not weighted
            weights = np.ones(rec.shape[0], dtype='f8')
            rec = recfunctions.rec_append_fields(rec,
                names='weight',
                data=weights,
                dtypes='f8')

        if fields is not None:
            rec = rec[fields]

        if return_idx:
            idx = self.h5data.get_where_list(selection.where(), **kwargs)
            return [(rec, idx)]

        return [rec]

