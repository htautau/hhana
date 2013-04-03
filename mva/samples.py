# std lib imports
import os
import sys
import atexit
from operator import add, itemgetter
import math
import warnings

# numpy imports
import numpy as np
from numpy.lib import recfunctions

# pytables imports
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tables

# higgstautau imports
from higgstautau import datasets
from higgstautau.decorators import cached_property, memoize_method
from higgstautau import samples as samples_db

# rootpy imports
import ROOT
from rootpy.plotting import Hist, Hist2D, Canvas, HistStack
from rootpy.io import root_open as ropen, TemporaryFile
from rootpy.tree import Tree, Cut
from rootpy import asrootpy
from rootpy.memory.keepalive import keepalive

# local imports
from . import log; log = log[__name__]
from . import variables
from . import NTUPLE_PATH, DEFAULT_STUDENT
from .utils import print_hist, rec_to_ndarray
from .lumi import LUMI
from .systematics import *
from .constants import *
from .classify import histogram_scores
from .stats.histfactory import to_uniform_binning
from .cachedtable import CachedTable

# Higgs cross sections
import yellowhiggs

VERBOSE = False

DB_HH = datasets.Database(name='datasets_hh', verbose=VERBOSE)
DB_TAUID = datasets.Database(name='datasets_tauid', verbose=VERBOSE)
FILES = {}
OS = Cut('tau1_charge * tau2_charge == -1')
NOT_OS = Cut('tau1_charge * tau2_charge >= 0') # changed by Zinonas
SS = Cut('tau1_charge * tau2_charge == 1')

P1P3_RECOUNTED = (
    (Cut('tau1_numTrack_recounted == 1') | Cut('tau1_numTrack_recounted == 3'))
    &
    (Cut('tau2_numTrack_recounted == 1') | Cut('tau2_numTrack_recounted == 3')))


# mass_jet1_jet2 > 100000
TEMPFILE = TemporaryFile()


def get_file(student, hdf=False, suffix=''):

    if hdf:
        ext = '.h5'
    else:
        ext = '.root'
    filename = student + ext
    if filename in FILES:
        return FILES[filename]
    file_path = os.path.join(NTUPLE_PATH, student + suffix, filename)
    log.info("opening %s ..." % file_path)
    if hdf:
        student_file = tables.openFile(file_path)
    else:
        student_file = ropen(file_path, 'READ')
    FILES[filename] = student_file
    return student_file


@atexit.register
def cleanup():

    TEMPFILE.Close()
    for filehandle in FILES.values():
        filehandle.close()


class Sample(object):

    REGIONS = {
        'ALL': Cut(),
        'OS': OS,
        '!OS': NOT_OS,
        'SS': SS}

    WEIGHT_BRANCHES = [
        'mc_weight',
        'pileup_weight', # 2012 PROBLEM
        'ggf_weight',
    ]

    SYSTEMATICS_COMPONENTS = []

    def __init__(self, year, scale=1., cuts=None,
                 student=DEFAULT_STUDENT,
                 **hist_decor):

        self.year = year
        if year == 2011:
            self.energy = 7
        else:
            self.energy = 8

        self.scale = scale
        if cuts is None:
            self._cuts = Cut()
        else:
            self._cuts = cuts
        self.student = student
        self.hist_decor = hist_decor
        #if isinstance(self, Higgs):
        #    self.hist_decor['fillstyle'] = 'hollow'
        #else:
        if 'fillstyle' not in hist_decor:
            self.hist_decor['fillstyle'] = 'solid'

    def get_histfactory_sample(self, hist_template,
                               expr_or_clf,
                               category, region,
                               cuts=None, p1p3=True,
                               scores=None,
                               systematics=True):

        log.info("creating histfactory sample for %s" % self.name)
        if isinstance(self, Data):
            sample = ROOT.RooStats.HistFactory.Data()
        else:
            sample = ROOT.RooStats.HistFactory.Sample(self.name)

        ndim = hist_template.GetDimension()
        do_systematics = (not isinstance(self, Data)
                          and self.systematics
                          and systematics)

        if isinstance(expr_or_clf, basestring):
            expr = expr_or_clf
            hist = hist_template.Clone()
            hist.Reset()
            self.draw_into(hist, expr, category, region, cuts, p1p3=p1p3)
            if ndim > 1:
                if do_systematics:
                    syst = hist.systematics
                # convert to 1D hist
                hist = hist.ravel()
                if do_systematics:
                    hist.systematics = syst

        else:
            # histogram classifier output
            if scores is not None:
                scores = self.scores(expr_or_clf, category, region, cuts)
            hist = histogram_scores(hist_template, scores)

        # set the nominal histogram
        print_hist(hist)
        uniform_hist = to_uniform_binning(hist)
        sample.SetHisto(uniform_hist)
        keepalive(sample, uniform_hist)

        # add systematics samples
        if do_systematics:
            for sys_component in self.__class__.SYSTEMATICS_COMPONENTS:
                terms = SYSTEMATICS[sys_component]
                if len(terms) == 1:
                    up_term = terms[0]
                    down_term = terms[0]
                else:
                    up_term, down_term = terms
                log.info("adding histosys for %s" % sys_component)
                histsys = ROOT.RooStats.HistFactory.HistoSys(sys_component)

                hist_up = hist.systematics[up_term]
                hist_down = hist.systematics[down_term]

                if ndim > 1:
                    # convert to 1D hists
                    hist_up = hist_up.ravel()
                    hist_down = hist_down.ravel()

                uniform_hist_up = to_uniform_binning(hist_up)
                uniform_hist_down = to_uniform_binning(hist_down)

                histsys.SetHistoHigh(uniform_hist_up)
                histsys.SetHistoLow(uniform_hist_down)
                keepalive(histsys, uniform_hist_up, uniform_hist_down)

                sample.AddHistoSys(histsys)
                keepalive(sample, histsys)

        if isinstance(self, Signal):
            log.info("defining SigXsecOverSM POI for %s" % self.name)
            sample.AddNormFactor('SigXsecOverSM', 0., 0., 60.)
        elif isinstance(self, Background):
            # only activate stat error on background samples
            log.info("activating stat error for %s" % self.name)
            sample.ActivateStatError()

        if hasattr(self, 'histfactory'):
            # perform sample-specific items
            log.info("calling %s histfactory method" % self.name)
            self.histfactory(sample)

        return sample

    def partitioned_records(self,
              category,
              region,
              fields=None,
              cuts=None,
              include_weight=True,
              systematic='NOMINAL',
              num_partitions=2):
        """
        Partition sample into num_partitions chunks of roughly equal size
        assuming no correlation between record index and field values.
        """
        partitions = []
        for start in range(num_partitions):
            recs = self.records(
                category,
                region,
                fields=fields,
                include_weight=include_weight,
                cuts=cuts,
                systematic=systematic,
                start=start,
                step=num_partitions)
            partitions.append(np.hstack(recs))

        return partitions

    def merged_records(self,
              category,
              region,
              fields=None,
              cuts=None,
              include_weight=True,
              systematic='NOMINAL'):

        recs = self.records(
                category,
                region,
                fields=fields,
                include_weight=include_weight,
                cuts=cuts,
                systematic=systematic)

        return np.hstack(recs)

    def array(self,
              category,
              region,
              fields=None,
              cuts=None,
              include_weight=True,
              systematic='NOMINAL'):

        return rec_to_ndarray(self.merged_records(
            category,
            region,
            fields=fields,
            cuts=cuts,
            include_weight=include_weight,
            systematic=systematic))

    @classmethod
    def check_systematic(cls, systematic):

        if systematic != 'NOMINAL' and issubclass(cls, Data):
            raise TypeError('Do not apply systematics on data!')

    @classmethod
    def get_sys_term_variation(cls, systematic):

        Sample.check_systematic(systematic)
        if systematic == 'NOMINAL':
            systerm = None
            variation = 'NOMINAL'
        elif len(systematic) > 1:
            # no support for this yet...
            systerm = None
            variation = 'NOMINAL'
        else:
            systerm, variation = systematic[0].split('_')
        return systerm, variation

    def get_weight_branches(self, systematic, no_cuts=False, weighted=True):

        if not weighted:
            return ["1.0"]
        systerm, variation = Sample.get_sys_term_variation(systematic)
        weight_branches = Sample.WEIGHT_BRANCHES[:]
        for term, variations in WEIGHT_SYSTEMATICS.items():
            if term == systerm:
                weight_branches += variations[variation]
            else:
                weight_branches += variations['NOMINAL']
        if not no_cuts and isinstance(self, Embedded_Ztautau):
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
        if isinstance(self, Embedded_Ztautau):
            for type, variations in EMBEDDING_SYSTEMATICS.items():
                for variation in variations:
                    if variation == 'NOMINAL':
                        continue
                    term = ('%s_%s' % (type, variation),)
                    yield self.get_weight_branches(term), term

    def cuts(self, category, region, systematic=None, p1p3=True):

        sys_cut = Cut()
        if p1p3:
            sys_cut &= P1P3_RECOUNTED
        if systematic is not None:
            systerm, variation = Sample.get_sys_term_variation(systematic)
            if isinstance(self, Embedded_Ztautau):
                for term, variations in EMBEDDING_SYSTEMATICS.items():
                    if term == systerm:
                        sys_cut &= variations[variation]
                    else:
                        sys_cut &= variations['NOMINAL']

        return (category.get_cuts(self.year) &
                Sample.REGIONS[region] & self._cuts & sys_cut)

    def draw(self, expr, category, region, bins, min, max,
             cuts=None, p1p3=True, weighted=True):

        hist = Hist(bins, min, max, title=self.label, **self.hist_decor)
        self.draw_into(hist, expr, category, region,
                       cuts=cuts, p1p3=p1p3, weighted=weighted)
        return hist

    def draw2d(self, expr, category, region,
               xbins, xmin, xmax,
               ybins, ymin, ymax,
               cuts=None, p1p3=True):

        hist = Hist2D(xbins, xmin, xmax, ybins, ymin, ymax,
                title=self.label, **self.hist_decor)
        self.draw_into(hist, expr, category, region, cuts=cuts, p1p3=p1p3)
        return hist


class Data(Sample):

    def __init__(self, year, **kwargs):

        super(Data, self).__init__(year=year, scale=1., **kwargs)
        rfile = get_file(self.student)
        h5file = get_file(self.student, hdf=True)
        dataname = 'data%d_JetTauEtmiss' % (year % 1E3)
        self.data = getattr(rfile, dataname)
        self.h5data = CachedTable.hook(getattr(h5file.root, dataname))

        self.label = ('%s Data $\sqrt{s} = %d$ TeV\n'
                      '$\int L dt = %.2f$ fb$^{-1}$' % (
                          self.year, self.energy, LUMI[self.year] / 1e3))
        self.name = 'Data'

    def events(self, category, region, cuts=None, p1p3=True):

        return self.data.GetEntries(self.cuts(category, region, p1p3=p1p3) & cuts)

    def draw_into(self, hist, expr, category, region,
                  cuts=None, p1p3=True, weighted=True):

        self.data.draw(expr, self.cuts(category, region, p1p3=p1p3) & cuts, hist=hist)

    def draw_array(self, hist, expr, category, region,
                   cuts=None, p1p3=True, weighted=True,
                   weight_hist=None, weight_clf=None):
        # TODO: draw from array
        # TODO: support expr and hist as lists
        # should offer a huge speedup for drawing multiple expressions with the
        # same cuts
        arr = self.array(category, region,
                fields=[expr], cuts=cuts,
                include_weight=True)
        if weight_hist is not None:
            scores = self.scores(weight_clf, category, region, cuts=cuts)[0]
            edges = np.array(list(weight_hist.xedges()))
            weights = np.array(weight_hist).take(edges.searchsorted(scores) - 1)
            weights = arr[:, -1] * weights
        else:
            weights = arr[:, -1]
        hist.fill_array(arr[:, 0], weights=weights)

    def scores(self, clf, category, region, cuts=None):

        if category != clf.category:
            raise ValueError(
                'classifier applied to category in which it was not trained')

        return clf.classify(self,
                category=category,
                region=region,
                cuts=cuts)

    def trees(self,
              category,
              region,
              cuts=None,
              systematic='NOMINAL',
              p1p3=True):

        Sample.check_systematic(systematic)
        TEMPFILE.cd()
        tree = asrootpy(self.data.CopyTree(self.cuts(category, region, p1p3=p1p3) & cuts))
        tree.userdata.weight_branches = []
        return [tree]

    def records(self,
                category,
                region,
                fields=None,
                cuts=None,
                include_weight=True,
                systematic='NOMINAL',
                p1p3=True,
                **kwargs):

        if include_weight and fields is not None:
            if 'weight' not in fields:
                fields = fields + ['weight']

        Sample.check_systematic(systematic)
        selection = self.cuts(category, region, p1p3=p1p3) & cuts

        log.info("requesting table from %s %d" %
                 (self.__class__.__name__, self.year))
        log.debug("using selection: %s" % selection)

        # read the table with a selection
        rec = self.h5data.readWhere(selection.where(),
                                    stop=len(self.h5data),
                                    **kwargs)
        # add weight field
        if include_weight:
            # data is not weighted
            weights = np.ones(rec.shape[0], dtype='f4')
            rec = recfunctions.rec_append_fields(rec,
                    names='weight',
                    data=weights,
                    dtypes='f4')
        if fields is not None:
            rec = rec[fields]
        return [rec]


class Signal:
    # mixin
    pass


class Background:
    # mixin
    pass


class MC(Sample):

    # TODO: remove 'JE[S|R]' here unless embedded classes should inherit from
    # elsewhere
    SYSTEMATICS_COMPONENTS = Sample.SYSTEMATICS_COMPONENTS + [
        'JES',
        'JER',
        'TES',
        'TAUID',
        'TRIGGER',
    ]

    def __init__(self, year, db=DB_HH, systematics=True, **kwargs):

        if isinstance(self, Background):
            sample_key = self.__class__.__name__.lower()
            sample_info = samples_db.get_sample(
                    'hadhad', year, 'background', sample_key)
            self.name = sample_info['name']
            self._label = sample_info['latex']
            self._label_root = sample_info['root']
            if 'color' in sample_info and 'color' not in kwargs:
                kwargs['color'] = sample_info['color']
            self.samples = sample_info['samples']

        elif isinstance(self, Signal):
            # samples already defined in Signal subclass
            # see Higgs class below
            assert len(self.samples) > 0

        else:
            raise TypeError(
                'MC sample %s does not inherit from Signal or Background' %
                self.__class__.__name__)

        super(MC, self).__init__(year=year, **kwargs)

        self.db = db
        self.datasets = []
        self.systematics = systematics
        rfile = get_file(self.student)
        h5file = get_file(self.student, hdf=True)

        for i, name in enumerate(self.samples):

            ds = self.db[name]
            treename = name.replace('.', '_')
            treename = treename.replace('-', '_')

            trees = {}
            tables = {}
            weighted_events = {}

            if isinstance(self, Embedded_Ztautau):
                events_bin = 0
            else:
                # use mc_weighted second bin
                events_bin = 1
            events_hist_suffix = '_cutflow'

            trees['NOMINAL'] = rfile.Get(treename)
            tables['NOMINAL'] =  CachedTable.hook(getattr(
                h5file.root, treename))

            weighted_events['NOMINAL'] = rfile.Get(
                    treename + events_hist_suffix)[events_bin]

            if self.systematics:

                systematics_terms, systematics_samples = \
                    samples_db.get_systematics('hadhad', self.year, name)

                # TODO: check that all expected systematics components are
                # included

                unused_terms = SYSTEMATICS_TERMS[:]

                if systematics_terms:
                    for sys_term in systematics_terms:

                        # merge terms such as JES_UP,TES_UP (embedding)
                        # and TES_UP (MC)
                        actual_sys_term = sys_term
                        for term in unused_terms:
                            if set(term) & set(sys_term):
                                if len(sys_term) < len(term):
                                    log.info("merging %s and %s" % (
                                        term, sys_term))
                                    sys_term = term
                                break

                        sys_name = treename + '_' + '_'.join(actual_sys_term)
                        trees[sys_term] = rfile.Get(sys_name)
                        tables[sys_term] = CachedTable.hook(getattr(
                            h5file.root, sys_name))

                        weighted_events[sys_term] = rfile.Get(
                                sys_name + events_hist_suffix)[events_bin]

                        unused_terms.remove(sys_term)

                if systematics_samples:
                    for sample_name, sys_term in systematics_samples.items():

                        log.info("%s -> %s %s" % (name, sample_name, sys_term))

                        sys_term = tuple(sys_term.split(','))
                        sys_ds = self.db[sample_name]
                        sample_name = sample_name.replace('.', '_')
                        sample_name = sample_name.replace('-', '_')

                        trees[sys_term] = rfile.Get(sample_name)
                        tables[sys_term] = CachedTable.hook(getattr(
                            h5file.root, sample_name))

                        weighted_events[sys_term] = getattr(rfile,
                                sample_name + events_hist_suffix)[events_bin]

                        unused_terms.remove(sys_term)

                if unused_terms:
                    log.debug("UNUSED TERMS for %s:" % self.name)
                    log.debug(unused_terms)

                    for term in unused_terms:
                        trees[term] = None # flag to use NOMINAL
                        tables[term] = None
                        weighted_events[term] = None # flag to use NOMINAL

            if isinstance(self, Higgs):
                # use yellowhiggs for cross sections
                xs, _ = yellowhiggs.xsbr(
                        self.energy, self.masses[i],
                        Higgs.MODES_DICT[self.modes[i]][0], 'tautau')
                log.debug("{0} {1} {2} {3} {4} {5}".format(
                    name,
                    self.masses[i],
                    self.modes[i],
                    Higgs.MODES_DICT[self.modes[i]][0],
                    self.energy,
                    xs))
                xs *= TAUTAUHADHADBR
                kfact = 1.
                effic = 1.

            elif isinstance(self, Embedded_Ztautau):
                xs, kfact, effic = 1., 1., 1.

            else:
                xs, kfact, effic = ds.xsec_kfact_effic

            log.debug("{0} {1} {2} {3}".format(ds.name, xs, kfact, effic))
            self.datasets.append(
                    (ds, trees, tables, weighted_events, xs, kfact, effic))

    @property
    def label(self):

        l = self._label
        #if self.scale != 1. and not isinstance(self,
        #        (MC_Ztautau, Embedded_Ztautau)):
        #    l += r' ($\sigma_{SM} \times %g$)' % self.scale
        return l

    def draw_into(self, hist, expr, category, region,
                  cuts=None, p1p3=True, weighted=True):

        if isinstance(expr, (list, tuple)):
            exprs = expr
        else:
            exprs = (expr,)

        if hasattr(hist, 'systematics'):
            sys_hists = hist.systematics
        else:
            sys_hists = {}

        selection = self.cuts(category, region, p1p3=p1p3) & cuts

        for ds, sys_trees, sys_tables, sys_events, xs, kfact, effic in self.datasets:

            log.debug(ds.name)

            nominal_tree = sys_trees['NOMINAL']
            nominal_events = sys_events['NOMINAL']

            nominal_weight = (LUMI[self.year] * self.scale *
                    xs * kfact * effic / nominal_events)

            nominal_weighted_selection = (
                '%f * %s * (%s)' %
                (nominal_weight,
                 ' * '.join(map(str,
                     self.get_weight_branches('NOMINAL', weighted=weighted))),
                 selection))

            log.debug(nominal_weighted_selection)

            current_hist = hist.Clone()
            current_hist.Reset()

            # fill nominal histogram
            for expr in exprs:
                nominal_tree.Draw(expr, nominal_weighted_selection,
                        hist=current_hist)

            hist += current_hist

            if not self.systematics:
                continue

            # iterate over systematic variation trees
            for sys_term in sys_trees.iterkeys():

                # skip the nominal tree
                if sys_term == 'NOMINAL':
                    continue

                sys_hist = current_hist.Clone()

                sys_tree = sys_trees[sys_term]
                sys_event = sys_events[sys_term]

                if sys_tree is not None:

                    sys_hist.Reset()

                    sys_weight = (LUMI[self.year] * self.scale *
                            xs * kfact * effic / sys_event)

                    sys_weighted_selection = (
                        '%f * %s * (%s)' %
                        (sys_weight,
                         ' * '.join(map(str,
                             self.get_weight_branches('NOMINAL',
                                 weighted=weighted))),
                         selection))

                    log.debug(sys_weighted_selection)

                    for expr in exprs:
                        sys_tree.Draw(expr, sys_weighted_selection, hist=sys_hist)

                if sys_term not in sys_hists:
                    sys_hists[sys_term] = sys_hist
                else:
                    sys_hists[sys_term] += sys_hist

            # iterate over weight systematics on the nominal tree
            for weight_branches, sys_term in self.iter_weight_branches():

                sys_hist = current_hist.Clone()
                sys_hist.Reset()

                weighted_selection = (
                    '%f * %s * (%s)' %
                    (nominal_weight,
                     ' * '.join(map(str, weight_branches)),
                     selection))

                log.debug(weighted_selection)

                for expr in exprs:
                    nominal_tree.Draw(expr, weighted_selection, hist=sys_hist)

                if sys_term not in sys_hists:
                    sys_hists[sys_term] = sys_hist
                else:
                    sys_hists[sys_term] += sys_hist

            # QCD + Ztautau fit error
            if isinstance(self, Ztautau):
                up_fit = current_hist.Clone()
                up_fit *= ((self.scale + self.scale_error) / self.scale)
                down_fit = current_hist.Clone()
                down_fit *= ((self.scale - self.scale_error) / self.scale)
                if ('ZFIT_UP',) not in sys_hists:
                    sys_hists[('ZFIT_UP',)] = up_fit
                    sys_hists[('ZFIT_DOWN',)] = down_fit
                else:
                    sys_hists[('ZFIT_UP',)] += up_fit
                    sys_hists[('ZFIT_DOWN',)] += down_fit
            else:
                for _term in [('ZFIT_UP',), ('ZFIT_DOWN',)]:
                    if _term not in sys_hists:
                        sys_hists[_term] = current_hist.Clone()
                    else:
                        sys_hists[_term] += current_hist.Clone()

            for _term in [('QCDFIT_UP',), ('QCDFIT_DOWN',)]:
                if _term not in sys_hists:
                    sys_hists[_term] = current_hist.Clone()
                else:
                    sys_hists[_term] += current_hist.Clone()

        # set the systematics
        hist.systematics = sys_hists

    def draw_array(self, hist, expr, category, region,
                   cuts=None, p1p3=True, weighted=True,
                   weight_hist=None, weight_clf=None):
        # TODO: support expr and hist as lists
        # should offer a huge speedup for drawing multiple expressions with the
        # same cuts
        arr = self.array(category, region,
                fields=[expr], cuts=cuts,
                include_weight=True)
        if weight_hist is not None:
            scores = self.scores(weight_clf, category, region, cuts=cuts,
                    systematics=True)
            edges = np.array(list(weight_hist.xedges()))
            weights = np.array(weight_hist).take(edges.searchsorted(scores['NOMINAL'][0]) - 1)
            weights = arr[:, -1] * weights
        else:
            weights = arr[:, -1]
        hist.fill_array(arr[:, 0], weights=weights)

        if hasattr(hist, 'systematics'):
            sys_hists = hist.systematics
        else:
            sys_hists = {}
        # set the systematics
        hist.systematics = sys_hists
        if not self.systematics:
            return

        for systematic in iter_systematics(False):

            arr = self.array(category, region,
                    fields=[expr], cuts=cuts,
                    include_weight=True,
                    systematic=systematic)
            sys_hist = hist.Clone()
            sys_hist.Reset()
            if weight_hist is not None:
                edges = np.array(list(weight_hist.xedges()))
                weights = np.array(weight_hist).take(edges.searchsorted(scores[systematic][0]) - 1)
                weights = arr[:, -1] * weights
            else:
                weights = arr[:, -1]
            sys_hist.fill_array(arr[:, 0], weights=weights)
            sys_hists[systematic] = sys_hist

    def scores(self, clf, category, region,
               cuts=None, scores_dict=None,
               systematics=True):

        # TODO check that weight systematics are included

        if category != clf.category:
            raise ValueError(
                'classifier applied to category in which it was not trained')

        if scores_dict is None:
            scores_dict = {}

        for systematic in iter_systematics(True):

            if ((not systematics or not self.systematics)
                 and systematic != 'NOMINAL'):
                continue

            scores, weights = clf.classify(self,
                    category=category,
                    region=region,
                    cuts=cuts,
                    systematic=systematic)

            if systematic not in scores_dict:
                scores_dict[systematic] = (scores, weights)
            else:
                prev_scores, prev_weights = scores_dict[systematic]
                scores_dict[systematic] = (
                        np.concatenate((prev_scores, scores)),
                        np.concatenate((prev_weights, weights)))
        return scores_dict

    def trees(self, category, region, cuts=None, systematic='NOMINAL', p1p3=True):
        """
        This is where all the magic happens...
        """
        TEMPFILE.cd()
        selection = self.cuts(category, region, p1p3=p1p3) & cuts
        weight_branches = self.get_weight_branches(systematic)
        if systematic in SYSTEMATICS_BY_WEIGHT:
            systematic = 'NOMINAL'

        trees = []
        for ds, sys_trees, sys_tables, sys_events, xs, kfact, effic in self.datasets:

            if systematic in (('ZFIT_UP',), ('ZFIT_DOWN',),
                              ('QCDFIT_UP',), ('QCDFIT_DOWN',)):
                tree = sys_trees['NOMINAL']
                events = sys_events['NOMINAL']
            else:
                tree = sys_trees[systematic]
                events = sys_events[systematic]

                if tree is None:
                    tree = sys_trees['NOMINAL']
                    events = sys_events['NOMINAL']

            scale = self.scale
            if isinstance(self, Ztautau):
                if systematic == ('ZFIT_UP',):
                    scale = self.scale + self.scale_error
                elif systematic == ('ZFIT_DOWN',):
                    scale = self.scale - self.scale_error
            weight = scale * LUMI[self.year] * xs * kfact * effic / events

            selected_tree = asrootpy(tree.CopyTree(selection))
            log.debug("{0} {1}".format(selected_tree.GetEntries(), weight))
            selected_tree.SetWeight(weight)
            selected_tree.userdata.weight_branches = weight_branches
            log.debug("{0} {1} {2}".format(
                self.name, selected_tree.GetEntries(),
                selected_tree.GetWeight()))
            trees.append(selected_tree)
        return trees

    def records(self,
                category,
                region,
                fields=None,
                cuts=None,
                include_weight=True,
                systematic='NOMINAL',
                p1p3=True,
                **kwargs):

        if include_weight and fields is not None:
            if 'weight' not in fields:
                fields = fields + ['weight']

        selection = self.cuts(category, region, systematic, p1p3) & cuts

        if systematic == 'NOMINAL':
            log.info("requesting table from %s" %
                     (self.__class__.__name__))
        else:
            log.info("requesting table from %s for systematic %s " %
                     (self.__class__.__name__, systematic))
        log.debug("using selection: %s" % selection)

        # TODO: handle cuts in weight expressions
        weight_branches = self.get_weight_branches(systematic, no_cuts=True)
        if systematic in SYSTEMATICS_BY_WEIGHT:
            systematic = 'NOMINAL'
        recs = []
        for ds, sys_trees, sys_tables, sys_events, xs, kfact, effic in self.datasets:

            if systematic in (('ZFIT_UP',), ('ZFIT_DOWN',),
                              ('QCDFIT_UP',), ('QCDFIT_DOWN',)):
                table = sys_tables['NOMINAL']
                events = sys_events['NOMINAL']
            else:
                table = sys_tables[systematic]
                events = sys_events[systematic]

                if table is None:
                    log.debug("systematics table was None, using NOMINAL")
                    table = sys_tables['NOMINAL']
                    events = sys_events['NOMINAL']

            scale = self.scale
            if isinstance(self, Ztautau):
                if systematic == ('ZFIT_UP',):
                    scale = self.scale + self.scale_error
                elif systematic == ('ZFIT_DOWN',):
                    scale = self.scale - self.scale_error
            weight = scale * LUMI[self.year] * xs * kfact * effic / events

            table_selection = selection.where()
            log.debug(table_selection)

            # read the table with a selection
            rec = table.readWhere(table_selection, stop=len(table), **kwargs)

            # add weight field
            if include_weight:
                weights = np.empty(rec.shape[0], dtype='f4')
                weights.fill(weight)
                rec = recfunctions.rec_append_fields(rec,
                        names='weight',
                        data=weights,
                        dtypes='f4')
                # merge the weight fields
                rec['weight'] *= reduce(np.multiply,
                        [rec[br] for br in weight_branches])
                # drop other weight fields
                rec = recfunctions.rec_drop_fields(rec, weight_branches)
            if fields is not None:
                rec = rec[fields]
            recs.append(rec)
        return recs

    def events(self, selection='', systematic='NOMINAL'):

        total = 0.
        for ds, sys_trees, sys_tables, sys_events, xs, kfact, effic in self.datasets:
            tree = sys_trees[systematic]
            events = sys_events[systematic]

            weight = LUMI[self.year] * self.scale * xs * kfact * effic / events

            total += weight * tree.GetEntries(selection)
        return total

    def iter(self, selection='', systematic='NOMINAL'):

        TEMPFILE.cd()
        for ds, sys_trees, sys_tables, sys_events, xs, kfact, effic in self.datasets:
            tree = sys_trees[systematic]
            events = sys_events[systematic]

            weight = LUMI[self.year] * self.scale * xs * kfact * effic / events

            if selection:
                selected_tree = asrootpy(tree.CopyTree(selection))
            else:
                selected_tree = tree
            for event in selected_tree:
                yield weight, event


class Ztautau(Background):
    pass


class MC_Ztautau(MC, Ztautau):

    SYSTEMATICS_COMPONENTS = MC.SYSTEMATICS_COMPONENTS + [
        'Z_FIT',
    ]

    def __init__(self, *args, **kwargs):
        """
        Instead of setting the k factor here
        the normalization is determined by a fit to the data
        """
        self.scale_error = 0.
        super(MC_Ztautau, self).__init__(
                *args, **kwargs)


class Embedded_Ztautau(MC, Ztautau):

    SYSTEMATICS_COMPONENTS = MC.SYSTEMATICS_COMPONENTS + [
        'Z_FIT',
    ]

    def __init__(self, *args, **kwargs):
        """
        Instead of setting the k factor here
        the normalization is determined by a fit to the data
        """
        self.scale_error = 0.
        super(Embedded_Ztautau, self).__init__(
                *args, **kwargs)


class EWK(MC, Background):

    pass


class Top(MC, Background):

    pass


class Diboson(MC, Background):

    pass


class Others(MC, Background):

    pass


class Higgs(MC, Signal):

    MASS_POINTS = range(100, 155, 5)

    MODES = ['Z', 'W', 'gg', 'VBF']

    MODES_DICT = {
        'gg': ('ggf', 'PowHegPythia_', 'PowHegPythia8_AU2CT10_'),
        'VBF': ('vbf', 'PowHegPythia_', 'PowHegPythia8_AU2CT10_'),
        'Z': ('zh', 'Pythia', 'Pythia8_AU2CTEQ6L1_'),
        'W': ('wh', 'Pythia', 'Pythia8_AU2CTEQ6L1_'),
    }

    # constant uncert term, high, low
    UNCERT_GGF = {
        'pdf_gg': (1.079, 0.923),
        'QCDscale_ggH1in': (1.133, 0.914),
    }

    UNCERT_VBF = {
        'pdf_qqbar': (1.027, 0.979),
        'QCDscale_qqH': (1.004, 0.996),
    }

    UNCERT_WZH = {
        'pdf_qqbar': (1.039, 0.961),
        'QCDscale_VH': (1.007, 0.992),
    }

    def histfactory(self, sample):

        if len(self.modes) != 1:
            raise TypeError(
                    'histfactory sample only valid for single production mode')
        if len(self.masses) != 1:
            raise TypeError(
                    'histfactory sample only valid for single mass point')
        mode = self.modes[0]
        if mode == 'gg':
            overall_dict = self.UNCERT_GGF
        elif mode == 'VBF':
            overall_dict = self.UNCERT_VBF
        elif mode in ('Z', 'W'):
            overall_dict = self.UNCERT_WZH
        else:
            raise ValueError('mode %s is not valid' % mode)
        for term, (high, low) in overall_dict.items():
            log.debug("defining overall sys %s" % term)
            sample.AddOverallSys(term, low, high)

    def __init__(self, year,
            mode=None, modes=None,
            mass=None, masses=None, **kwargs):

        if masses is None:
            if mass is not None:
                assert mass in Higgs.MASS_POINTS
                masses = [mass]
            else:
                masses = Higgs.MASS_POINTS
        else:
            assert len(masses) > 0
            for mass in masses:
                assert mass in Higgs.MASS_POINTS
            assert len(set(masses)) == len(masses)

        if modes is None:
            if mode is not None:
                assert mode in Higgs.MODES
                modes = [mode]
            else:
                modes = Higgs.MODES
        else:
            assert len(modes) > 0
            for mode in modes:
                assert mode in Higgs.MODES
            assert len(set(modes)) == len(modes)

        str_mass = ''
        if len(masses) == 1:
            str_mass = '(%d)' % masses[0]

        str_mode = ''
        if len(modes) == 1:
            str_mode = modes[0]
            self.name = 'Signal_%s' % modes[0]
        else:
            self.name = 'Signal'

        #self._label = r'%s$H%s\rightarrow\tau_{\mathrm{had}}\tau_{\mathrm{had}}$' % (
        #        str_mode, str_mass)
        self._label = r'%sH%s' % (str_mode, str_mass)
        if year == 2011:
            suffix = 'mc11c'
            generator_index = 1
        elif year == 2012:
            suffix = 'mc12a'
            generator_index = 2
        else:
            raise ValueError('No Higgs defined for year %d' % year)

        self.samples = []
        self.masses = []
        self.modes = []
        for mode in modes:
            generator = Higgs.MODES_DICT[mode][generator_index]
            for mass in masses:
                self.samples.append('%s%sH%d_tautauhh.%s' % (
                    generator, mode, mass, suffix))
                self.masses.append(mass)
                self.modes.append(mode)

        super(Higgs, self).__init__(year=year, **kwargs)


class QCD(Sample, Background):

    SYSTEMATICS_COMPONENTS = MC.SYSTEMATICS_COMPONENTS + [
        'QCD_FIT',
    ]

    @staticmethod
    def sample_compatibility(data, mc):

        if not isinstance(mc, (list, tuple)):
            raise TypeError("mc must be a list or tuple of MC samples")
        if not mc:
            raise ValueError("mc must contain at least one MC sample")
        systematics = mc[0].systematics
        for m in mc:
            if data.year != m.year:
                raise ValueError("MC and Data years do not match")
            if m.systematics != systematics:
                raise ValueError(
                    "two MC samples with inconsistent systematics setting")

    def __init__(self, data, mc,
                 scale=1.,
                 shape_region='SS',
                 cuts=None,
                 color='#59d454'):

        QCD.sample_compatibility(data, mc)
        super(QCD, self).__init__(year=data.year, scale=scale, color=color)
        self.data = data
        self.mc = mc
        self.name = 'QCD'
        self.label = 'QCD Multi-jet'
        self.scale = 1.
        self.scale_error = 0.
        self.shape_region = shape_region
        self.systematics = mc[0].systematics

    def draw_into(self, hist, expr, category, region,
                  cuts=None, p1p3=True, weighted=True):

        MC_bkg_notOS = hist.Clone()
        for mc in self.mc:
            mc.draw_into(MC_bkg_notOS, expr, category, self.shape_region,
                         cuts=cuts, p1p3=p1p3, weighted=weighted)

        data_hist = hist.Clone()
        self.data.draw_into(data_hist, expr,
                            category, self.shape_region,
                            cuts=cuts, p1p3=p1p3, weighted=weighted)

        hist += (data_hist - MC_bkg_notOS) * self.scale

        if hasattr(MC_bkg_notOS, 'systematics'):
            if not hasattr(hist, 'systematics'):
                hist.systematics = {}
            for sys_term, sys_hist in MC_bkg_notOS.systematics.items():
                scale = self.scale
                if sys_term == ('FIT_UP',):
                    scale = self.scale + self.scale_error
                elif sys_term == ('FIT_DOWN',):
                    scale = self.scale - self.scale_error
                qcd_hist = (data_hist - sys_hist) * scale
                if sys_term not in hist.systematics:
                    hist.systematics[sys_term] = qcd_hist
                else:
                    hist.systematics[sys_term] += qcd_hist

        hist.SetTitle(self.label)

    def draw_array(self, hist, expr, category, region,
                  cuts=None, p1p3=True, weighted=True,
                  weight_hist=None, weight_clf=None):

        MC_bkg_notOS = hist.Clone()
        for mc in self.mc:
            mc.draw_array(MC_bkg_notOS, expr, category, self.shape_region,
                         cuts=cuts, p1p3=p1p3, weighted=weighted,
                         weight_hist=weight_hist, weight_clf=weight_clf)

        data_hist = hist.Clone()
        self.data.draw_array(data_hist, expr,
                            category, self.shape_region,
                            cuts=cuts, p1p3=p1p3, weighted=weighted,
                            weight_hist=weight_hist, weight_clf=weight_clf)

        hist += (data_hist - MC_bkg_notOS) * self.scale

        if hasattr(MC_bkg_notOS, 'systematics'):
            if not hasattr(hist, 'systematics'):
                hist.systematics = {}
            for sys_term, sys_hist in MC_bkg_notOS.systematics.items():
                scale = self.scale
                if sys_term == ('FIT_UP',):
                    scale = self.scale + self.scale_error
                elif sys_term == ('FIT_DOWN',):
                    scale = self.scale - self.scale_error
                qcd_hist = (data_hist - sys_hist) * scale
                if sys_term not in hist.systematics:
                    hist.systematics[sys_term] = qcd_hist
                else:
                    hist.systematics[sys_term] += qcd_hist

        hist.SetTitle(self.label)

    def scores(self, clf, category, region,
               cuts=None, systematics=True):

        if category != clf.category:
            raise ValueError(
                'classifier applied to category in which it was not trained')

        # SS data
        data_scores, data_weights = self.data.scores(
                clf,
                category,
                region=self.shape_region,
                cuts=cuts)

        scores_dict = {}
        # subtract SS MC
        for mc in self.mc:
            mc.scores(
                    clf,
                    category,
                    region=self.shape_region,
                    cuts=cuts,
                    scores_dict=scores_dict,
                    systematics=systematics)

        for sys_term in scores_dict.keys()[:]:
            sys_scores, sys_weights = scores_dict[sys_term]
            scale = self.scale
            if sys_term == ('QCDFIT_UP',):
                scale += self.scale_error
            elif sys_term == ('QCDFIT_DOWN',):
                scale -= self.scale_error
            # subtract SS MC
            sys_weights *= -1 * scale
            # add SS data
            sys_scores = np.concatenate((sys_scores, np.copy(data_scores)))
            sys_weights = np.concatenate((sys_weights, data_weights * scale))
            scores_dict[sys_term] = (sys_scores, sys_weights)

        return scores_dict

    def trees(self, category, region, cuts=None,
              systematic='NOMINAL'):

        TEMPFILE.cd()
        data_tree = asrootpy(
                self.data.data.CopyTree(
                    self.data.cuts(
                        category,
                        region=self.shape_region) & cuts))
        data_tree.userdata.weight_branches = []
        trees = [data_tree]
        for mc in self.mc:
            _trees = mc.trees(
                    category,
                    region=self.shape_region,
                    cuts=cuts,
                    systematic=systematic)
            for tree in _trees:
                tree.Scale(-1)
            trees += _trees

        scale = self.scale
        if systematic == ('QCDFIT_UP',):
            scale += self.scale_error
        elif systematic == ('QCDFIT_DOWN',):
            scale -= self.scale_error

        for tree in trees:
            tree.Scale(scale)
        return trees

    def records(self,
                category,
                region,
                fields=None,
                cuts=None,
                include_weight=True,
                systematic='NOMINAL',
                **kwargs):

        assert include_weight == True

        data_records = self.data.records(
                category=category,
                region=self.shape_region,
                fields=fields,
                cuts=cuts,
                include_weight=include_weight,
                systematic='NOMINAL',
                **kwargs)
        arrays = data_records

        for mc in self.mc:
            _arrays = mc.records(
                    category=category,
                    region=self.shape_region,
                    fields=fields,
                    cuts=cuts,
                    include_weight=include_weight,
                    systematic=systematic,
                    **kwargs)
            # FIX: weight may not be present if include_weight=False
            for array in _arrays:
                for partition in array:
                    partition['weight'] *= -1
            arrays.extend(_arrays)

        scale = self.scale
        if systematic == ('QCDFIT_UP',):
            scale += self.scale_error
        elif systematic == ('QCDFIT_DOWN',):
            scale -= self.scale_error

        # FIX: weight may not be present if include_weight=False
        for array in arrays:
            for partition in array:
                partition['weight'] *= scale
        return arrays


class MC_TauID(MC):

    def __init__(self, **kwargs):

        self.name = 'TauID'
        self._label = 'TauID'
        self.samples = ['PythiaWtaunu_incl.mc11c']
        super(MC_TauID, self).__init__(student='TauIDProcessor',
                db=DB_TAUID, **kwargs)
