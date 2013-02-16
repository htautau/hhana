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
from rootpy.io import open as ropen
from rootpy.tree import Tree, Cut
from rootpy import asrootpy

# local imports
from . import log; log = log[__name__]
from . import categories
from . import variables
from .periods import LUMI
from .systematics import *

# Higgs cross sections
import yellowhiggs

VERBOSE = False

NTUPLE_PATH = os.getenv('HIGGSTAUTAU_NTUPLE_DIR')
if not NTUPLE_PATH:
    sys.exit("You did not source higgtautau/setup.sh")
NTUPLE_PATH = os.path.join(NTUPLE_PATH, 'prod')

DEFAULT_STUDENT = 'HHProcessor'
TAUTAUHADHADBR = 0.4197744 # = (1. - 0.3521) ** 2
DB_HH = datasets.Database(name='datasets_hh', verbose=VERBOSE)
DB_TAUID = datasets.Database(name='datasets_tauid', verbose=VERBOSE)
FILES = {}
OS = Cut('tau1_charge * tau2_charge == -1')
NOT_OS = Cut('tau1_charge * tau2_charge >= 0') # changed by Zinonas
SS = Cut('tau1_charge * tau2_charge == 1')
# mass_jet1_jet2 > 100000
TEMPFILE = ropen('tempfile.root', 'recreate')


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
    os.unlink(TEMPFILE.GetName())
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
        self.hist_decor['fillstyle'] = 'solid'

    def get_histfactory_sample(self, expr, category, region,
                               bins, min, max,
                               cuts=None):

        log.info("creating histfactory sample for %s" % self.name)
        sample = ROOT.RooStats.HistFactory.Sample(self.name)
        ndim = 1
        if ':' in expr:
            hist = self.draw2d(expr, category, region,
                               bins, min, max,
                               bins, min, max,
                               cuts)
            if isinstance(self, MC) and self.systematics:
                syst = hist.systematics
            hist = hist.ravel()
            if isinstance(self, MC) and self.systematics:
                hist.systematics = syst
            ndim = 2
        else:
            hist = self.draw(expr, category, region, bins, min, max, cuts)
        log.info(str(list(hist)))
        # set the nominal histogram
        sample.SetHisto(hist)
        log.info("nominal hist integral: %f" % hist.Integral())
        if isinstance(self, MC):
            if self.systematics:
                for sys_name, terms in SYSTEMATICS.items():
                    # add systematics terms
                    if len(terms) == 1:
                        up_term = terms[0]
                        down_term = terms[0]
                    else:
                        up_term, down_term = terms
                    log.info("adding histosys for %s" % sys_name)
                    histsys = ROOT.RooStats.HistFactory.HistoSys(sys_name)

                    hist_up = hist.systematics[up_term]
                    hist_down = hist.systematics[down_term]

                    if ndim == 2:
                        hist_up = hist_up.ravel()
                        hist_down = hist_down.ravel()

                    histsys.SetHistoHigh(hist_up)
                    histsys.SetHistoLow(hist_down)

                    sample.AddHistoSys(histsys)
            if isinstance(self, Signal):
                log.info("defining SigXsecOverSM POI")
                sample.AddNormFactor('SigXsecOverSM', 0., 0., 60.)
            else:
                log.info("activating stat error")
                sample.ActivateStatError()
        if hasattr(self, 'histfactory'):
            # perform sample-specific histfactory operations
            self.histfactory(sample)
        return sample

    def split(self,
              fields,
              category,
              region,
              cuts=None,
              systematic='NOMINAL'):

        if 'weight' not in fields:
            fields = fields + ['weight']

        # split using parity of EventNumber
        arrays = self.tables(
                category,
                region,
                fields=fields,
                include_weight=True,
                cuts=cuts,
                systematic=systematic)

        return np.hstack(map(itemgetter(0), arrays)), \
               np.hstack(map(itemgetter(1), arrays))

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

    def get_weight_branches(self, systematic, no_cuts=False):

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
            sys_cut &= categories.P1P3_RECOUNTED
        if systematic is not None:
            systerm, variation = Sample.get_sys_term_variation(systematic)
            if isinstance(self, Embedded_Ztautau):
                for term, variations in EMBEDDING_SYSTEMATICS.items():
                    if term == systerm:
                        sys_cut &= variations[variation]
                    else:
                        sys_cut &= variations['NOMINAL']

        if category in categories.CATEGORIES:
            return (categories.CATEGORIES[category]['cuts'] &
                    categories.CATEGORIES[category]['year_cuts'][self.year] &
                    Sample.REGIONS[region] & self._cuts & sys_cut)
        elif category in categories.CONTROLS:
            return (categories.CONTROLS[category]['cuts'] &
                    categories.CONTROLS[category]['year_cuts'][self.year] &
                    Sample.REGIONS[region] & self._cuts & sys_cut)
        else:
            raise ValueError(
                    'no such category or control region: %s' % category)

    def draw(self, expr, category, region, bins, min, max, cuts=None, p1p3=True):

        hist = Hist(bins, min, max, title=self.label, **self.hist_decor)
        self.draw_into(hist, expr, category, region, cuts=cuts, p1p3=p1p3)
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
        self.h5data = []
        for i in xrange(2):
            self.h5data.append(
                getattr(h5file.root, dataname + "_%d" % i))

        self.label = ('%s Data $\sqrt{s} = %d$ TeV\n'
                      '$\int L dt = %.2f$ fb$^{-1}$' % (
                          self.year, self.energy, LUMI[self.year] / 1e3))
        self.name = 'Data'

    def draw_into(self, hist, expr, category, region, cuts=None, p1p3=True):

        self.data.draw(expr, self.cuts(category, region, p1p3=p1p3) & cuts, hist=hist)

    def scores(self, clf, region, cuts=None):

        return clf.classify(self,
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

    def tables(self,
               category,
               region,
               fields=None,
               cuts=None,
               include_weight=True,
               systematic='NOMINAL',
               p1p3=True):

        Sample.check_systematic(systematic)
        selection = self.cuts(category, region, p1p3=p1p3) & cuts

        log.info("requesting table from %s %d with selection: %s" %
                 (self.__class__.__name__, self.year, selection))

        # read the table with a selection
        tables = []
        for partition in self.h5data:
            table = partition.readWhere(selection.where())
            # add weight field
            if include_weight:
                # data is not weighted
                weights = np.ones(table.shape[0], dtype='f4')
                table = recfunctions.rec_append_fields(table,
                        names='weight',
                        data=weights,
                        dtypes='f4')
            if fields is not None:
                table = table[fields]
            tables.append(table)
        return [tables]


class Signal:
    pass


class Background:
    pass


class MC(Sample):

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
            tables['NOMINAL'] = (
                    getattr(h5file.root, treename + "_0"),
                    getattr(h5file.root, treename + "_1"))

            weighted_events['NOMINAL'] = rfile.Get(
                    treename + events_hist_suffix)[events_bin]

            if self.systematics:

                systematics_terms, systematics_samples = \
                    samples_db.get_systematics('hadhad', self.year, name)

                unused_terms = SYSTEMATICS_TERMS[:]

                if False and systematics_terms:
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
                        tables[sys_term] = (
                                getattr(h5file.root, sys_name + "_0"),
                                getattr(h5file.root, sys_name + "_1"))

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
                        tables[sys_term] = (
                                getattr(h5file.root, sample_name + "_0"),
                                getattr(h5file.root, sample_name + "_1"))

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
        if self.scale != 1. and not isinstance(self,
                (MC_Ztautau, Embedded_Ztautau)):
            l += r' ($\sigma_{SM} \times %g$)' % self.scale
        return l

    def draw_into(self, hist, expr, category, region, cuts=None, p1p3=True):

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
                 ' * '.join(map(str, self.get_weight_branches('NOMINAL'))),
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
                             self.get_weight_branches('NOMINAL'))),
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

    def scores(self, clf, region, cuts=None, scores_dict=None):

        if scores_dict is None:
            scores_dict = {}

        for systematic in iter_systematics(True):

            if not self.systematics and systematic != 'NOMINAL':
                continue

            scores, weights = clf.classify(self,
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

    def tables(self,
               category,
               region,
               fields=None,
               cuts=None,
               include_weight=True,
               systematic='NOMINAL',
               p1p3=True):

        selection = self.cuts(category, region, systematic, p1p3) & cuts

        if systematic == 'NOMINAL':
            log.info("requesting table from %s with selection: %s" %
                    (self.__class__.__name__, selection))
        else:
            log.debug("requesting table from %s for systematic %s "
                      "with selection: %s" %
                      (self.__class__.__name__, systematic, selection))

        weight_branches = self.get_weight_branches(systematic, no_cuts=True)
        if systematic in SYSTEMATICS_BY_WEIGHT:
            systematic = 'NOMINAL'
        tables = []
        for ds, sys_trees, sys_tables, sys_events, xs, kfact, effic in self.datasets:

            if systematic in (('ZFIT_UP',), ('ZFIT_DOWN',),
                              ('QCDFIT_UP',), ('QCDFIT_DOWN',)):
                left_table, right_table = sys_tables['NOMINAL']
                events = sys_events['NOMINAL']
            else:
                table = sys_tables[systematic]
                events = sys_events[systematic]

                if table is None:
                    left_table, right_table = sys_tables['NOMINAL']
                    events = sys_events['NOMINAL']
                else:
                    left_table, right_table = table

            scale = self.scale
            if isinstance(self, Ztautau):
                if systematic == ('ZFIT_UP',):
                    scale = self.scale + self.scale_error
                elif systematic == ('ZFIT_DOWN',):
                    scale = self.scale - self.scale_error
            weight = scale * LUMI[self.year] * xs * kfact * effic / events

            table_selection = selection.where()
            log.debug(table_selection)

            partitions = []
            for i, table in enumerate([left_table, right_table]):
                # read the table with a selection
                table = table.readWhere(table_selection)

                # add weight field
                if include_weight:
                    weights = np.empty(table.shape[0], dtype='f4')
                    weights.fill(weight)
                    table = recfunctions.rec_append_fields(table,
                            names='weight',
                            data=weights,
                            dtypes='f4')
                    # merge the weight fields
                    table['weight'] *= reduce(np.multiply,
                            [table[br] for br in weight_branches])
                    # drop other weight fields
                    table = recfunctions.rec_drop_fields(table, weight_branches)
                if fields is not None:
                    table = table[fields]
                partitions.append(table)
            tables.append(partitions)
        return tables

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


class Ztautau:

    pass


class MC_Ztautau(MC, Ztautau, Background):

    def __init__(self, *args, **kwargs):
        """
        Instead of setting the k factor here
        the normalization is determined by a fit to the data
        """
        self.scale_error = 0.
        super(MC_Ztautau, self).__init__(
                *args, **kwargs)


class Embedded_Ztautau(MC, Ztautau, Background):

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

    MODES = ['gg', 'VBF', 'Z', 'W']

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


class QCD(Sample):

    def __init__(self, data, mc,
                 scale=1.,
                 shape_region='SS',
                 cuts=None,
                 color='#59d454'):

        assert len(mc) > 0
        assert data.year == mc[0].year
        super(QCD, self).__init__(year=data.year, scale=scale, color=color)
        self.data = data
        self.mc = mc
        self.name = 'QCD'
        self.label = 'QCD Multi-jet'
        self.scale = 1.
        self.scale_error = 0.
        self.shape_region = shape_region

    def draw_into(self, hist, expr, category, region, cuts=None, p1p3=True):

        MC_bkg_notOS = hist.Clone()
        for mc in self.mc:
            mc.draw_into(MC_bkg_notOS, expr, category, self.shape_region,
                         cuts=cuts, p1p3=p1p3)

        data_hist = hist.Clone()
        self.data.draw_into(data_hist, expr,
                            category, self.shape_region, cuts=cuts, p1p3=p1p3)

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

    def scores(self,
               clf,
               region,
               cuts=None):

        # SS data
        data_scores, data_weights = self.data.scores(
                clf,
                region=self.shape_region,
                cuts=cuts)

        scores_dict = {}
        # subtract SS MC
        for mc in self.mc:
            mc.scores(
                    clf,
                    region=self.shape_region,
                    cuts=cuts,
                    scores_dict=scores_dict)

        for sys_term in scores_dict.keys():
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

    def tables(self,
            category,
            region,
            fields=None,
            cuts=None,
            include_weight=True,
            systematic='NOMINAL'):

        assert include_weight == True

        data_tables = self.data.tables(
                category=category,
                region=self.shape_region,
                fields=fields,
                cuts=cuts,
                include_weight=include_weight,
                systematic='NOMINAL')
        arrays = data_tables

        for mc in self.mc:
            _arrays = mc.tables(
                    category=category,
                    region=self.shape_region,
                    fields=fields,
                    cuts=cuts,
                    include_weight=include_weight,
                    systematic=systematic)
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


if __name__ == '__main__':

    from background_estimation import qcd_ztautau_norm

    # tests
    category='boosted'
    shape_region = '!OS'
    target_region = 'OS'

    ztautau = MC_Ztautau(year=2011, systematics=False)
    others = Others(year=2011, systematics=False)
    data = Data(year=2011)
    qcd = QCD(data=data, mc=[others, ztautau],
          shape_region=shape_region)

    qcd_scale, qcd_scale_error, ztautau_scale, ztautau_scale_error = qcd_ztautau_norm(
        year=2011,
        ztautau=ztautau,
        backgrounds=[others],
        data=data,
        category=category,
        target_region=target_region,
        qcd_shape_region=shape_region,
        use_cache=True)

    qcd.scale = qcd_scale
    qcd.scale_error = qcd_scale_error
    ztautau.scale = ztautau_scale
    ztautau.scale_error = ztautau_scale_error

    expr = 'tau1_BDTJetScore'
    cuts = None
    bins = 20
    min, max = -0.5, 1.5

    print '--- Others'
    other_hist = others.draw(
            expr,
            category, target_region,
            bins, min, max,
            cuts=cuts)
    print "--- QCD"
    qcd_hist = qcd.draw(
            expr,
            category, target_region,
            bins, min, max,
            cuts=cuts)
    print '--- Z'
    ztautau_hist = ztautau.draw(
            expr,
            category, target_region,
            bins, min, max,
            cuts=cuts)
    print '--- Data'
    data_hist = data.draw(
            expr,
            category, target_region,
            bins, min, max,
            cuts=cuts)

    print "Data: %f" % sum(data_hist)
    print "QCD: %f" % sum(qcd_hist)
    print "Z: %f" % sum(ztautau_hist)
    print "Others: %f" % sum(other_hist)
    print "Data / Model: %f" % (sum(data_hist) / (sum(qcd_hist) +
        sum(ztautau_hist) + sum(other_hist)))

    # test scores
    from categories import CATEGORIES
    import pickle
    import numpy as np

    branches = CATEGORIES[category]['features']

    train_frac = .5

    with open('clf_%s.pickle' % category, 'r') as f:
        clf = pickle.load(f)
        print clf
    print '--- Others'
    other_scores, other_weights = others.scores(
            clf, branches,
            train_frac,
            category, target_region,
            cuts=cuts)['NOMINAL']
    print '--- QCD'
    qcd_scores, qcd_weights = qcd.scores(
            clf, branches,
            train_frac,
            category, target_region,
            cuts=cuts)['NOMINAL']
    print '--- Z'
    ztautau_scores, ztautau_weights = ztautau.scores(
            clf, branches,
            train_frac,
            category, target_region,
            cuts=cuts)['NOMINAL']
    print '--- Data'
    data_scores, data_weights = data.scores(
            clf, branches,
            train_frac,
            category, target_region,
            cuts=cuts)

    print "Data: %d" % (len(data_scores))
    print "QCD: %f" % np.sum(qcd_weights)
    print "Z: %f" % np.sum(ztautau_weights)
    print "Others: %f" % np.sum(other_weights)
