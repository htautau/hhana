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
from rootpy.fit import histfactory

# local imports
from . import log; log = log[__name__]
from . import variables
from . import NTUPLE_PATH, DEFAULT_STUDENT
from .utils import print_hist, rec_to_ndarray
from .lumi import LUMI
from .systematics import *
from .constants import *
from .classify import histogram_scores
from .stats.utils import kylefix, statsfix, zero_negs, smooth
from .cachedtable import CachedTable
from .regions import REGIONS
from .lumi import get_lumi_uncert

# Higgs cross sections
import yellowhiggs

VERBOSE = False

DB_HH = datasets.Database(name='datasets_hh', verbose=VERBOSE)
FILES = {}


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

    WORKSPACE_SYSTEMATICS = []

    def __init__(self, year, scale=1., cuts=None,
                 student=DEFAULT_STUDENT,
                 root=False,
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

        self.root = root
        self.student = student
        self.hist_decor = hist_decor
        #if isinstance(self, Higgs):
        #    self.hist_decor['fillstyle'] = 'hollow'
        #else:
        if 'fillstyle' not in hist_decor:
            self.hist_decor['fillstyle'] = 'solid'

    @property
    def label(self):

        if self.root:
            return self._label_root
        return self._label

    def get_hist(self,
            hist_template,
            expr_or_clf,
            category, region,
            cuts=None,
            clf=None,
            scores=None,
            min_score=None,
            max_score=None,
            systematics=True,
            suffix=None,
            field_scale=None,
            weight_hist=None):

        do_systematics = (not isinstance(self, Data)
                          and self.systematics
                          and systematics)

        ndim = hist_template.GetDimension()

        if min_score is None:
            min_score = getattr(category, 'workspace_min_clf', None)
        if max_score is None:
            max_score = getattr(category, 'workspace_max_clf', None)

        histname = 'category_{0}_{1}'.format(category.name, self.name)
        if suffix is not None:
            histname += suffix
        hist = hist_template.Clone(name=histname)
        hist.Reset()

        if isinstance(expr_or_clf, (basestring, tuple, list)):
            expr = expr_or_clf
            field_hist = dict()
            field_hist[expr] = hist
            self.draw_array(field_hist, category, region,
                cuts=cuts,
                weighted=True,
                field_scale=field_scale,
                weight_hist=weight_hist,
                clf=clf,
                min_score=min_score,
                max_score=max_score,
                systematics=do_systematics,
                systematics_components=self.WORKSPACE_SYSTEMATICS)

            if ndim == 2:
                # convert to 1D hist
                rhist = hist.ravel(name = hist.name + '_ravel')
                if do_systematics:
                    rsyst = {}
                    for term, syshist in hist.systematics.items():
                        rsyst[term] = syshist.ravel(name=syshist.name + '_ravel')
                    rhist.systematics = rsyst
                hist = rhist

        else:
            # histogram classifier output
            if scores is None:
                scores = self.scores(
                    expr_or_clf, category, region, cuts,
                    systematics=do_systematics,
                    systematics_components=self.WORKSPACE_SYSTEMATICS)
            histogram_scores(
                hist, scores,
                min_score=min_score,
                max_score=max_score,
                inplace=True)

        # convert to uniform binning and zero out negative bins
        hist = statsfix(hist, fix_systematics=True)

        # always apply kylefix on backgrounds
        if (isinstance(self, Background) and
            not getattr(self, 'NO_KYLEFIX', False)):
            log.info("applying kylefix()")
            # TODO also apply kylefix on systematics?
            # IMPORTANT: if kylefix is not applied on systematics, normalization
            # can be inconsistent between nominal and systematics creating a
            # bias in the OverallSys when separating the variation into
            # normalization and shape components!
            hist = kylefix(hist, fix_systematics=True)

        print_hist(hist)
        return hist

    def get_histfactory_sample(self,
            hist_template,
            expr_or_clf,
            category, region,
            cuts=None,
            clf=None,
            scores=None,
            min_score=None,
            max_score=None,
            systematics=True,
            suffix=None,
            field_scale=None,
            weight_hist=None):

        log.info("creating histfactory sample for {0}".format(self.name))

        if isinstance(self, Data):
            sample = histfactory.Data(self.name)
        else:
            sample = histfactory.Sample(self.name)

        hist = self.get_hist(
            hist_template,
            expr_or_clf,
            category, region,
            cuts=cuts,
            clf=clf,
            scores=scores,
            min_score=min_score,
            max_score=max_score,
            systematics=systematics,
            suffix=suffix,
            field_scale=field_scale,
            weight_hist=weight_hist)

        # set the nominal histogram
        sample.hist = hist
        sample.histname = hist.name

        do_systematics = (not isinstance(self, Data)
                          and self.systematics
                          and systematics)

        # add systematics samples
        if do_systematics:

            SYSTEMATICS = get_systematics(self.year)

            for sys_component in self.WORKSPACE_SYSTEMATICS:

                log.info("adding histosys for %s" % sys_component)

                terms = SYSTEMATICS[sys_component]
                if len(terms) == 1:
                    up_term = terms[0]
                    hist_up = hist.systematics[up_term]
                    # use nominal hist for "down" side
                    hist_down = hist
                    # smooth the shape systematics
                    hist_up = smooth(hist, hist_up)

                else:
                    up_term, down_term = terms
                    hist_up = hist.systematics[up_term]
                    hist_down = hist.systematics[down_term]
                    # smooth the shape systematics
                    hist_up = smooth(hist, hist_up)
                    hist_down  smooth(hist, hist_down)

                if sys_component == 'JES_FlavComp':
                    if ((isinstance(self, Signal) and self.mode == 'gg') or
                         isinstance(self, Others)):
                        sys_component += '_G'
                    else:
                        sys_component += '_Q'

                elif sys_component == 'JES_PURho':
                    if isinstance(self, Others):
                        sys_component += '_QG'
                    elif isinstance(self, Signal):
                        if self.mode == 'gg':
                            sys_component += '_GG'
                        else:
                            sys_component += '_QQ'

                histsys = histfactory.HistoSys(
                    'ATLAS_{0}_{1:d}'.format(sys_component, self.year),
                    low=hist_down,
                    high=hist_up)

                norm, shape = histfactory.split_norm_shape(histsys, hist)

                # drop norm on backgrounds if < 1% and signals if < .5%
                if (isinstance(self, Background) and (
                        norm.high >= 1.01 or norm.low <= 0.99)) or (
                    isinstance(self, Signal) and (
                        norm.high >= 1.005 or norm.low <= 0.995)):
                    sample.AddOverallSys(norm)

                # drop all jet related shape terms from Others (JES, JVF, JER)
                if isinstance(self, Others) and (
                        sys_component.startswith('JES') or
                        sys_component.startswith('JVF') or
                        sys_component.startswith('JER')):
                    continue

                # if you fit the ratio of nominal to up / down to a "pol0" and
                # get reasonably good chi2, then you may consider dropping the
                # histosys part
                sample.AddHistoSys(shape)

        if isinstance(self, QCD):
            log.info("adding QCD shape systematic")
            curr_model = self.shape_region
            # add QCD shape systematic
            if isinstance(expr_or_clf, tuple):
                # OSFF x (SS / SSFF) model in the track-fit category
                assert curr_model == 'SS'
                models = []
                for model in ('OSFF', 'SSFF'):
                    log.info("getting QCD shape for {0}".format(model))
                    self.shape_region = model
                    models.append(self.get_hist(
                        hist_template,
                        expr_or_clf,
                        category, region,
                        cuts=cuts,
                        clf=clf,
                        scores=None,
                        min_score=min_score,
                        max_score=max_score,
                        systematics=False,
                        suffix=(suffix or '') + '_%s' % model,
                        field_scale=field_scale,
                        weight_hist=weight_hist))
                OSFF, SSFF = models
                shape_sys = OSFF
                shape_sys *= sample.hist / SSFF

            else:
                # SS_TRK model elsewhere
                assert curr_model == 'nOS'
                self.shape_region = 'SS_TRK'
                log.info("getting QCD shape for SS_TRK")
                shape_sys = self.get_hist(
                    hist_template,
                    expr_or_clf,
                    category, region,
                    cuts=cuts,
                    clf=clf,
                    scores=None,
                    min_score=min_score,
                    max_score=max_score,
                    systematics=False,
                    suffix=(suffix or '') + '_SS_TRK',
                    field_scale=field_scale,
                    weight_hist=weight_hist)

            # restore previous shape model
            self.shape_region = curr_model

            # normalize shape_sys to the same integral as the nominal shape
            shape_sys *= sample.hist.Integral() / shape_sys.Integral()

            # smooth the shape systematic
            shape_sys = smooth(sample.hist, shape_sys)

            # reflect shape about the nominal to get high and low variations
            shape_sys_reflect = sample.hist + (sample.hist - shape_sys)
            shape_sys_reflect.name = shape_sys.name + '_reflected'
            shape_sys_reflect = zero_negs(shape_sys_reflect)

            histsys = histfactory.HistoSys(
                'ATLAS_HADHAD_QCD_SHAPE{0}_{1:d}'.format(
                    '_CONTROL' if category.analysis_control else '',
                    self.year),
                low=shape_sys,
                high=shape_sys_reflect)

            sample.AddHistoSys(histsys)

        if isinstance(self, Signal):
            log.info("defining SigXsecOverSM POI for %s" % self.name)
            sample.AddNormFactor('SigXsecOverSM', 0., 0., 200., False)

        elif isinstance(self, Background):
            # only activate stat error on background samples
            log.info("activating stat error for %s" % self.name)
            sample.ActivateStatError()

        if not isinstance(self, Data):
            norm_by_theory = getattr(self, 'NORM_BY_THEORY', True)
            sample.SetNormalizeByTheory(norm_by_theory)
            if norm_by_theory:
                lumi_uncert = get_lumi_uncert(self.year)
                lumi_sys = histfactory.OverallSys(
                    'ATLAS_LUMI_{0:d}'.format(self.year),
                    high=1. + lumi_uncert,
                    low=1. - lumi_uncert)
                sample.AddOverallSys(lumi_sys)

        if hasattr(self, 'histfactory'):
            # perform sample-specific items
            log.info("calling %s histfactory method" % self.name)
            self.histfactory(sample, category, systematics=do_systematics)

        return sample

    def partitioned_records(self,
              category,
              region,
              fields=None,
              cuts=None,
              include_weight=True,
              systematic='NOMINAL',
              key=None,
              num_partitions=2,
              return_idx=False):
        """
        Partition sample into num_partitions chunks of roughly equal size
        assuming no correlation between record index and field values.
        """
        partitions = []
        for start in range(num_partitions):
            if key is None:
                # split by index
                log.info("splitting records by index parity")
                recs = self.records(
                    category,
                    region,
                    fields=fields,
                    include_weight=include_weight,
                    cuts=cuts,
                    systematic=systematic,
                    return_idx=return_idx,
                    start=start,
                    step=num_partitions)
            else:
                # split by field values modulo the number of partitions
                partition_cut = Cut('((abs({0})%{1})>={2})&&((abs({0})%{1})<{3})'.format(
                    key, num_partitions, start, start + 1))
                log.info(
                    "splitting records by key parity: {0}".format(
                        partition_cut))
                recs = self.records(
                    category,
                    region,
                    fields=fields,
                    include_weight=include_weight,
                    cuts=partition_cut & cuts,
                    systematic=systematic,
                    return_idx=return_idx)

            if return_idx:
                partitions.append(recs)
            else:
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
            systerm, variation = systematic[0].rsplit('_', 1)
        return systerm, variation

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
            ]
            if isinstance(self, Embedded_Ztautau):
                weight_branches.append('embedding_reco_unfold')
            for term, variations in WEIGHT_SYSTEMATICS.items():
                if term == systerm:
                    weight_branches += variations[variation]
                else:
                    weight_branches += variations['NOMINAL']
        else:
            weight_branches = []
        if not no_cuts and isinstance(self, Embedded_Ztautau):
            for term, variations in EMBEDDING_SYSTEMATICS.items():
                if term == systerm:
                    if variations[variation]:
                        weight_branches.append(variations[variation])
                else:
                    if variations['NOMINAL']:
                        weight_branches.append(variations['NOMINAL'])
        #log.info("weight fields: {0}".format(', '.join(weight_branches)))
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

    def cuts(self, category, region, systematic='NOMINAL', **kwargs):

        sys_cut = Cut()
        if isinstance(self, Embedded_Ztautau):
            systerm, variation = Sample.get_sys_term_variation(systematic)
            for term, variations in EMBEDDING_SYSTEMATICS.items():
                if term == systerm:
                    sys_cut &= variations[variation]
                else:
                    sys_cut &= variations['NOMINAL']
        return (category.get_cuts(self.year, **kwargs) &
                REGIONS[region] & self._cuts & sys_cut)

    def draw(self, expr, category, region, bins, min, max,
             cuts=None, weighted=True, systematics=True):

        hist = Hist(bins, min, max, title=self.label, **self.hist_decor)
        self.draw_into(hist, expr, category, region,
                       cuts=cuts, weighted=weighted,
                       systematics=systematics)
        return hist

    def draw2d(self, expr, category, region,
               xbins, xmin, xmax,
               ybins, ymin, ymax,
               cuts=None,
               systematics=True,
               ravel=False):

        hist = Hist2D(xbins, xmin, xmax, ybins, ymin, ymax,
                title=self.label, **self.hist_decor)
        self.draw_into(hist, expr, category, region, cuts=cuts,
                systematics=systematics)
        if ravel:
            rhist = hist.ravel()
            if hasattr(hist, 'sytematics'):
                rhist.systematics = {}
                for term, syshist in hist.systematics.items():
                    rhist.systematics[term] = syshist.ravel()
            return rhist
        return hist

    def draw_array_helper(self, field_hist, category, region,
                          cuts=None,
                          weighted=True,
                          field_scale=None,
                          weight_hist=None,
                          scores=None,
                          min_score=None,
                          max_score=None,
                          systematic='NOMINAL'):

        all_fields = []
        for f in field_hist.iterkeys():
            if isinstance(f, basestring):
                all_fields.append(f)
            else:
                all_fields.extend(list(f))

        # TODO: only get unblinded vars
        rec = self.merged_records(category, region,
            fields=all_fields, cuts=cuts,
            include_weight=True,
            systematic=systematic)

        if scores is not None:
            # sanity
            assert (scores[1] == rec['weight']).all()
            # ignore the score weights since they should be the same as the rec
            # weights
            scores = scores[0]

        if weight_hist is not None and scores is not None:
            edges = np.array(list(weight_hist.xedges()))
            weights = np.array(weight_hist).take(
                edges.searchsorted(scores) - 1)
            weights = rec['weight'] * weights
        else:
            weights = rec['weight']

        if scores is not None:
            if min_score is not None:
                idx = scores > min_score
                rec = rec[idx]
                weights = weights[idx]
                scores = scores[idx]

            if max_score is not None:
                idx = scores < max_score
                rec = rec[idx]
                weights = weights[idx]
                scores = scores[idx]

        for fields, hist in field_hist.items():
            # fields can be a single field or list of fields
            if not isinstance(fields, (list, tuple)):
                fields = [fields]
            if hist is None:
                # this var might be blinded
                continue
            # defensive copy
            if isinstance(fields, tuple):
                # select columns in numpy recarray with a list
                fields = list(fields)
            arr = np.copy(rec[fields])
            if field_scale is not None:
                for field in fields:
                    if field in field_scale:
                        arr[field] *= field_scale[field]
            # convert to array
            arr = rec_to_ndarray(arr, fields=fields)
            # include the scores if the histogram dimensionality allows
            if scores is not None and hist.GetDimension() == len(fields) + 1:
                arr = np.c_[arr, scores]
            elif hist.GetDimension() != len(fields):
                raise TypeError(
                    'histogram dimensionality does not match '
                    'number of fields: %s' % (', '.join(fields)))
            #print self.name, systematic, fields, sum(weights)
            #print list(hist)
            hist.fill_array(arr, weights=weights)
            #print list(hist)
            #print "=" * 80


class Data(Sample):

    def __init__(self, year, markersize=1.2, **kwargs):

        super(Data, self).__init__(
            year=year, scale=1.,
            markersize=markersize, **kwargs)
        rfile = get_file(self.student)
        h5file = get_file(self.student, hdf=True)
        dataname = 'data%d_JetTauEtmiss' % (year % 1E3)
        self.data = getattr(rfile, dataname)
        self.h5data = CachedTable.hook(getattr(h5file.root, dataname))

        self._label = ('%s Data $\sqrt{s} = %d$ TeV\n'
                       '$\int dt L = %.1f$ fb$^{-1}$' % (
                       self.year, self.energy, LUMI[self.year] / 1e3))
        self._label_root = ('%s Data #sqrt{s} = %d TeV\n'
                            '#scale[0.7]{#int}dt L = %.1f fb^{-1}' % (
                            self.year, self.energy, LUMI[self.year] / 1e3))

        self.name = 'Data'

    def events(self, category, region, cuts=None, raw=False):

        selection = self.cuts(category, region) & cuts
        log.debug("requesting number of events from %s using cuts: %s" %
                  (self.data.GetName(), selection))
        return self.data.GetEntries(selection)

    def draw_into(self, hist, expr, category, region,
                  cuts=None, weighted=True, systematics=True):

        self.data.draw(expr, self.cuts(category, region) & cuts, hist=hist)

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
                   systematics_components=None):

        if scores is None and clf is not None:
            scores = self.scores(
                clf, category, region, cuts=cuts)

        self.draw_array_helper(field_hist, category, region,
            cuts=cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            scores=scores,
            min_score=min_score,
            max_score=max_score)

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

        Sample.check_systematic(systematic)
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

        Sample.check_systematic(systematic)
        selection = self.cuts(category, region) & cuts

        log.info("requesting table from Data %d" % self.year)
        log.debug("using selection: %s" % selection)

        # read the table with a selection
        rec = self.h5data.read_where(selection.where(), **kwargs)

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

        if return_idx:
            idx = self.h5data.get_where_list(selection.where(), **kwargs)
            return [(rec, idx)]
        return [rec]


class Signal(object):
    # mixin
    pass


class Background(object):
    # mixin
    pass


class MC(Sample):

    # TODO: remove 'JE[S|R]' here unless embedded classes should inherit from
    # elsewhere
    WORKSPACE_SYSTEMATICS = Sample.WORKSPACE_SYSTEMATICS + [
        #'JES',
        'JES_Modelling',
        'JES_Detector',
        'JES_EtaModelling',
        'JES_EtaMethod',
        'JES_PURho',
        'JES_FlavComp',
        'JES_FlavResp',
        'JVF',
        'JER',
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

                if systematics_terms:
                    for sys_term in systematics_terms:

                        sys_name = treename + '_' + '_'.join(sys_term)
                        trees[sys_term] = rfile.Get(sys_name)
                        tables[sys_term] = CachedTable.hook(getattr(
                            h5file.root, sys_name))
                        weighted_events[sys_term] = rfile.Get(
                                sys_name + events_hist_suffix)[events_bin]

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

    def draw_into(self, hist, expr, category, region,
                  cuts=None,
                  weighted=True,
                  systematics=True,
                  systematics_components=None,
                  scale=1.):

        if isinstance(expr, (list, tuple)):
            exprs = expr
        else:
            exprs = (expr,)

        do_systematics = self.systematics and systematics

        if do_systematics:
            if hasattr(hist, 'systematics'):
                sys_hists = hist.systematics
            else:
                sys_hists = {}

        selection = self.cuts(category, region) & cuts

        for ds, sys_trees, _, sys_events, xs, kfact, effic in self.datasets:

            log.debug(ds.name)

            nominal_tree = sys_trees['NOMINAL']
            nominal_events = sys_events['NOMINAL']

            nominal_weight = (
                LUMI[self.year] *
                scale * self.scale *
                xs * kfact * effic / nominal_events)

            nominal_weighted_selection = (
                '%f * %s * (%s)' %
                (nominal_weight,
                 '*'.join(map(str,
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

            if not do_systematics:
                continue

            # iterate over systematic variations skipping the nominal
            for sys_term in iter_systematics(False,
                    components=systematics_components):

                sys_hist = current_hist.Clone()

                if sys_term in sys_trees:
                    sys_tree = sys_trees[sys_term]
                    sys_event = sys_events[sys_term]
                    sys_hist.Reset()

                    sys_weight = (
                        LUMI[self.year] *
                        scale * self.scale *
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

                else:
                    log.debug(
                        "tree for %s not present for %s "
                        "using NOMINAL" % (sys_term, ds.name))

                    # QCD + Ztautau fit error
                    if isinstance(self, Ztautau):
                        if sys_term == ('ZFIT_UP',):
                            sys_hist *= (
                                    (self.scale + self.scale_error) /
                                    self.scale)
                        elif sys_term == ('ZFIT_DOWN',):
                            sys_hist *= (
                                    (self.scale - self.scale_error) /
                                    self.scale)

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

        if do_systematics:
            # set the systematics
            hist.systematics = sys_hists

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
                   scale=1.):

        do_systematics = self.systematics and systematics

        if scores is None and clf is not None:
            scores = self.scores(
                clf, category, region, cuts=cuts,
                systematics=systematics)

        self.draw_array_helper(field_hist, category, region,
            cuts=cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            scores=scores['NOMINAL'] if scores else None,
            min_score=min_score,
            max_score=max_score,
            systematic='NOMINAL')

        if not do_systematics:
            return

        all_sys_hists = {}

        for field, hist in field_hist.items():
            if not hasattr(hist, 'systematics'):
                hist.systematics = {}
            all_sys_hists[field] = hist.systematics

        for systematic in iter_systematics(False,
                components=systematics_components):

            sys_field_hist = {}
            for field, hist in field_hist.items():
                if systematic in all_sys_hists[field]:
                    sys_hist = all_sys_hists[field][systematic]
                else:
                    sys_hist = hist.Clone(
                        name=hist.name + '_' + systematic_name(systematic))
                    sys_hist.Reset()
                    all_sys_hists[field][systematic] = sys_hist
                sys_field_hist[field] = sys_hist

            self.draw_array_helper(sys_field_hist, category, region,
                cuts=cuts,
                weighted=weighted,
                field_scale=field_scale,
                weight_hist=weight_hist,
                scores=scores[systematic] if scores else None,
                min_score=min_score,
                max_score=max_score,
                systematic=systematic)

        """
        print self.name
        for term, sys_hist in field_hist['tau1_pt'].systematics.items():
            print term, list(sys_hist)
        print "=" * 80
        """

    def scores(self, clf, category, region,
               cuts=None, scores_dict=None,
               systematics=True,
               systematics_components=None,
               scale=1.):

        # TODO check that weight systematics are included
        do_systematics = self.systematics and systematics

        if scores_dict is None:
            scores_dict = {}

        for systematic in iter_systematics(True,
                components=systematics_components):

            if not do_systematics and systematic != 'NOMINAL':
                continue

            scores, weights = clf.classify(self,
                category=category,
                region=region,
                cuts=cuts,
                systematic=systematic)

            weights *= scale

            if systematic not in scores_dict:
                scores_dict[systematic] = (scores, weights)
            else:
                prev_scores, prev_weights = scores_dict[systematic]
                scores_dict[systematic] = (
                    np.concatenate((prev_scores, scores)),
                    np.concatenate((prev_weights, weights)))
        return scores_dict

    def trees(self, category, region,
              cuts=None, systematic='NOMINAL',
              scale=1.):

        TEMPFILE.cd()
        selection = self.cuts(category, region) & cuts
        weight_branches = self.get_weight_branches(systematic)
        if systematic in SYSTEMATICS_BY_WEIGHT:
            systematic = 'NOMINAL'

        trees = []
        for ds, sys_trees, _, sys_events, xs, kfact, effic in self.datasets:

            try:
                tree = sys_trees[systematic]
                events = sys_events[systematic]
            except KeyError:
                log.debug(
                    "tree for %s not present for %s "
                    "using NOMINAL" % (systematic, ds.name))
                tree = sys_trees['NOMINAL']
                events = sys_events['NOMINAL']

            actual_scale = self.scale
            if isinstance(self, Ztautau):
                if systematic == ('ZFIT_UP',):
                    actual_scale = self.scale + self.scale_error
                elif systematic == ('ZFIT_DOWN',):
                    actual_scale = self.scale - self.scale_error

            weight = (
                scale * actual_scale *
                LUMI[self.year] *
                xs * kfact * effic / events)

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
                scale=1.,
                return_idx=False,
                **kwargs):

        if include_weight and fields is not None:
            if 'weight' not in fields:
                fields = fields + ['weight']

        selection = self.cuts(category, region, systematic) & cuts
        table_selection = selection.where()

        if systematic == 'NOMINAL':
            log.info("requesting table from %s" %
                     (self.__class__.__name__))
        else:
            log.info("requesting table from %s for systematic %s " %
                     (self.__class__.__name__, systematic_name(systematic)))
        log.debug("using selection: %s" % selection)

        # TODO: handle cuts in weight expressions
        weight_branches = self.get_weight_branches(systematic, no_cuts=True)
        if systematic in SYSTEMATICS_BY_WEIGHT:
            systematic = 'NOMINAL'

        recs = []
        if return_idx:
            idxs = []
        for ds, _, sys_tables, sys_events, xs, kfact, effic in self.datasets:

            try:
                table = sys_tables[systematic]
                events = sys_events[systematic]
            except KeyError:
                log.debug(
                    "table for %s not present for %s "
                    "using NOMINAL" % (systematic, ds.name))
                table = sys_tables['NOMINAL']
                events = sys_events['NOMINAL']

            actual_scale = self.scale
            if isinstance(self, Ztautau):
                if systematic == ('ZFIT_UP',):
                    log.debug("scaling up for ZFIT_UP")
                    actual_scale += self.scale_error
                elif systematic == ('ZFIT_DOWN',):
                    log.debug("scaling down for ZFIT_DOWN")
                    actual_scale -= self.scale_error

            weight = (
                scale * actual_scale *
                LUMI[self.year] *
                xs * kfact * effic / events)

            # read the table with a selection
            try:
                rec = table.read_where(table_selection, **kwargs)
            except Exception as e:
                print table
                print e
                continue
                #raise

            if return_idx:
                idx = table.get_where_list(table_selection, **kwargs)
                idxs.append(idx)

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
                if rec['weight'].shape[0] > 1 and rec['weight'].sum() == 0:
                    log.warning("{0}: weights sum to zero!".format(table.name))
                    for br in weight_branches:
                        log.warning("{0}: {1}".format(br, repr(rec[br])))
                # drop other weight fields
                rec = recfunctions.rec_drop_fields(rec, weight_branches)

            if fields is not None:
                try:
                    rec = rec[fields]
                except Exception as e:
                    print table
                    print rec.shape
                    print rec.dtype
                    print e
                    continue
                    #raise
            recs.append(rec)

        if return_idx:
            return zip(recs, idxs)
        return recs

    def events(self, category, region,
               cuts=None,
               systematic='NOMINAL',
               weighted=True,
               scale=1.,
               raw=False):

        total = 0.
        hist = Hist(1, -100, 100)
        for ds, sys_trees, _, sys_events, xs, kfact, effic in self.datasets:

            try:
                tree = sys_trees[systematic]
                events = sys_events[systematic]
            except KeyError:
                log.debug(
                    "tree for %s not present for %s "
                    "using NOMINAL" % (systematic, ds.name))
                tree = sys_trees['NOMINAL']
                events = sys_events['NOMINAL']

            if raw:
                selection = self.cuts(category, region, systematic=systematic) & cuts
                log.debug("requesing number of events from %s using cuts: %s"
                          % (tree.GetName(), selection))
                total += tree.GetEntries(selection)
            else:
                weight = LUMI[self.year] * self.scale * xs * kfact * effic / events
                weighted_selection = Cut(' * '.join(map(str,
                         self.get_weight_branches(systematic, weighted=weighted))))
                selection = Cut(str(weight)) * weighted_selection * (
                        self.cuts(category, region, systematic=systematic) & cuts)
                log.debug("requesing number of events from %s using cuts: %s"
                          % (tree.GetName(), selection))
                hist.Reset()
                curr_total = tree.Draw('1', selection, hist=hist)
                total += hist.Integral()
        return total * scale


class Ztautau(Background):

    NORM_BY_THEORY = False

    def histfactory(self, sample, category, systematics=True):

        sample.AddNormFactor('ATLAS_norm_Z_{0:d}'.format(self.year),
                             1., 0., 50., False)

    def __init__(self, *args, **kwargs):
        """
        Instead of setting the k factor here
        the normalization is determined by a fit to the data
        """
        self.scale_error = 0.
        super(Ztautau, self).__init__(*args, **kwargs)


class MC_Ztautau(Ztautau, MC):

    WORKSPACE_SYSTEMATICS = MC.WORKSPACE_SYSTEMATICS


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


class Higgs(MC, Signal):

    MASS_POINTS = range(100, 155, 5)

    MODES = ['Z', 'W', 'gg', 'VBF']

    MODES_DICT = {
        'gg': ('ggf', 'PowHegPythia_', 'PowHegPyth8_AU2CT10_'),
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

    NORM_BY_THEORY = True

    def histfactory(self, sample, category, systematics=True):
        if not systematics:
            return
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
            log.info("defining overall sys %s" % term)
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
        self._label = r'%sH%s$\rightarrow\tau_{h}\tau_{h}$' % (str_mode, str_mass)
        self._label_root = '%sH%s#rightarrow#tau_{h}#tau_{h}' % (str_mode, str_mass)

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

        if len(self.modes) == 1:
            self.mode = self.modes[0]
        else:
            self.mode = None
        if len(self.masses) == 1:
            self.mass = self.masses[0]
        else:
            self.mass = None

        super(Higgs, self).__init__(year=year, **kwargs)


class QCD(Sample, Background):

    # don't include MC systematics in workspace for QCD
    WORKSPACE_SYSTEMATICS = [] #MC.WORKSPACE_SYSTEMATICS
    NORM_BY_THEORY = False

    def histfactory(self, sample, category, systematics=True):

        sample.AddNormFactor('ATLAS_norm_QCD_{0:d}'.format(self.year),
                             1., 0., 50., False)

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
                 scale_error=0.,
                 data_scale=1.,
                 mc_scales=None,
                 shape_region='SS',
                 cuts=None,
                 color='#59d454',
                 root=False):

        QCD.sample_compatibility(data, mc)
        super(QCD, self).__init__(
            year=data.year,
            scale=scale,
            color=color,
            root=root)
        self.data = data
        self.mc = mc
        self.name = 'QCD'
        self._label = 'QCD Multi-jet (%s)' % shape_region.replace('_', ' ')
        self._label_root = self._label
        self.scale = 1.
        self.data_scale = data_scale
        if mc_scales is not None:
            if len(mc_scales) != len(mc):
                raise ValueError("length of MC scales must match number of MC")
            self.mc_scales = mc_scales
        else:
            # default scales to 1.
            self.mc_scales = [1. for m in self.mc]
        self.scale_error = scale_error
        self.shape_region = shape_region
        self.systematics = mc[0].systematics

    def events(self, category, region, cuts=None,
               systematic='NOMINAL',
               raw=False):

        data = self.data.events(category, self.shape_region, cuts=cuts)
        mc_subtract = 0.
        for mc_scale, mc in zip(self.mc_scales, self.mc):
            mc_subtract += mc.events(
                category, self.shape_region,
                cuts=cuts,
                systematic=systematic,
                raw=raw,
                scale=mc_scale)

        if raw:
            return self.data_scale * data + mc_subtract

        log.info("QCD: Data(%.3f) - MC(%.3f)" % (
            self.data_scale * data, mc_subtract))
        log.info("MC subtraction: %.1f%%" % (
            100. * mc_subtract / (self.data_scale * data)))

        return (self.data_scale * data - mc_subtract) * self.scale

    def draw_into(self, hist, expr, category, region,
                  cuts=None, weighted=True, systematics=True):

        MC_bkg = hist.Clone()
        MC_bkg.Reset()
        for mc_scale, mc in zip(self.mc_scales, self.mc):
            mc.draw_into(MC_bkg, expr, category, self.shape_region,
                         cuts=cuts, weighted=weighted,
                         systematics=systematics, scale=mc_scale)

        data_hist = hist.Clone()
        data_hist.Reset()
        self.data.draw_into(data_hist, expr,
                            category, self.shape_region,
                            cuts=cuts, weighted=weighted)

        log.info("QCD: Data(%.3f) - MC(%.3f)" % (
            data_hist.Integral(),
            MC_bkg.Integral()))

        hist += (data_hist * self.data_scale - MC_bkg) * self.scale

        if systematics and hasattr(MC_bkg, 'systematics'):
            if not hasattr(hist, 'systematics'):
                hist.systematics = {}
            for sys_term, sys_hist in MC_bkg.systematics.items():
                scale = self.scale
                if sys_term == ('QCDFIT_UP',):
                    scale = self.scale + self.scale_error
                elif sys_term == ('QCDFIT_DOWN',):
                    scale = self.scale - self.scale_error
                qcd_hist = (data_hist * self.data_scale - sys_hist) * scale
                if sys_term not in hist.systematics:
                    hist.systematics[sys_term] = qcd_hist
                else:
                    hist.systematics[sys_term] += qcd_hist

        hist.SetTitle(self.label)

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
                   systematics_components=None):

        do_systematics = self.systematics and systematics

        field_hist_MC_bkg = dict([(expr, hist.Clone())
            for expr, hist in field_hist.items()])

        for mc_scale, mc in zip(self.mc_scales, self.mc):
            mc.draw_array(field_hist_MC_bkg, category, self.shape_region,
                cuts=cuts,
                weighted=weighted,
                field_scale=field_scale,
                weight_hist=weight_hist,
                clf=clf,
                scores=scores,
                min_score=min_score,
                max_score=max_score,
                systematics=systematics,
                systematics_components=systematics_components,
                scale=mc_scale)

        field_hist_data = dict([(expr, hist.Clone())
            for expr, hist in field_hist.items()])

        self.data.draw_array(field_hist_data,
            category, self.shape_region,
            cuts=cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            clf=clf,
            scores=scores,
            min_score=min_score,
            max_score=max_score)

        for expr, h in field_hist.items():
            mc_h = field_hist_MC_bkg[expr]
            d_h = field_hist_data[expr]
            h += (d_h * self.data_scale - mc_h) * self.scale
            h.SetTitle(self.label)
            if not do_systematics:
                continue
            if hasattr(mc_h, 'systematics'):
                if not hasattr(h, 'systematics'):
                    h.systematics = {}
                for sys_term, sys_hist in mc_h.systematics.items():
                    scale = self.scale
                    if sys_term == ('QCDFIT_UP',):
                        scale = self.scale + self.scale_error
                    elif sys_term == ('QCDFIT_DOWN',):
                        scale = self.scale - self.scale_error
                    qcd_hist = (d_h * self.data_scale - sys_hist) * scale
                    qcd_hist.name = h.name + '_' + systematic_name(sys_term)
                    if sys_term not in h.systematics:
                        h.systematics[sys_term] = qcd_hist
                    else:
                        h.systematics[sys_term] += qcd_hist

    def scores(self, clf, category, region,
               cuts=None,
               systematics=True,
               systematics_components=None,
               **kwargs):

        # data
        data_scores, data_weights = self.data.scores(
            clf,
            category,
            region=self.shape_region,
            cuts=cuts,
            **kwargs)

        scores_dict = {}
        # subtract MC
        for mc_scale, mc in zip(self.mc_scales, self.mc):
            mc.scores(
                clf,
                category,
                region=self.shape_region,
                cuts=cuts,
                scores_dict=scores_dict,
                systematics=systematics,
                systematics_components=systematics_components,
                scale=mc_scale,
                **kwargs)

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
        for mc_scale, mc in zip(self.mc_scales, self.mc):
            _trees = mc.trees(
                category,
                region=self.shape_region,
                cuts=cuts,
                systematic=systematic,
                scale=mc_scale)
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
                return_idx=False,
                **kwargs):

        assert include_weight == True

        data_records = self.data.records(
            category=category,
            region=self.shape_region,
            fields=fields,
            cuts=cuts,
            include_weight=include_weight,
            systematic='NOMINAL',
            return_idx=return_idx,
            **kwargs)
        arrays = data_records

        for mc_scale, mc in zip(self.mc_scales, self.mc):
            _arrays = mc.records(
                category=category,
                region=self.shape_region,
                fields=fields,
                cuts=cuts,
                include_weight=include_weight,
                systematic=systematic,
                scale=mc_scale,
                return_idx=return_idx,
                **kwargs)
            # FIX: weight may not be present if include_weight=False
            for array in _arrays:
                if return_idx:
                    for partition, idx in array:
                        partition['weight'] *= -1
                else:
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
            if return_idx:
                for partition, idx in array:
                    partition['weight'] *= scale
            else:
                for partition in array:
                    partition['weight'] *= scale

        return arrays
