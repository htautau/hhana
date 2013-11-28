# std lib imports
import os
import sys
from operator import add, itemgetter

# numpy imports
import numpy as np
from numpy.lib import recfunctions

# rootpy imports
import ROOT
from rootpy.plotting import Hist, Hist2D, Canvas, HistStack
from rootpy.plotting.hist import _HistBase
from rootpy.tree import Tree, Cut
from rootpy.stats import histfactory

# root_numpy imports
from root_numpy import rec2array, stack

# local imports
from . import log
from .. import variables
from .. import DEFAULT_STUDENT, ETC_DIR
from ..utils import print_hist, ravel
from ..classify import histogram_scores, Classifier
from ..regions import REGIONS
from ..systematics import WEIGHT_SYSTEMATICS, get_systematics
from ..lumi import get_lumi_uncert


def get_workspace_np_name(sample, syst, year):

    npname = 'ATLAS_{0}_{1:d}'.format(syst, year)

    # https://twiki.cern.ch/twiki/bin/viewauth/AtlasProtected/HiggsPropertiesNuisanceParameterNames
    npname = npname.replace('JES_Detector_2012', 'JES_2012_Detector1')
    npname = npname.replace('JES_EtaMethod_2012', 'JES_2012_Eta_StatMethod')
    npname = npname.replace('JES_EtaModelling_2012', 'JES_Eta_Modelling')
    npname = npname.replace('JES_FlavComp_TAU_G_2012', 'JES_FlavComp_TAU_G')
    npname = npname.replace('JES_FlavComp_TAU_Q_2012', 'JES_FlavComp_TAU_Q')
    npname = npname.replace('JES_FlavResp_2012', 'JES_FlavResp')
    npname = npname.replace('JES_Modelling_2012', 'JES_2012_Modelling1')
    npname = npname.replace('JES_PURho_TAU_GG_2012', 'JES_2012_PileRho_TAU_GG')
    npname = npname.replace('JES_PURho_TAU_QG_2012', 'JES_2012_PileRho_TAU_QG')
    npname = npname.replace('JES_PURho_TAU_QQ_2012', 'JES_2012_PileRho_TAU_QQ')
    npname = npname.replace('FAKERATE_2012', 'TAU_JFAKE_2012')
    npname = npname.replace('TAUID_2012', 'TAU_ID_2012')
    npname = npname.replace('ISOL_2012', 'ANA_EMB_ISOL')
    npname = npname.replace('MFS_2012', 'ANA_EMB_MFS')
    npname = npname.replace('MET_RESOSOFTTERMS_2012', 'MET_RESOSOFT')
    npname = npname.replace('MET_SCALESOFTTERMS_2012', 'MET_SCALESOFT')

    from .ztautau import Embedded_Ztautau

    if isinstance(sample, Embedded_Ztautau):
        npname = npname.replace('TRIGGER', 'TRIGGER_EMB_HH')
    else:
        npname = npname.replace('TRIGGER', 'TRIGGER_HH')

    return npname


class Sample(object):

    WORKSPACE_SYSTEMATICS = []

    def __init__(self, year, scale=1., cuts=None,
                 student=DEFAULT_STUDENT,
                 mpl=False,
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

        self.mpl = mpl
        self.student = student
        self.hist_decor = hist_decor
        if 'fillstyle' not in hist_decor:
            self.hist_decor['fillstyle'] = 'solid'

    @property
    def label(self):

        if self.mpl:
            return self._label
        return self._label_root

    def get_field_hist(self, vars, category, templates=None):

        field_hist = {}
        for field, var_info in vars.items():

            if templates is not None and field in templates:
                field_hist[field] = templates[field].Clone(
                    title=self.label, **self.hist_decor)
                continue
            elif isinstance(var_info, _HistBase):
                field_hist[field] = var_info.Clone(
                    title=self.label, **self.hist_decor)
                continue

            _range = var_info['range']
            if isinstance(_range, dict):
                _range = _range.get(category.name.upper(), _range[None])
            if len(_range) == 3:
                bins, min, max = _range
            else:
                min, max = _range
                bins = var_info['bins']

            hist = Hist(bins, min, max,
                title=self.label,
                type='D',
                **self.hist_decor)
            field_hist[field] = hist
        return field_hist

    def get_hist_array(self,
            field_hist_template,
            category, region,
            cuts=None,
            clf=None,
            scores=None,
            min_score=None,
            max_score=None,
            systematics=True,
            systematics_components=None,
            suffix=None,
            field_scale=None,
            weight_hist=None,
            weighted=True,
            bootstrap_data=False):

        from .data import Data

        do_systematics = (not isinstance(self, Data)
                          and self.systematics
                          and systematics)

        if min_score is None:
            min_score = getattr(category, 'workspace_min_clf', None)
        if max_score is None:
            max_score = getattr(category, 'workspace_max_clf', None)

        histname = 'hh_category_{0}_{1}'.format(category.name, self.name)
        if suffix is not None:
            histname += suffix

        field_hist = {}
        for field, hist in field_hist_template.items():
            new_hist = hist.Clone(name=histname + '_{0}'.format(field))
            new_hist.Reset()
            field_hist[field] = new_hist

        self.draw_array(field_hist, category, region,
            cuts=cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            clf=clf,
            min_score=min_score,
            max_score=max_score,
            systematics=do_systematics,
            systematics_components=systematics_components,
            bootstrap_data=bootstrap_data)

        return field_hist

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
            systematics_components=None,
            suffix=None,
            field_scale=None,
            weight_hist=None,
            weighted=True):

        from .data import Data

        do_systematics = (not isinstance(self, Data)
                          and self.systematics
                          and systematics)

        if min_score is None:
            min_score = getattr(category, 'workspace_min_clf', None)
        if max_score is None:
            max_score = getattr(category, 'workspace_max_clf', None)

        histname = 'hh_category_{0}_{1}'.format(category.name, self.name)
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
                weighted=weighted,
                field_scale=field_scale,
                weight_hist=weight_hist,
                clf=clf,
                min_score=min_score,
                max_score=max_score,
                systematics=do_systematics,
                systematics_components=systematics_components)

        else:
            # histogram classifier output
            if scores is None:
                scores = self.scores(
                    expr_or_clf, category, region, cuts,
                    systematics=do_systematics,
                    systematics_components=systematics_components)
            histogram_scores(
                hist, scores,
                min_score=min_score,
                max_score=max_score,
                inplace=True)

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
            weight_hist=None,
            weighted=True,
            no_signal_fixes=False):

        from .data import Data
        from .qcd import QCD
        from .others import Others

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
            systematics_components=self.WORKSPACE_SYSTEMATICS,
            suffix=suffix,
            field_scale=field_scale,
            weight_hist=weight_hist,
            weighted=weighted)

        # copy of unaltered nominal hist required by QCD shape
        nominal_hist = hist.Clone()

        # convert to 1D if 2D (also handles systematics if present)
        hist = ravel(hist)

        #print_hist(hist)

        # set the nominal histogram
        sample.hist = hist

        do_systematics = (not isinstance(self, Data)
                          and self.systematics
                          and systematics)

        # add systematics samples
        if do_systematics:

            SYSTEMATICS = get_systematics(self.year)

            for sys_component in self.WORKSPACE_SYSTEMATICS:

                terms = SYSTEMATICS[sys_component]
                if len(terms) == 1:
                    up_term = terms[0]
                    hist_up = hist.systematics[up_term]
                    # use nominal hist for "down" side
                    hist_down = hist

                else:
                    up_term, down_term = terms
                    hist_up = hist.systematics[up_term]
                    hist_down = hist.systematics[down_term]

                if sys_component == 'JES_FlavComp':
                    if ((isinstance(self, Signal) and self.mode == 'gg') or
                         isinstance(self, Others)):
                        sys_component += '_TAU_G'
                    else:
                        sys_component += '_TAU_Q'

                elif sys_component == 'JES_PURho':
                    if isinstance(self, Others):
                        sys_component += '_TAU_QG'
                    elif isinstance(self, Signal):
                        if self.mode == 'gg':
                            sys_component += '_TAU_GG'
                        else:
                            sys_component += '_TAU_QQ'

                npname = get_workspace_np_name(self, sys_component, self.year)

                histsys = histfactory.HistoSys(
                    npname,
                    low=hist_down,
                    high=hist_up)

                sample.AddHistoSys(histsys)

            if isinstance(self, QCD):
                high, low = self.get_shape_systematic(
                     nominal_hist,
                     expr_or_clf,
                     category, region,
                     cuts=cuts,
                     clf=clf,
                     min_score=min_score,
                     max_score=max_score,
                     suffix=suffix,
                     field_scale=field_scale,
                     weight_hist=weight_hist,
                     weighted=weighted)

                low = ravel(low)
                high = ravel(high)

                #log.info("QCD low shape")
                #print_hist(low)
                #log.info("QCD high shape")
                #print_hist(high)

                npname = 'ATLAS_ANA_HH_{0:d}_QCD'.format(self.year)
                if category.analysis_control and self.decouple_shape:
                    npname += '_CR'
                histsys = histfactory.HistoSys(npname, low=low, high=high)
                sample.AddHistoSys(histsys)

        if isinstance(self, Signal):
            sample.AddNormFactor('SigXsecOverSM', 0., 0., 200., False)

        elif isinstance(self, Background):
            # only activate stat error on background samples
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

        if hasattr(self, 'histfactory') and not (
                isinstance(self, Signal) and no_signal_fixes):
            # perform sample-specific items
            self.histfactory(sample, category, systematics=do_systematics)

        return sample

    def get_histfactory_sample_array(self,
            field_hist_template,
            category, region,
            cuts=None,
            clf=None,
            scores=None,
            min_score=None,
            max_score=None,
            systematics=True,
            suffix=None,
            field_scale=None,
            weight_hist=None,
            weighted=True,
            no_signal_fixes=False,
            bootstrap_data=False):

        from .data import Data
        from .qcd import QCD
        from .others import Others

        log.info("creating histfactory samples for {0}".format(self.name))

        field_hist = self.get_hist_array(
            field_hist_template,
            category, region,
            cuts=cuts,
            clf=clf,
            scores=scores,
            min_score=min_score,
            max_score=max_score,
            systematics=systematics,
            systematics_components=self.WORKSPACE_SYSTEMATICS,
            suffix=suffix,
            field_scale=field_scale,
            weight_hist=weight_hist,
            weighted=weighted,
            bootstrap_data=bootstrap_data)

        do_systematics = (not isinstance(self, Data)
                          and self.systematics
                          and systematics)

        if isinstance(self, QCD) and do_systematics:
            qcd_shapes = self.get_shape_systematic_array(
                field_hist,
                category, region,
                cuts=cuts,
                clf=clf,
                min_score=min_score,
                max_score=max_score,
                suffix=suffix,
                field_scale=field_scale,
                weight_hist=weight_hist,
                weighted=weighted)

        field_samples = {}

        for field, hist in field_hist.items():

            if isinstance(self, Data):
                sample = histfactory.Data(self.name)
            else:
                sample = histfactory.Sample(self.name)

            # copy of unaltered nominal hist required by QCD shape
            nominal_hist = hist.Clone()

            # convert to 1D if 2D (also handles systematics if present)
            hist = ravel(hist)

            #print_hist(hist)

            # set the nominal histogram
            sample.hist = hist

            # add systematics samples
            if do_systematics:

                SYSTEMATICS = get_systematics(self.year)

                for sys_component in self.WORKSPACE_SYSTEMATICS:

                    terms = SYSTEMATICS[sys_component]
                    if len(terms) == 1:
                        up_term = terms[0]
                        hist_up = hist.systematics[up_term]
                        # use nominal hist for "down" side
                        hist_down = hist

                    else:
                        up_term, down_term = terms
                        hist_up = hist.systematics[up_term]
                        hist_down = hist.systematics[down_term]

                    if sys_component == 'JES_FlavComp':
                        if ((isinstance(self, Signal) and self.mode == 'gg') or
                             isinstance(self, Others)):
                            sys_component += '_TAU_G'
                        else:
                            sys_component += '_TAU_Q'

                    elif sys_component == 'JES_PURho':
                        if isinstance(self, Others):
                            sys_component += '_TAU_QG'
                        elif isinstance(self, Signal):
                            if self.mode == 'gg':
                                sys_component += '_TAU_GG'
                            else:
                                sys_component += '_TAU_QQ'

                    npname = get_workspace_np_name(self, sys_component, self.year)

                    histsys = histfactory.HistoSys(
                        npname,
                        low=hist_down,
                        high=hist_up)

                    sample.AddHistoSys(histsys)

                if isinstance(self, QCD):

                    high, low = qcd_shapes[field]

                    low = ravel(low)
                    high = ravel(high)

                    #log.info("QCD low shape")
                    #print_hist(low)
                    #log.info("QCD high shape")
                    #print_hist(high)

                    npname = 'ATLAS_ANA_HH_{0:d}_QCD'.format(self.year)
                    if category.analysis_control and self.decouple_shape:
                        npname += '_CR'
                    histsys = histfactory.HistoSys(npname, low=low, high=high)
                    sample.AddHistoSys(histsys)

            if isinstance(self, Signal):
                sample.AddNormFactor('SigXsecOverSM', 0., 0., 200., False)

            elif isinstance(self, Background):
                # only activate stat error on background samples
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

            # HACK: disable calling this on signal for now since while plotting
            # we only want to show the combined signal but in the histfactory
            # method we require only a single mode
            if hasattr(self, 'histfactory') and not (
                    isinstance(self, Signal) and no_signal_fixes):
                # perform sample-specific items
                self.histfactory(sample, category, systematics=do_systematics)

            field_samples[field] = sample

        return field_samples

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
              clf=None,
              clf_name='classifier',
              include_weight=True,
              systematic='NOMINAL'):

        recs = self.records(
            category,
            region,
            fields=fields,
            include_weight=include_weight,
            cuts=cuts,
            systematic=systematic)

        if include_weight and fields is not None:
            if 'weight' not in fields:
                fields = list(fields) + ['weight']
        rec = stack(recs, fields=fields)

        if clf is not None:
            scores, _ = clf.classify(
                self, category, region,
                cuts=cuts, systematic=systematic)
            rec = recfunctions.rec_append_fields(rec,
                names=clf_name,
                data=scores,
                dtypes='f4')

        return rec

    def array(self,
              category,
              region,
              fields=None,
              cuts=None,
              clf=None,
              clf_name='classifer',
              include_weight=True,
              systematic='NOMINAL'):

        return rec2array(self.merged_records(
            category,
            region,
            fields=fields,
            cuts=cuts,
            clf=clf,
            clf_name=clf_name,
            include_weight=include_weight,
            systematic=systematic))

    @classmethod
    def get_sys_term_variation(cls, systematic):

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
            for term, variations in WEIGHT_SYSTEMATICS.items():
                if term == systerm:
                    weight_branches += variations[variation]
                else:
                    weight_branches += variations['NOMINAL']
        else:
            weight_branches = []
        return weight_branches

    def iter_weight_branches(self):

        for type, variations in WEIGHT_SYSTEMATICS.items():
            for variation in variations:
                if variation == 'NOMINAL':
                    continue
                term = ('%s_%s' % (type, variation),)
                yield self.get_weight_branches(term), term

    def cuts(self, category, region, systematic='NOMINAL', **kwargs):

        cuts = Cut(self._cuts)
        if category is not None:
            cuts &= category.get_cuts(self.year, **kwargs)
        if region is not None:
            cuts &= REGIONS[region]
        return cuts

    def draw(self, expr, category, region, bins, min, max,
             cuts=None, weighted=True, systematics=True):

        hist = Hist(bins, min, max,
            title=self.label,
            type='D',
            **self.hist_decor)
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
            title=self.label,
            type='D',
            **self.hist_decor)
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
                          clf=None,
                          min_score=None,
                          max_score=None,
                          systematic='NOMINAL',
                          bootstrap_data=False):

        from .data import Data, DataInfo

        all_fields = []
        classifiers = []
        for f in field_hist.iterkeys():
            if isinstance(f, basestring):
                all_fields.append(f)
            elif isinstance(f, Classifier):
                classifiers.append(f)
            else:
                all_fields.extend(list(f))
        if len(classifiers) > 1:
            raise RuntimeError(
                "more than one classifier in fields is not supported")
        elif len(classifiers) == 1:
            classifier = classifiers[0]
        else:
            classifier = None

        if isinstance(self, Data) and bootstrap_data:
            log.info("using bootstrapped data")
            analysis = bootstrap_data
            recs = []
            scores = []
            for s in analysis.backgrounds:
                rec = s.merged_records(category, region,
                    fields=all_fields, cuts=cuts,
                    include_weight=True,
                    clf=clf,
                    systematic=systematic)
                recs.append(rec)
            b_rec = stack(recs, fields=all_fields + ['classifier', 'weight'])
            s_rec = analysis.higgs_125.merged_records(category, region,
                fields=all_fields, cuts=cuts,
                include_weight=True,
                clf=clf,
                systematic=systematic)

            # handle negative weights separately
            b_neg = b_rec[b_rec['weight'] < 0]
            b_pos = b_rec[b_rec['weight'] >= 0]

            def bootstrap(rec):
                prob = np.abs(rec['weight'])
                prob = prob / prob.sum()
                # random sample without replacement
                log.warning(str(int(round(abs(rec['weight'].sum())))))
                sample_idx = np.random.choice(
                    rec.shape[0], size=int(round(abs(rec['weight'].sum()))),
                    replace=False, p=prob)
                return rec[sample_idx]

            rec = stack([
                bootstrap(b_neg),
                bootstrap(b_pos),
                bootstrap(s_rec)],
                fields=all_fields + ['classifier', 'weight'])

            rec['weight'][:] = 1.
            scores = rec['classifier']
        else:
            # TODO: only get unblinded vars
            rec = self.merged_records(category, region,
                fields=all_fields, cuts=cuts,
                include_weight=True,
                clf=classifier,
                systematic=systematic)

        if isinstance(scores, tuple):
            # sanity
            #assert (scores[1] == rec['weight']).all()
            # ignore the score weights since they should be the same as the rec
            # weights
            scores = scores[0]

        if weight_hist is not None and scores is not None:
            log.warning("applying a weight histogram")
            edges = np.array(list(weight_hist.xedges()))
            # handle strange cases
            edges[0] -= 1E10
            edges[-1] += 1E10
            weights = np.array(list(weight_hist.y())).take(
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
            if isinstance(fields, Classifier):
                fields = ['classifier']
            # fields can be a single field or list of fields
            elif not isinstance(fields, (list, tuple)):
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
            arr = rec2array(arr, fields=fields)
            # HACK HACK HACK
            _weights = weights
            if fields == ['dEta_jets']:
                log.warning("HACK HACK")
                nonzero = arr > 0
                arr = arr[nonzero]
                _weights = weights[nonzero]
            # include the scores if the histogram dimensionality allows
            if scores is not None and hist.GetDimension() == len(fields) + 1:
                arr = np.c_[arr, scores]
            elif hist.GetDimension() != len(fields):
                raise TypeError(
                    'histogram dimensionality does not match '
                    'number of fields: %s' % (', '.join(fields)))
            hist.fill_array(arr, weights=_weights)
            if isinstance(self, Data):
                if hasattr(hist, 'datainfo'):
                    hist.datainfo += self.info
                else:
                    hist.datainfo = DataInfo(self.info.lumi, self.info.energies)


class Signal(object):
    # mixin
    pass


class Background(object):
    # mixin
    pass

