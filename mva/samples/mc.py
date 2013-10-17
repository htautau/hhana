from .sample import Sample, Signal, Background
from .db import DB, TEMPFILE, get_file
from . import log
from ..cachedtable import CachedTable
from ..lumi import LUMI
from ..systematics import SYSTEMATICS_BY_WEIGHT, iter_systematics, systematic_name

from higgstautau import samples as samples_db

from rootpy import asrootpy
from rootpy.plotting import Hist
from rootpy.tree import Cut

import numpy as np
from numpy.lib import recfunctions


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

    def __init__(self, year, db=DB, systematics=True, **kwargs):

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

        from .ztautau import Embedded_Ztautau

        for i, name in enumerate(self.samples):

            ds = self.db[name]
            treename = name.replace('.', '_')
            treename = treename.replace('-', '_')

            trees = {}
            tables = {}
            weighted_events = {}

            if isinstance(self, Embedded_Ztautau):
                events_bin = 1
            else:
                # use mc_weighted second bin
                events_bin = 2
            events_hist_suffix = '_cutflow'

            trees['NOMINAL'] = rfile.Get(treename)
            tables['NOMINAL'] =  CachedTable.hook(getattr(
                h5file.root, treename))

            weighted_events['NOMINAL'] = rfile.Get(
                treename + events_hist_suffix)[events_bin].value

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
                            sys_name + events_hist_suffix)[events_bin].value

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
                            sample_name + events_hist_suffix)[events_bin].value

            if hasattr(self, 'xsec_kfact_effic'):
                xs, kfact, effic = self.xsec_kfact_effic(i)
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

        from .ztautau import Ztautau

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
                   scale=1.,
                   bootstrap_data=False):

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

        from .ztautau import Ztautau

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

        from .ztautau import Ztautau

        if include_weight and fields is not None:
            if 'weight' not in fields:
                fields = list(fields) + ['weight']

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
                weights = np.empty(rec.shape[0], dtype='f8')
                weights.fill(weight)
                # merge the weight fields
                weights *= reduce(np.multiply,
                    [rec[br] for br in weight_branches])
                # drop other weight fields
                rec = recfunctions.rec_drop_fields(rec, weight_branches)
                # add the combined weight
                rec = recfunctions.rec_append_fields(rec,
                    names='weight',
                    data=weights,
                    dtypes='f8')
                if rec['weight'].shape[0] > 1 and rec['weight'].sum() == 0:
                    log.warning("{0}: weights sum to zero!".format(table.name))

            if fields is not None:
                try:
                    rec = rec[fields]
                except Exception as e:
                    print table
                    print rec.shape
                    print rec.dtype
                    print e
                    raise
            recs.append(rec)

        if return_idx:
            return zip(recs, idxs)
        return recs

    def events(self, category=None, region=None,
               cuts=None,
               systematic='NOMINAL',
               weighted=True,
               hist=None,
               scale=1.):

        if hist is None:
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

            weight = LUMI[self.year] * self.scale * xs * kfact * effic / events

            selection = Cut(' * '.join(map(str,
                self.get_weight_branches(systematic, weighted=weighted))))
            selection = Cut(str(weight * scale)) * selection * (
                self.cuts(category, region, systematic=systematic) & cuts)

            log.debug("requesing number of events from %s using cuts: %s"
                % (tree.GetName(), selection))

            tree.Draw('1', selection, hist=hist)
        return hist

