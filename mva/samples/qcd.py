# rootpy imports
from rootpy import asrootpy
from rootpy.plotting import Hist

# numpy imports
import numpy as np

# local imports
from .sample import Sample, Background
from . import log; log = log[__name__]
from ..systematics import systematic_name


class QCD(Sample, Background):
    # don't include MC systematics in workspace for QCD
    WORKSPACE_SYSTEMATICS = [] #MC.WORKSPACE_SYSTEMATICS
    NORM_BY_THEORY = False

    def systematics_components(self):
        return []

    def histfactory(self, sample, category, systematics=True):
        if self.workspace_norm is False:
            return
        if self.workspace_norm is not None:
            sample.AddNormFactor(
                'ATLAS_norm_HH_{0:d}_QCD'.format(self.year),
                self.workspace_norm,
                self.workspace_norm,
                self.workspace_norm,
                True) # const
        elif self.constrain_norm:
            # overallsys
            error = self.scale_error / self.scale
            sample.AddOverallSys(
                'ATLAS_norm_HH_{0:d}_QCD'.format(self.year),
                1. - error,
                1. + error)
        else:
            sample.AddNormFactor(
                'ATLAS_norm_HH_{0:d}_QCD'.format(self.year),
                1., 0., 50., False) # floating

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
                 shape_region='nOS',
                 decouple_shape=True,
                 workspace_norm=None,
                 constrain_norm=False,
                 cuts=None,
                 name='Fakes',
                 label='Fakes',
                 **kwargs):
        QCD.sample_compatibility(data, mc)
        super(QCD, self).__init__(
            year=data.year,
            scale=scale,
            name=name,
            label=label,
            **kwargs)
        self.data = data
        self.mc = mc
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
        self.decouple_shape = decouple_shape
        self.workspace_norm = workspace_norm
        self.constrain_norm = constrain_norm
        self.systematics = mc[0].systematics

    def events(self, category=None, region=None, cuts=None,
               systematic='NOMINAL'):
        data = Hist(1, -100, 100)
        mc_subtract = data.Clone()
        self.data.events(category, self.shape_region, cuts=cuts, hist=data)
        for mc_scale, mc in zip(self.mc_scales, self.mc):
            mc.events(
                category, self.shape_region,
                cuts=cuts,
                systematic=systematic,
                hist=mc_subtract,
                scale=mc_scale)
        log.info("QCD: Data(%.3f) - MC(%.3f)" % (
            (self.data_scale * data)[1].value, mc_subtract[1].value))
        log.info("MC subtraction: %.1f%%" % (
            100. * (mc_subtract[1].value) / ((self.data_scale * data)[1].value)))
        return (data * self.data_scale - mc_subtract) * self.scale

    def draw_into(self, hist, expr, category, region,
                  cuts=None, weighted=True, systematics=True):
        # TODO: handle QCD shape systematic
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
                   field_weight_hist=None,
                   clf=None,
                   scores=None,
                   min_score=None,
                   max_score=None,
                   systematics=True,
                   systematics_components=None,
                   bootstrap_data=False):
        # TODO: support for field_weight_hist
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

        if do_systematics and systematics_components is None:
            # get shape systematic
            field_shape_sys = self.get_shape_systematic_array(
                # use the nominal hist here
                field_hist,
                category, region,
                cuts=cuts,
                clf=clf,
                min_score=min_score,
                max_score=max_score,
                field_scale=field_scale,
                weight_hist=weight_hist)
        else:
            field_shape_sys = None

        for expr, h in field_hist.items():
            mc_h = field_hist_MC_bkg[expr]
            d_h = field_hist_data[expr]

            if not do_systematics or not hasattr(mc_h, 'systematics'):
                continue
            if not hasattr(h, 'systematics'):
                h.systematics = {}

            if field_shape_sys is not None:
                # add shape systematics
                high, low = field_shape_sys[expr]
                h.systematics[('QCDSHAPE_DOWN',)] = low
                h.systematics[('QCDSHAPE_UP',)] = high

            for sys_term, sys_hist in mc_h.systematics.items():
                if sys_term in (('QCDSHAPE_UP',), ('QCDSHAPE_DOWN',)):
                    continue
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
                category=None,
                region=None,
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

        if return_idx:
            arrays = [(d.copy(), idx) for d, idx in data_records]
        else:
            arrays = [d.copy() for d in data_records]

        for mc_scale, mc in zip(self.mc_scales, self.mc):
            _arrays = []
            _arrs = mc.records(
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
            if return_idx:
                for partition, idx in _arrs:
                    partition = partition.copy()
                    partition['weight'] *= -1
                    _arrays.append((partition, idx))
            else:
                for partition in _arrs:
                    partition = partition.copy()
                    partition['weight'] *= -1
                    _arrays.append(partition)
            arrays.extend(_arrays)

        scale = self.scale
        if systematic == ('QCDFIT_UP',):
            scale += self.scale_error
        elif systematic == ('QCDFIT_DOWN',):
            scale -= self.scale_error

        # FIX: weight may not be present if include_weight=False
        if return_idx:
            for partition, idx in arrays:
                partition['weight'] *= scale
        else:
            for partition in arrays:
                partition['weight'] *= scale

        return arrays

    def get_shape_systematic(self, nominal_hist,
                             expr_or_clf,
                             category, region,
                             cuts=None,
                             clf=None,
                             min_score=None,
                             max_score=None,
                             suffix=None,
                             field_scale=None,
                             weight_hist=None,
                             weighted=True):
        log.info("creating QCD shape systematic")
        # HACK
        # use preselection as reference in which all models should have the same
        # expected number of QCD events
        # get number of events at preselection for nominal model
        from ..categories import Category_Preselection
        nominal_events = self.events(Category_Preselection, None)[1].value

        hist_template = nominal_hist.Clone()
        hist_template.Reset()

        curr_model = self.shape_region
        # add QCD shape systematic
        if curr_model == 'SS':
            # OSFF x (SS / SSFF) model in the track-fit category
            models = []
            events = []
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
                    weight_hist=weight_hist,
                    weighted=weighted))
                events.append(self.events(Category_Preselection, None)[1].value)

            OSFF, SSFF = models
            OSFF_events, SSFF_events = events
            shape_sys = OSFF
            nominal_hist_norm = nominal_hist / nominal_hist.Integral()
            SSFF_norm = SSFF / SSFF.Integral()
            shape_sys *= nominal_hist_norm / SSFF_norm
            # this is approximate
            # normalize shape_sys such that it would have the same number of
            # events as the nominal at preselection
            shape_sys *= nominal_events / float(OSFF_events)

        elif curr_model == 'nOS':
            # SS model elsewhere
            self.shape_region = 'SS'
            log.info("getting QCD shape for SS")
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
                suffix=(suffix or '') + '_SS',
                field_scale=field_scale,
                weight_hist=weight_hist,
                weighted=weighted)
            SS_events = self.events(Category_Preselection, None)[1].value
            # normalize shape_sys such that it would have the same number of
            # events as the nominal at preselection
            shape_sys *= nominal_events / float(SS_events)

        else:
            raise ValueError(
                "no QCD shape systematic defined for nominal {0}".format(
                    curr_model))

        # restore previous shape model
        self.shape_region = curr_model

        # reflect shape about the nominal to get high and low variations
        shape_sys_reflect = nominal_hist + (nominal_hist - shape_sys)
        shape_sys_reflect.name = shape_sys.name + '_reflected'

        return shape_sys, shape_sys_reflect

    def get_shape_systematic_array(
            self, field_nominal_hist,
            category, region,
            cuts=None,
            clf=None,
            min_score=None,
            max_score=None,
            suffix=None,
            field_scale=None,
            weight_hist=None,
            weighted=True):

        log.info("creating QCD shape systematic")

        # HACK
        # use preselection as reference in which all models should have the same
        # expected number of QCD events
        # get number of events at preselection for nominal model
        from ..categories import Category_Preselection
        nominal_events = self.events(Category_Preselection, None)[1].value

        field_hist_template = {}
        for field, hist in field_nominal_hist.items():
            new_hist = hist.Clone()
            new_hist.Reset()
            field_hist_template[field] = new_hist

        curr_model = self.shape_region
        # add QCD shape systematic
        if curr_model == 'SS':
            # OSFF x (SS / SSFF) model in the track-fit category
            models = []
            for model in ('OSFF', 'SSFF'):
                log.info("getting QCD shape for {0}".format(model))
                self.shape_region = model
                models.append(self.get_hist_array(
                    field_hist_template,
                    category, region,
                    cuts=cuts,
                    clf=clf,
                    scores=None,
                    min_score=min_score,
                    max_score=max_score,
                    systematics=False,
                    suffix=(suffix or '') + '_%s' % model,
                    field_scale=field_scale,
                    weight_hist=weight_hist,
                    weighted=weighted))
                if model == 'OSFF':
                    norm_events = self.events(Category_Preselection, None)[1].value

            OSFF, SSFF = models
            field_shape_sys = {}
            for field, nominal_hist in field_nominal_hist.items():
                shape_sys = OSFF[field]
                shape_sys *= (nominal_hist.normalize(copy=True) /
                    SSFF[field].normalize(copy=True))
                field_shape_sys[field] = shape_sys

        elif curr_model == 'nOS':
            # SS model elsewhere
            self.shape_region = 'SS'
            log.info("getting QCD shape for SS")
            field_shape_sys = self.get_hist_array(
                field_hist_template,
                category, region,
                cuts=cuts,
                clf=clf,
                scores=None,
                min_score=min_score,
                max_score=max_score,
                systematics=False,
                suffix=(suffix or '') + '_SS',
                field_scale=field_scale,
                weight_hist=weight_hist,
                weighted=weighted)
            norm_events = self.events(Category_Preselection, None)[1].value

        else:
            raise ValueError(
                "no QCD shape systematic defined for nominal {0}".format(
                    curr_model))

        # restore previous shape model
        self.shape_region = curr_model

        field_shape_sys_reflect = {}

        for field, shape_sys in field_shape_sys.items():
            nominal_hist = field_nominal_hist[field]

            # this may be approximate
            # normalize shape_sys such that it would have the same number of
            # events as the nominal at preselection
            shape_sys *= nominal_events / float(norm_events)

            # reflect shape about the nominal to get high and low variations
            shape_sys_reflect = nominal_hist + (nominal_hist - shape_sys)
            shape_sys_reflect.name = shape_sys.name + '_reflected'
            field_shape_sys_reflect[field] = (shape_sys, shape_sys_reflect)

        return field_shape_sys_reflect
