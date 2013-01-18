import os
from array import array

import ROOT
ROOT.gROOT.SetBatch(True)
from ROOT import TF1, TF2, TLatex

from rootpy.tree import Cut
from rootpy.plotting import Hist, Hist2D, HistStack, Legend, Canvas

import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from logger import log; log = log[__name__]
from utils import set_colours, draw
import categories
import bkg_scales_cache
from config import plots_dir
import samples


class FitError(Exception):
    pass

def draw_fit_1d(
        expr, bins,
        min, max,
        model,
        data,
        category,
        region,
        name,
        output_name,
        output_formats=('png', 'eps', 'pdf'),
        root=False,
        systematics=None,
        cuts=None,
        after=False):

    PLOTS_DIR = plots_dir(__file__)

    model_hists = []
    for sample in model:
        hist = sample.draw(expr, category, region,
                bins, min, max, cuts)
        model_hists.append(hist)

    data_hist = data.draw(expr, category, region,
            bins, min, max, cuts)

    if after:
        output_name += '_after'

    draw(model=model_hists,
        data=data_hist,
        name=name,
        category_name=category,
        category=category,
        show_ratio=True,
        systematics=systematics,
        root=root,
        dir=PLOTS_DIR,
        output_formats=output_formats,
        output_name=output_name)

def draw_fit(
        expr, bins,
        xmin, xmax,
        ymin, ymax,
        model,
        data,
        category,
        region,
        name,
        output_name,
        output_formats=('png', 'eps', 'pdf'),
        root=False,
        systematics=None,
        cuts=None,
        after=False):

    PLOTS_DIR = plots_dir(__file__)

    model_hists = []
    for sample in model:
        hist2d = sample.draw2d(expr, category, region,
                bins, xmin, xmax, bins, ymin, ymax, cuts)
        hist = hist2d.ravel()
        if hasattr(hist2d, 'systematics'):
            hist.systematics = {}
            for term, _hist in hist2d.systematics.items():
                hist.systematics[term] = _hist.ravel()
        model_hists.append(hist)

    data_hist2d = data.draw2d(expr, category, region,
            bins, xmin, xmax, bins, ymin, ymax, cuts)
    data_hist = data_hist2d.ravel()
    if hasattr(data_hist2d, 'systematics'):
        data_hist.systematics = {}
        for term, hist in data_hist2d.systematics.items():
            data_hist.systematics[term] = hist.ravel()

    if after:
        output_name += '_after'

    draw(model=model_hists,
        data=data_hist,
        name=name,
        category_name=category,
        category=category,
        show_ratio=True,
        systematics=systematics,
        root=root,
        dir=PLOTS_DIR,
        output_formats=output_formats,
        output_name=output_name)


def qcd_ztautau_norm(
        year,
        ztautau,
        others,
        qcd,
        data,
        category,
        target_region,
        mass_regions,
        cuts=None,
        bins=10,
        draw=False,
        use_cache=True,
        param='TRACK',
        systematics=None,
        root=False):

    is_embedded = isinstance(ztautau, samples.Embedded_Ztautau)
    param = param.upper()

    if use_cache and bkg_scales_cache.has_category(
            year, category, is_embedded, param):
        qcd_scale, qcd_scale_error, ztautau_scale, ztautau_scale_error = \
                 bkg_scales_cache.get_scales(year, category, is_embedded, param)
        qcd.scale = qcd_scale
        qcd.scale_error = qcd_scale_error
        ztautau.scale = ztautau_scale
        ztautau.scale_error = ztautau_scale_error
        return

    assert ROOT.TH1.GetDefaultSumw2() == True

    qcd_shape_region = qcd.shape_region

    log.info("fitting scale factors for embedding: %s" % str(is_embedded))
    log.info("fitting scale factors for %s category" % category)

    if param == 'BDT':
        xmin, xmax = .6, 1
        ymin, ymax = .55, 1
        expr = 'tau1_BDTJetScore:tau2_BDTJetScore'
        xlabel = '#tau_{1} BDT Score'
        ylabel = '#tau_{2} BDT Score'
        name = 'BDT Score Grid'
        ndim = 2
    elif param == 'TRACK':
        xmin, xmax = .5, 6.5
        ymin, ymax = .5, 6.5
        bins = int(ymax - ymin) # ignore bins args above
        expr = 'tau1_numTrack_recounted:tau2_numTrack_recounted'
        xlabel = '#tau_{1} Number of Tracks'
        ylabel = '#tau_{2} Number of Tracks'
        name = 'Number of Tracks Grid'
        ndim = 2
    elif param == 'TRACK1D':
        min, max = .5, 6.5
        bins = int(max - min) # ignore bins args above
        expr = 'tau1_ntrack_full'
        xlabel = '#tau_{1} Number of Tracks'
        name = 'Number of Tracks Grid'
        ndim = 1
    else:
        raise ValueError('No fit defined for %s parameters.' % param)

    output_name = "%dd_%s_fit_%s" % (ndim, param, category)
    if is_embedded:
        output_name += '_embedding'

    log.info("performing %d-dimensional fit using %s" % (ndim, expr))
    log.info("using %d bins on each axis" % bins)

    assert(ndim in (1, 2))
    control = mass_regions.control_region
    control &= cuts

    log.info("fitting scale factors in control region: %s" % control)

    if ndim == 1:
        hist = Hist(bins, min, max, name='fit_%s' % category)
    else:
        hist = Hist2D(bins, xmin, xmax, bins, ymin, ymax, name='fit_%s' % category)

    ztautau_hist = hist.Clone(title=ztautau.label)
    ztautau_hist_control = hist.Clone(title=ztautau.label)

    bkg_hist = hist.Clone(title=others.label)
    bkg_hist_control = hist.Clone(title=others.label)

    data_hist = hist.Clone(title=data.label)
    data_hist_control = hist.Clone(title=data.label)

    ztautau.draw_into(
            ztautau_hist,
            expr,
            category, target_region,
            cuts=control)

    ztautau.draw_into(
            ztautau_hist_control,
            expr,
            category, qcd_shape_region,
            cuts=control)

    others.draw_into(
            bkg_hist,
            expr,
            category, target_region,
            cuts=control)

    others.draw_into(
            bkg_hist_control,
            expr,
            category, qcd_shape_region,
            cuts=control)

    data.draw_into(
            data_hist,
            expr,
            category, target_region,
            cuts=control)

    data.draw_into(
            data_hist_control,
            expr,
            category, qcd_shape_region,
            cuts=control)

    # initialize Ztautau to OS data - SS data

    ztautau_init_factor = ((data_hist.Integral() - data_hist_control.Integral())
            / ztautau_hist.Integral())
    log.debug(ztautau_init_factor)
    ztautau_hist *= ztautau_init_factor
    ztautau_hist_control *= ztautau_init_factor
    ztautau.scale = ztautau_init_factor

    log.debug(ztautau_hist.Integral())
    log.debug(data_hist.Integral())

    if draw:
        if ndim == 1:
            draw_fit_1d(
                    expr, bins,
                    min, max,
                    model=[
                        qcd,
                        others,
                        ztautau],
                    data=data,
                    category=category,
                    region=target_region,
                    output_name=output_name,
                    name=name,
                    after=False,
                    systematics=systematics,
                    cuts=control,
                    root=root)

        else:
            draw_fit(
                    expr, bins,
                    xmin, xmax,
                    ymin, ymax,
                    model=[
                        qcd,
                        others,
                        ztautau],
                    data=data,
                    category=category,
                    region=target_region,
                    output_name=output_name,
                    name=name,
                    after=False,
                    systematics=systematics,
                    cuts=control,
                    root=root)

    class Model(object):

        def __init__(self, ndim=1):

            self.ndim=ndim
            if ndim == 1:
                self.func = TF1('model_%s' % category, self, 0, 10, 2)
            else:
                self.func = TF2('model_%s' % category, self, 0, 10, 0, 10, 2)

            self.func.SetParName(0, 'QCD_scale')
            self.func.SetParName(1, 'Ztautau_scale')
            self.func.SetParameter(0, 1.)
            self.func.SetParameter(1, 1.)

        def __call__(self, args, p):

            model = ( (data_hist_control
                       - ztautau_hist_control * p[1]
                       - bkg_hist_control) * p[0]
                    + ztautau_hist * p[1] + bkg_hist)
            bin = model.FindBin(*(list(args)[:self.ndim]))
            return model.GetBinContent(bin)

    model = Model(ndim=ndim)
    model_func = model.func
    model_func.SetLineWidth(0)
    fit_result = data_hist.Fit(model_func, 'WLMN')

    ztautau_hist /= ztautau_init_factor
    ztautau_hist_control /= ztautau_init_factor

    qcd_scale = model_func.GetParameter('QCD_scale')
    qcd_scale_error = model_func.GetParError(0)
    ztautau_scale = model_func.GetParameter('Ztautau_scale')
    ztautau_scale_error = model_func.GetParError(1)

    # scale by ztautau_init
    ztautau_scale *= ztautau_init_factor
    ztautau_scale_error *= ztautau_init_factor

    #qcd_scale *= factor
    #ztautau_scale *= factor

    qcd.scale = qcd_scale
    qcd.scale_error = qcd_scale_error
    ztautau.scale = ztautau_scale
    ztautau.scale_error = ztautau_scale_error

    #data_hist.GetFunction('model_%s' % category).Delete()

    # check norm in control region
    qcd_hist = (data_hist_control
                - ztautau_hist_control * ztautau_scale
                - bkg_hist_control) * qcd_scale

    factor = data_hist.Integral() / (qcd_hist + ztautau_hist * ztautau_scale +
            bkg_hist).Integral()

    # check norm overall
    ztautau_hist_overall = hist.Clone(title=ztautau.label)
    ztautau_hist_control_overall = hist.Clone(title=ztautau.label)

    bkg_hist_overall = hist.Clone(title=others.label)
    bkg_hist_control_overall = hist.Clone(title=others.label)

    data_hist_overall = hist.Clone(title=data.label)
    data_hist_control_overall = hist.Clone(title=data.label)

    ztautau.draw_into(
            ztautau_hist_overall,
            expr,
            category, target_region)

    ztautau.draw_into(
            ztautau_hist_control_overall,
            expr,
            category, qcd_shape_region)

    others.draw_into(
            bkg_hist_overall,
            expr,
            category, target_region)

    others.draw_into(
            bkg_hist_control_overall,
            expr,
            category, qcd_shape_region)

    data.draw_into(
            data_hist_overall,
            expr,
            category, target_region)

    data.draw_into(
            data_hist_control_overall,
            expr,
            category, qcd_shape_region)

    qcd_hist_overall = (data_hist_control_overall
                - ztautau_hist_control_overall
                - bkg_hist_control_overall) * qcd_scale

    overall_factor = data_hist_overall.Integral() / (qcd_hist_overall +
            ztautau_hist_overall + bkg_hist_overall).Integral()

    log.info("")
    log.info("data / model in this control region: %.3f" % factor)
    log.info("data / model overall: %.3f" % overall_factor)
    log.info("")

    if draw:
        if ndim == 1:
            draw_fit_1d(
                    expr, bins,
                    min, max,
                    model=[
                        qcd,
                        others,
                        ztautau],
                    data=data,
                    category=category,
                    region=target_region,
                    name=name,
                    output_name=output_name,
                    after=True,
                    systematics=systematics,
                    cuts=control,
                    root=root)

        else:
            draw_fit(
                    expr, bins,
                    xmin, xmax,
                    ymin, ymax,
                    model=[
                        qcd,
                        others,
                        ztautau],
                    data=data,
                    category=category,
                    region=target_region,
                    name=name,
                    output_name=output_name,
                    after=True,
                    systematics=systematics,
                    cuts=control,
                    root=root)

    bkg_scales_cache.set_scales(
            year,
            category, is_embedded, param,
            qcd_scale, qcd_scale_error,
            ztautau_scale, ztautau_scale_error)
