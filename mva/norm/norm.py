# deprecated:
"""
import os

import ROOT
ROOT.gROOT.SetBatch(True)
from ROOT import TF1, TF2, TLatex

from rootpy.tree import Cut
from rootpy.plotting import Hist, Hist2D, HistStack, Legend, Canvas

import numpy as np

from .. import log; log = log[__name__]
from ..plotting import set_colours, draw
from .. import plots_dir
from .. import categories
from .. import samples
from . import cache


# deprecated:
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

    if use_cache and cache.has_category(
            year, category, is_embedded, param):
        qcd_scale, qcd_scale_error, ztautau_scale, ztautau_scale_error = \
                 cache.get_scales(year, category, is_embedded, param)
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
    elif param == 'TRACK':
        xmin, xmax = .5, 4.5
        ymin, ymax = .5, 4.5
        bins = int(ymax - ymin) # ignore bins args above
        expr = 'tau1_numTrack_recounted:tau2_numTrack_recounted'
        xlabel = '#tau_{1} Number of Tracks'
        ylabel = '#tau_{2} Number of Tracks'
        name = 'Number of Tracks Grid'
    else:
        raise ValueError('No fit defined for %s parameters.' % param)

    output_name = "%s_fit_%s" % (param, category)
    if is_embedded:
        output_name += '_embedding'

    log.info("performing 2-dimensional fit using %s" % expr)
    log.info("using %d bins on each axis" % bins)

    control = mass_regions.control_region
    control &= cuts

    log.info("fitting scale factors in control region: %s" % control)

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
            cuts=control,
            p1p3=False)

    ztautau.draw_into(
            ztautau_hist_control,
            expr,
            category, qcd_shape_region,
            cuts=control,
            p1p3=False)

    others.draw_into(
            bkg_hist,
            expr,
            category, target_region,
            cuts=control,
            p1p3=False)

    others.draw_into(
            bkg_hist_control,
            expr,
            category, qcd_shape_region,
            cuts=control,
            p1p3=False)

    data.draw_into(
            data_hist,
            expr,
            category, target_region,
            cuts=control,
            p1p3=False)

    data.draw_into(
            data_hist_control,
            expr,
            category, qcd_shape_region,
            cuts=control,
            p1p3=False)

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

        def __init__(self):

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
            bin = model.FindBin(*(list(args)[:2]))
            return model.GetBinContent(bin)

    model = Model()
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
            category, target_region,
            p1p3=False)

    ztautau.draw_into(
            ztautau_hist_control_overall,
            expr,
            category, qcd_shape_region,
            p1p3=False)

    others.draw_into(
            bkg_hist_overall,
            expr,
            category, target_region,
            p1p3=False)

    others.draw_into(
            bkg_hist_control_overall,
            expr,
            category, qcd_shape_region,
            p1p3=False)

    data.draw_into(
            data_hist_overall,
            expr,
            category, target_region,
            p1p3=False)

    data.draw_into(
            data_hist_control_overall,
            expr,
            category, qcd_shape_region,
            p1p3=False)

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

    cache.set_scales(
            year,
            category, is_embedded, param,
            qcd_scale, qcd_scale_error,
            ztautau_scale, ztautau_scale_error)
"""
