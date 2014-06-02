# stdlib imports
import os
import sys
import math
import itertools
from itertools import izip

# numpy imports
import numpy as np

# matplotlib imports
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.font_manager as fm
from matplotlib import rc
from matplotlib.ticker import (AutoMinorLocator, NullFormatter,
                               MaxNLocator, FuncFormatter, MultipleLocator)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ROOT/rootpy imports
import ROOT
from rootpy.context import invisible_canvas
from rootpy.plotting import Canvas, Pad, Legend, Hist, Hist2D, HistStack, Graph
import rootpy.plotting.root2matplotlib as rplt
from rootpy.io import root_open
from rootpy.plotting.shapes import Line, Arrow
import rootpy.plotting.utils as rootpy_utils
from rootpy.plotting.style.atlas.labels import ATLAS_label
from rootpy.plotting.contrib.quantiles import qqgraph
from rootpy.plotting.contrib import plot_corrcoef_matrix
from rootpy.memory.keepalive import keepalive
from rootpy.stats.histfactory import HistoSys, split_norm_shape

# local imports
from .variables import VARIABLES
from . import PLOTS_DIR, MMC_MASS, save_canvas
from .systematics import iter_systematics, systematic_name
from .templates import RatioPlot
from . import log; log = log[__name__]

from statstools.utils import efficiency_cut, significance


def package_path(name):
    return os.path.splitext(os.path.abspath('latex/%s.sty' % name))[0]


LATEX_PREAMBLE = '''
\usepackage[EULERGREEK]{%s}
\sansmath
''' % package_path('sansmath')

"""
LATEX_PREAMBLE = '''
\usepackage[math-style=upright]{%s}
''' % package_path('unicode-math')
"""

#plt.rcParams['ps.useafm'] = True
#rc('text', usetex=True)
#rc('font', family='sans-serif')
rc('text.latex', preamble=LATEX_PREAMBLE)
#plt.rcParams['pdf.fonttype'] = 42


def set_colors(hists, colors=cm.jet):
    if hasattr(colors, '__call__'):
        for i, h in enumerate(hists):
            color = colors((i + 1) / float(len(hists) + 1))
            h.SetColor(color)
    else:
        for h, color in izip(hists, colors):
            h.SetColor(color)


def format_legend(l):
    #frame = l.get_frame()
    #frame.set_alpha(.8)
    #frame.set_fill(False) # eps does not support alpha values
    #frame.set_linewidth(0)
    for t in l.get_texts():
        # left align all contents
        t.set_ha('left')
    l.get_title().set_ha("left")


def root_axes(ax,
              xtick_formatter=None,
              xtick_locator=None,
              xtick_rotation=None,
              logy=False, integer=False, no_xlabels=False,
              vscale=1.,
              bottom=None):
    #ax.patch.set_linewidth(2)
    if integer:
        ax.xaxis.set_major_locator(
            xtick_locator or MultipleLocator(1))
        ax.tick_params(axis='x', which='minor',
                       bottom='off', top='off')
    else:
        ax.xaxis.set_minor_locator(AutoMinorLocator())

    if not logy:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    if no_xlabels:
        ax.xaxis.set_major_formatter(NullFormatter())
    elif xtick_formatter:
        ax.xaxis.set_major_formatter(xtick_formatter)

    if xtick_rotation is not None:
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=xtick_rotation)

    ax.yaxis.set_label_coords(-0.13, 1.)
    ax.xaxis.set_label_coords(1., -0.15 / vscale)


def draw_contours(hist, n_contours=3, contours=None,
                  linecolors=None, linestyles=None, linewidths=None,
                  labelcontours=True, labelcolors=None,
                  labelsizes=None, labelformats=None, same=False,
                  min_points=5):
    from rootpy import asrootpy
    if contours is None:
        contours = np.linspace(hist.min(), hist.max(), n_contours + 1,
                               endpoint=False)[1:]
    hist = hist.Clone()
    hist.SetContour(len(contours), np.asarray(contours, dtype=float))
    graphs = []
    levels = []
    with invisible_canvas() as c:
        hist.Draw('CONT LIST')
        c.Update()
        conts = asrootpy(ROOT.gROOT.GetListOfSpecials().FindObject('contours'))
        for i, cont in enumerate(conts):
            for curve in cont:
                if len(curve) < min_points:
                    continue
                graphs.append(curve.Clone())
                levels.append(contours[i])
    if not same:
        axes = hist.Clone()
        axes.Draw('AXIS')
    from itertools import cycle
    if linecolors is None:
        linecolors = ['black']
    elif not isinstance(linecolors, list):
        linecolors = [linecolors]
    if linestyles is None:
        from rootpy.plotting.base import linestyles_text2root
        linestyles = linestyles_text2root.keys()
    elif not isinstance(linestyles, list):
        linestyles = [linestyles]
    if linewidths is None:
        linewidths = [1]
    elif not isinstance(linewidths, list):
        linewidths = [linewidths]
    if labelsizes is not None:
        if not isinstance(labelsizes, list):
            labelsizes = [labelsizes]
        labelsizes = cycle(labelsizes)
    if labelcolors is not None:
        if not isinstance(labelcolors, list):
            labelcolors = [labelcolors]
        labelcolors = cycle(labelcolors)
    if labelformats is None:
        labelformats = ['%0.2g']
    elif not isinstance(labelformats, list):
        labelformats = [labelformats]
    linecolors = cycle(linecolors)
    linestyles = cycle(linestyles)
    linewidths = cycle(linewidths)
    labelformats = cycle(labelformats)
    from rootpy.plotting.base import convert_color
    from math import atan, pi, copysign
    label = ROOT.TLatex()
    xmin, xmax = hist.bounds(axis=0)
    ymin, ymax = hist.bounds(axis=1)
    stepx = (xmax - xmin) / 100
    stepy = (ymax - ymin) / 100
    for level, graph in zip(levels, graphs):
        graph.linecolor = linecolors.next()
        graph.linewidth = linewidths.next()
        graph.linestyle = linestyles.next()
        graph.Draw('C')
        if labelcontours:
            if labelsizes is not None:
                label.SetTextSize(labelsizes.next())
            if labelcolors is not None:
                label.SetTextColor(convert_color(labelcolors.next(), 'ROOT'))
            point_idx = len(graph) / 2
            point = graph[point_idx]
            padx, pady = 0, 0
            if len(graph) > 5:
                # use derivative to get text angle
                x1, y1 = graph[point_idx - 2]
                x2, y2 = graph[point_idx + 2]
                dx = (x2 - x1) / stepx
                dy = (y2 - y1) / stepy
                if dx == 0:
                    label.SetTextAngle(0)
                    label.SetTextAlign(12)
                else:
                    padx = copysign(1, -dy) * stepx
                    pady = copysign(1, dx) * stepy
                    if pady < 0:
                        align = 23
                    else:
                        align = 21
                    angle = atan(dy / dx) * 180 / pi
                    label.SetTextAngle(angle)
                    label.SetTextAlign(align)
            else:
                label.SetTextAngle(0)
                label.SetTextAlign(21)
            label.DrawLatex(point[0] + padx, point[1] + pady,
                            (labelformats.next()) % level)


def correlations(signal, signal_weight,
                 background, background_weight,
                 fields, category, output_suffix=''):
    names = [
        VARIABLES[field]['title'] if field in VARIABLES else field
            for field in fields]

    # draw correlation plots
    plot_corrcoef_matrix(signal, fields=names,
         output_name=os.path.join(PLOTS_DIR,
             "correlation_signal_%s%s.png" % (
             category.name, output_suffix)),
         title='%s Signal' % category.label,
         weights=signal_weight)
    plot_corrcoef_matrix(background, fields=names,
         output_name=os.path.join(PLOTS_DIR,
             "correlation_background_%s%s.png" % (
             category.name, output_suffix)),
         title='%s Background' % category.label,
         weights=background_weight)


def draw_scatter(fields,
                 category,
                 region,
                 output_name,
                 backgrounds,
                 signals=None,
                 data=None,
                 signal_scale=1.,
                 signal_colors=cm.spring,
                 classifier=None,
                 cuts=None,
                 unblind=False):
    nplots = 1
    figheight = 6.
    figwidth = 6.
    background_arrays = []
    background_clf_arrays = []
    for background in backgrounds:
        background_arrays.append(
            background.merged_records(
                category, region,
                fields=fields,
                cuts=cuts))
        if classifier is not None:
            background_clf_arrays.append(
                background.scores(
                    classifier,
                    category,
                    region,
                    cuts=cuts,
                    systematics=False)['NOMINAL'][0])

    if data is not None:
        nplots += 1
        figwidth += 6.
        data_array = data.merged_records(
            category, region,
            fields=fields,
            cuts=cuts)
        if classifier is not None:
            data_clf_array = data.scores(
                classifier,
                category,
                region,
                cuts=cuts)[0]

    if signals is not None:
        nplots += 1
        figwidth += 6.
        if data is not None:
            signal_index = 3
        else:
            signal_index = 2
        signal_arrays = []
        signal_clf_arrays = []
        for signal in signals:
            signal_arrays.append(
                signal.merged_records(
                    category, region,
                    fields=fields,
                    cuts=cuts))
            if classifier is not None:
                signal_clf_arrays.append(
                    signal.scores(
                        classifier,
                        category,
                        region,
                        cuts=cuts,
                        systematics=False)['NOMINAL'][0])

    if classifier is not None:
        fields = fields + [classifier]
    all_pairs = list(itertools.combinations(fields, 2))

    for x, y in all_pairs:
        # always make the classifier along the x axis
        if not isinstance(y, basestring):
            tmp = x
            x = y
            y = tmp

        with_classifier = not isinstance(x, basestring)

        plt.figure(figsize=(figwidth, figheight), dpi=200)
        axes = []

        ax_bkg = plt.subplot(1, nplots, 1)
        axes.append(ax_bkg)

        if not with_classifier:
            xscale = VARIABLES[x].get('scale', 1.)
        yscale = VARIABLES[y].get('scale', 1.)

        xmin, xmax = float('inf'), float('-inf')
        ymin, ymax = float('inf'), float('-inf')

        for i, (array, background) in enumerate(zip(background_arrays,
                                                    backgrounds)):
            if with_classifier:
                x_array = background_clf_arrays[i]
            else:
                x_array = array[x] * xscale
            y_array = array[y] * yscale

            # update max and min bounds
            lxmin, lxmax = x_array.min(), x_array.max()
            lymin, lymax = y_array.min(), y_array.max()
            if lxmin < xmin:
                xmin = lxmin
            if lxmax > xmax:
                xmax = lxmax
            if lymin < ymin:
                ymin = lymin
            if lymax > ymax:
                ymax = lymax

            weight = array['weight']
            ax_bkg.scatter(
                    x_array, y_array,
                    c=background.hist_decor['color'],
                    label=background.label,
                    s=weight * 10,
                    #edgecolors='',
                    linewidths=1,
                    marker='o',
                    alpha=0.75)

        if data is not None:
            data_ax = plt.subplot(1, nplots, 2)
            axes.append(data_ax)

            if with_classifier:
                x_array = data_clf_array
            else:
                x_array = data_array[x] * xscale
            y_array = data_array[y] * yscale

            # if blinded don't show above the midpoint of the BDT score
            if with_classifier and not unblind:
                midpoint = (x_array.max() + x_array.min()) / 2.
                x_array = x_array[data_clf_array < midpoint]
                y_array = y_array[data_clf_array < midpoint]
                data_ax.text(0.9, 0.2, 'BLINDED',
                                  verticalalignment='center',
                                  horizontalalignment='right',
                                        transform=data_ax.transAxes,
                                        fontsize=20)

            # update max and min bounds
            lxmin, lxmax = x_array.min(), x_array.max()
            lymin, lymax = y_array.min(), y_array.max()
            if lxmin < xmin:
                xmin = lxmin
            if lxmax > xmax:
                xmax = lxmax
            if lymin < ymin:
                ymin = lymin
            if lymax > ymax:
                ymax = lymax

            weight = data_array['weight']
            data_ax.scatter(
                    x_array, y_array,
                    c='black',
                    label=data.label,
                    s=weight * 10,
                    #edgecolors='',
                    linewidths=0,
                    marker='.')

        if signal is not None:
            sig_ax = plt.subplot(1, nplots, signal_index)
            axes.append(sig_ax)

            for i, (array, signal) in enumerate(zip(signal_arrays, signals)):

                if with_classifier:
                    x_array = signal_clf_arrays[i]
                else:
                    x_array = array[x] * xscale
                y_array = array[y] * yscale

                # update max and min bounds
                lxmin, lxmax = x_array.min(), x_array.max()
                lymin, lymax = y_array.min(), y_array.max()
                if lxmin < xmin:
                    xmin = lxmin
                if lxmax > xmax:
                    xmax = lxmax
                if lymin < ymin:
                    ymin = lymin
                if lymax > ymax:
                    ymax = lymax
                color = signal_colors((i + 1) / float(len(signals) + 1))
                weight = array['weight']
                sig_ax.scatter(
                        x_array, y_array,
                        c=color,
                        label=signal.label,
                        s=weight * 10 * signal_scale,
                        #edgecolors='',
                        linewidths=0,
                        marker='o',
                        alpha=0.75)

        xwidth = xmax - xmin
        ywidth = ymax - ymin
        xpad = xwidth * .1
        ypad = ywidth * .1

        if with_classifier:
            x_name = "BDT Score"
            x_filename = "bdt_score"
            x_units = None
        else:
            x_name = VARIABLES[x]['title']
            x_filename = VARIABLES[x]['filename']
            x_units = VARIABLES[x].get('units', None)

        y_name = VARIABLES[y]['title']
        y_filename = VARIABLES[y]['filename']
        y_units = VARIABLES[y].get('units', None)

        for ax in axes:
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)

            ax.legend(loc='upper right')

            if x_units is not None:
                ax.set_xlabel('%s [%s]' % (x_name, x_units))
            else:
                ax.set_xlabel(x_name)
            if y_units is not None:
                ax.set_ylabel('%s [%s]' % (y_name, y_units))
            else:
                ax.set_ylabel(y_name)

        plt.suptitle(category.label)
        plt.savefig(os.path.join(PLOTS_DIR, 'scatter_%s_%s_%s%s.png') % (
            category.name, x_filename, y_filename, output_name),
            bbox_inches='tight')

        """
        Romain Madar:

        Display the 1D histogram of (x_i - <x>)(y_i - <y>) over the events {i}.
        The mean of this distribution will be the "usual correlation" but this
        plot allows to look at the tails and asymmetry, for data and MC.
        """


def get_2d_field_hist(var):
    var_info = VARIABLES[var]
    bins = var_info['bins']
    min, max = var_info['range']
    hist = Hist2D(100, min, max, 100, -1, 1)
    return hist



def draw_2d_hist(classifier,
                 category,
                 region,
                 backgrounds,
                 signals=None,
                 data=None,
                 cuts=None,
                 y=MMC_MASS,
                 output_suffix=''):
    fields = [y]
    background_arrays = []
    background_clf_arrays = []
    for background in backgrounds:
        sys_mass = {}
        for systematic in iter_systematics(True):
            sys_mass[systematic] = (
                background.merged_records(
                    category, region,
                    fields=fields,
                    cuts=cuts,
                    systematic=systematic))
        background_arrays.append(sys_mass)
        background_clf_arrays.append(
            background.scores(
                classifier,
                category,
                region,
                cuts=cuts,
                systematics=True))

    if signals is not None:
        signal_arrays = []
        signal_clf_arrays = []
        for signal in signals:
            sys_mass = {}
            for systematic in iter_systematics(True):
                sys_mass[systematic] = (
                    signal.merged_records(
                        category, region,
                        fields=fields,
                        cuts=cuts,
                        systematic=systematic))
            signal_arrays.append(sys_mass)
            signal_clf_arrays.append(
                signal.scores(
                    classifier,
                    category,
                    region,
                    cuts=cuts,
                    systematics=True))

    xmin, xmax = float('inf'), float('-inf')
    if data is not None:
        data_array = data.merged_records(
            category, region,
            fields=fields,
            cuts=cuts)
        data_clf_array = data.scores(
            classifier,
            category,
            region,
            cuts=cuts)[0]
        lxmin, lxmax = data_clf_array.min(), data_clf_array.max()
        if lxmin < xmin:
            xmin = lxmin
        if lxmax > xmax:
            xmax = lxmax

    for array_dict in background_clf_arrays + signal_clf_arrays:
        for sys, (array, _) in array_dict.items():
            lxmin, lxmax = array.min(), array.max()
            if lxmin < xmin:
                xmin = lxmin
            if lxmax > xmax:
                xmax = lxmax

    yscale = VARIABLES[y].get('scale', 1.)

    if cuts:
        output_suffix += '_' + cuts.safe()
    output_name = "histos_2d_" + category.name + output_suffix + ".root"
    hist_template = get_2d_field_hist(y)

    # scale BDT scores such that they are between -1 and 1
    xscale = max(abs(xmax), abs(xmin))

    with root_open(output_name, 'recreate') as f:

        for background, array_dict, clf_dict in zip(backgrounds,
                                                    background_arrays,
                                                    background_clf_arrays):
            for systematic in iter_systematics(True):
                x_array = clf_dict[systematic][0] / xscale
                y_array = array_dict[systematic][y] * yscale
                weight = array_dict[systematic]['weight']
                hist = hist_template.Clone(name=background.name +
                        ('_%s' % systematic_name(systematic)))
                hist.fill_array(np.c_[y_array, x_array], weights=weight)
                hist.Write()

        if signal is not None:
            for signal, array_dict, clf_dict in zip(signals,
                                                    signal_arrays,
                                                    signal_clf_arrays):
                for systematic in iter_systematics(True):
                    x_array = clf_dict[systematic][0] / xscale
                    y_array = array_dict[systematic][y] * yscale
                    weight = array_dict[systematic]['weight']
                    hist = hist_template.Clone(name=signal.name +
                            ('_%s' % systematic_name(systematic)))
                    hist.fill_array(np.c_[y_array, x_array], weights=weight)
                    hist.Write()

        if data is not None:
            x_array = data_clf_array / xscale
            y_array = data_array[y] * yscale
            weight = data_array['weight']
            hist = hist_template.Clone(name=data.name)
            hist.fill_array(np.c_[y_array, x_array], weights=weight)
            hist.Write()


def uncertainty_band(model, systematics, systematics_components):
    # TODO determine systematics from model itself
    if not isinstance(model, (list, tuple)):
        model = [model]
    # add separate variations in quadrature
    # also include stat error in quadrature
    total_model = sum(model)
    var_high = []
    var_low = []
    for term, variations in systematics.items():
        if len(variations) == 2:
            high, low = variations
        elif len(variations) == 1:
            high = variations[0]
            low = 'NOMINAL'
        else:
            print variations
            raise ValueError(
                "only one or two variations per term are allowed")

        if systematics_components is not None:
            if high not in systematics_components:
                log.warning("filtering out {0}".format(high))
                high = 'NOMINAL'
            if low not in systematics_components:
                log.warning("filtering out {0}".format(low))
                low = 'NOMINAL'

        if high == 'NOMINAL' and low == 'NOMINAL':
            continue

        total_high = model[0].Clone()
        total_high.Reset()
        total_low = total_high.Clone()
        total_max = total_high.Clone()
        total_min = total_high.Clone()
        for m in model:
            if high == 'NOMINAL' or high not in m.systematics:
                total_high += m.Clone()
            else:
                #print m.title, high, list(m.systematics[high])
                total_high += m.systematics[high]

            if low == 'NOMINAL' or low not in m.systematics:
                total_low += m.Clone()
            else:
                #print m.title, low, list(m.systematics[low])
                total_low += m.systematics[low]

        if total_low.Integral() <= 0:
            log.warning("{0}_DOWN is non-positive".format(term))
        if total_high.Integral() <= 0:
            log.warning("{0}_UP is non-positive".format(term))

        for i in total_high.bins_range():
            total_max[i].value = max(total_high[i].value, total_low[i].value, total_model[i].value)
            total_min[i].value = min(total_high[i].value, total_low[i].value, total_model[i].value)

        if total_min.Integral() <= 0:
            log.warning("{0}: lower bound is non-positive".format(term))
        if total_max.Integral() <= 0:
            log.warning("{0}: upper bound is non-positive".format(term))

        var_high.append(total_max)
        var_low.append(total_min)

        log.warning("{0}, {1}".format(str(term), str(variations)))
        log.warning("{0} {1}".format(total_max.integral(), total_min.integral()))

    log.info(str(systematics_components))
    if systematics_components is None:
        # include stat error variation
        total_model_stat_high = total_model.Clone()
        total_model_stat_low = total_model.Clone()
        for i in xrange(len(total_model)):
            total_model_stat_high[i].value += total_model.yerrh(i)
            total_model_stat_low[i].value -= total_model.yerrl(i)
        var_high.append(total_model_stat_high)
        var_low.append(total_model_stat_low)

    # sum variations in quadrature bin-by-bin
    high_band = total_model.Clone()
    high_band.Reset()
    low_band = high_band.Clone()
    for i in xrange(len(high_band)):
        sum_high = math.sqrt(
            sum([(v[i].value - total_model[i].value)**2 for v in var_high]))
        sum_low = math.sqrt(
            sum([(v[i].value - total_model[i].value)**2 for v in var_low]))
        high_band[i] = sum_high
        low_band[i] = sum_low
    return total_model, high_band, low_band


def draw_samples(
        hist_template,
        expr,
        category,
        region,
        model,
        data=None,
        signal=None,
        cuts=None,
        ravel=True,
        weighted=True,
        **kwargs):
    """
    extra kwargs are passed to draw()
    """
    hist_template = hist_template.Clone()
    hist_template.Reset()
    ndim = hist_template.GetDimension()

    model_hists = []
    for sample in model:
        hist = hist_template.Clone(title=sample.label, **sample.hist_decor)
        hist.decorate(**sample.hist_decor)
        sample.draw_into(hist, expr,
                category, region, cuts,
                weighted=weighted)
        if ndim > 1 and ravel:
            # ravel() the nominal and systematics histograms
            sys_hists = getattr(hist, 'systematics', None)
            hist = hist.ravel()
            hist.title = sample.label
            hist.decorate(**sample.hist_decor)
            if sys_hists is not None:
                hist.systematics = sys_hists
            if hasattr(hist, 'systematics'):
                sys_hists = {}
                for term, _hist in hist.systematics.items():
                    sys_hists[term] = _hist.ravel()
                hist.systematics = sys_hists
        model_hists.append(hist)

    if signal is not None:
        signal_hists = []
        for sample in signal:
            hist = hist_template.Clone(title=sample.label, **sample.hist_decor)
            hist.decorate(**sample.hist_decor)
            sample.draw_into(hist, expr,
                    category, region, cuts,
                    weighted=weighted)
            if ndim > 1 and ravel:
                # ravel() the nominal and systematics histograms
                sys_hists = getattr(hist, 'systematics', None)
                hist = hist.ravel()
                hist.title = sample.label
                hist.decorate(**sample.hist_decor)
                if sys_hists is not None:
                    hist.systematics = sys_hists
                if hasattr(hist, 'systematics'):
                    sys_hists = {}
                    for term, _hist in hist.systematics.items():
                        sys_hists[term] = _hist.ravel()
                    hist.systematics = sys_hists
            signal_hists.append(hist)
    else:
        signal_hists = None

    if data is not None:
        data_hist = hist_template.Clone(title=data.label, **data.hist_decor)
        data_hist.decorate(**data.hist_decor)
        data.draw_into(data_hist, expr, category, region, cuts,
                       weighted=weighted)
        if ndim > 1 and ravel:
            data_hist = data_hist.ravel()

        log.info("Data events: %d" % sum(data_hist.y()))
        log.info("Model events: %f" % sum(sum(model_hists).y()))
        for hist in model_hists:
            log.info("{0} {1}".format(hist.GetTitle(), sum(hist.y())))
        if signal is not None:
            log.info("Signal events: %f" % sum(sum(signal_hists).y()))
        log.info("Data / Model: %f" % (sum(data_hist.y()) /
            sum(sum(model_hists).y())))

    else:
        data_hist = None

    draw(model=model_hists,
         data=data_hist,
         signal=signal_hists,
         category=category,
         **kwargs)


def draw_samples_array(
        vars,
        category,
        region,
        model,
        data=None,
        signal=None,
        cuts=None,
        ravel=False,
        weighted=True,
        weight_hist=None,
        clf=None,
        min_score=None,
        max_score=None,
        plots=None,
        output_suffix='',
        unblind=False,
        **kwargs):
    """
    extra kwargs are passed to draw()
    """
    # filter out plots that will not be made
    used_vars = {}
    field_scale = {}
    if plots is not None:
        for plot in plots:
            if plot in vars:
                var_info = vars[plot]
                if (var_info.get('cats', None) is not None and
                    category.name.upper() not in var_info['cats']):
                    raise ValueError(
                        "variable %s is not valid in the category %s" %
                        (plot, category.name.upper()))
                used_vars[plot] = var_info
                if 'scale' in var_info:
                    field_scale[plot] = var_info['scale']
            else:
                raise ValueError(
                    "variable %s is not defined in mva/variables.py" % plot)
    else:
        for expr, var_info in vars.items():
            if (var_info.get('cats', None) is not None and
                category.name.upper() not in var_info['cats']):
                continue
            used_vars[expr] = var_info
            if 'scale' in var_info:
                field_scale[expr] = var_info['scale']
    vars = used_vars
    if not vars:
        raise RuntimeError("no variables selected")

    model_hists = []
    for sample in model:
        field_hist, _ = sample.get_field_hist(vars)
        sample.draw_array(field_hist,
            category, region, cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            clf=clf,
            min_score=min_score,
            max_score=max_score,)
        model_hists.append(field_hist)

    if signal is not None:
        signal_hists = []
        for sample in signal:
            field_hist, _ = sample.get_field_hist(vars)
            sample.draw_array(field_hist,
                category, region, cuts,
                weighted=weighted,
                field_scale=field_scale,
                weight_hist=weight_hist,
                clf=clf,
                min_score=min_score,
                max_score=max_score)
            signal_hists.append(field_hist)
    else:
        signal_hists = None

    if data is not None:
        data_field_hist, _ = data.get_field_hist(vars)
        data.draw_array(data_field_hist, category, region, cuts,
            weighted=weighted,
            field_scale=field_scale,
            weight_hist=weight_hist,
            clf=clf,
            min_score=min_score,
            max_score=max_score)
        """
        log.info("Data events: %d" % sum(data_hist))
        log.info("Model events: %f" % sum(sum(model_hists)))
        for hist in model_hists:
            log.info("{0} {1}".format(hist.GetTitle(), sum(hist)))
        if signal is not None:
            log.info("Signal events: %f" % sum(sum(signal_hists)))
        log.info("Data / Model: %f" % (sum(data_hist) /
            sum(sum(model_hists))))
        """

    else:
        data_field_hist = None

    figs = {}
    for field, var_info in vars.items():
        if unblind:
            blind = False
        else:
            blind = var_info.get('blind', False)
        output_name = var_info['filename'] + output_suffix
        if cuts:
            output_name += '_' + cuts.safe()

        fig = draw(model=[m[field] for m in model_hists],
             data=data_field_hist[field] if data_field_hist else None,
             data_info=str(data_field_hist[field].datainfo) if data_field_hist else None,
             signal=[s[field] for s in signal_hists] if signal_hists else None,
             category=category,
             name=var_info['root'],
             units=var_info.get('units', None),
             output_name=output_name,
             blind=blind,
             integer=var_info.get('integer', False),
             **kwargs)
        figs[field] = fig
    return figs


def draw_channel_array(
        analysis,
        vars,
        category,
        region,
        cuts=None,
        mass=125,
        mode=None,
        scale_125=False,
        ravel=False,
        weighted=True,
        weight_hist=None,
        clf=None,
        min_score=None,
        max_score=None,
        templates=None,
        plots=None,
        output_suffix='',
        unblind=False,
        bootstrap_data=False,
        **kwargs):
    # filter out plots that will not be made
    used_vars = {}
    field_scale = {}
    if plots is not None:
        for plot in plots:
            if plot in vars:
                var_info = vars[plot]
                if (var_info.get('cats', None) is not None and
                    category.name.upper() not in var_info['cats']):
                    raise ValueError(
                        "variable %s is not valid in the category %s" %
                        (plot, category.name.upper()))
                used_vars[plot] = var_info
                if 'scale' in var_info:
                    field_scale[plot] = var_info['scale']
            else:
                raise ValueError(
                    "variable %s is not defined in mva/variables.py" % plot)
    else:
        for expr, var_info in vars.items():
            if (var_info.get('cats', None) is not None and
                category.name.upper() not in var_info['cats']):
                continue
            used_vars[expr] = var_info
            if 'scale' in var_info:
                field_scale[expr] = var_info['scale']
    vars = used_vars
    if not vars:
        raise RuntimeError("no variables selected")

    field_channel = analysis.get_channel_array(vars,
        category, region, cuts,
        include_signal=True,
        mass=mass,
        mode=mode,
        scale_125=scale_125,
        clf=clf,
        min_score=min_score,
        max_score=max_score,
        weighted=weighted,
        templates=templates,
        field_scale=field_scale,
        weight_hist=weight_hist,
        no_signal_fixes=True,
        bootstrap_data=bootstrap_data)

    figs = {}
    for field, var_info in vars.items():
        if unblind:
            blind = False
        else:
            blind = var_info.get('blind', False)
        output_name = var_info['filename'] + output_suffix
        if cuts:
            output_name += '_' + cuts.safe()
        ypadding = kwargs.pop('ypadding', var_info.get('ypadding', None))
        legend_position = kwargs.pop('legend_position', var_info.get('legend', 'right'))
        fig = draw_channel(field_channel[field],
                           data_info=str(analysis.data.info),
                           category=category,
                           name=var_info['root'],
                           units=var_info.get('units', None),
                           output_name=output_name,
                           blind=blind,
                           integer=var_info.get('integer', False),
                           ypadding=ypadding,
                           legend_position=legend_position,
                           **kwargs)
        figs[field] = fig
    return field_channel, figs


def draw_channel(channel, fit=None, no_data=False, **kwargs):
    """
    Draw a HistFactory::Channel only include OverallSys systematics
    in resulting band as an illustration of the level of uncertainty
    since correlations of the NPs are not known and it is not
    possible to draw the statistically correct error band.
    """
    if fit is not None:
        log.warning("applying snapshot on channel {0}".format(channel.name))
        channel = channel.apply_snapshot(fit)
    if channel.data and channel.data.hist and not no_data:
        data_hist = channel.data.hist
    else:
        data_hist = None
    model_hists = []
    signal_hists = []
    systematics_terms = {}
    sys_names = channel.sys_names()
    for sample in channel.samples:
        nominal_hist = sample.hist
        _systematics = {}
        for sys_name in sys_names:
            systematics_terms[sys_name] = (
                sys_name + '_UP',
                sys_name + '_DOWN')
            dn_hist, up_hist = sample.sys_hist(sys_name)
            hsys = HistoSys(sys_name, low=dn_hist, high=up_hist)
            norm, shape = split_norm_shape(hsys, nominal_hist)
            # include only overallsys component
            _systematics[sys_name + '_DOWN'] = nominal_hist * norm.low
            _systematics[sys_name + '_UP'] = nominal_hist * norm.high
        nominal_hist.systematics = _systematics
        if sample.GetNormFactor('SigXsecOverSM') is not None:
            signal_hists.append(nominal_hist)
        else:
            model_hists.append(nominal_hist)
    if 'systematics' in kwargs:
        del kwargs['systematics']
    figs = []
    for logy in (False, True):
        figs.append(draw(
            data=data_hist,
            model=model_hists or None,
            signal=signal_hists or None,
            systematics=systematics_terms,
            logy=logy,
            **kwargs))
    return figs


def format_plot(pad, template, xaxis, yaxis,
                xaxes=None, xlimits=None,
                ylabel='Events', xlabel=None,
                units=None, data_info=None,
                left_label=None, right_label=None,
                atlas_label='Internal',
                textsize=22,
                integer=False):

    # set the axis labels
    binw = list(template.xwidth())
    binwidths = list(set(['%.2g' % w for w in binw]))
    if units is not None:
        if xlabel is not None:
            xlabel = '%s [%s]' % (xlabel, units)
        if ylabel and len(binwidths) == 1 and binwidths[0] != '1':
            # constant width bins
            ylabel = '%s / %s %s' % (ylabel, binwidths[0], units)
    elif ylabel and len(binwidths) == 1 and binwidths[0] != '1':
        ylabel = '%s / %s' % (ylabel, binwidths[0])

    if ylabel:
        yaxis.SetTitle(ylabel)
    if xlabel:
        xaxis.SetTitle(xlabel)

    divisions = min(template.nbins(), 7) if integer else 507
    if xlimits is None:
        xlimits = template.bounds()

    if xaxes is None:
        xaxes = [xaxis]

    for axis in xaxes:
        axis.SetLimits(*xlimits)
        axis.SetRangeUser(*xlimits)
        axis.SetNdivisions(divisions)

    # draw the left label
    if left_label:
        label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.03, 0.89,
            left_label)
        label.SetNDC()
        label.SetTextFont(43)
        label.SetTextSize(textsize)
        with pad:
            label.Draw()
        keepalive(pad, label)

    # draw the right label
    if right_label is not None:
        label = ROOT.TLatex(0.7, 0.82, right_label)
        label.SetNDC()
        label.SetTextFont(43)
        label.SetTextSize(textsize)
        with pad:
            label.Draw()
        keepalive(pad, label)

    # draw the luminosity label
    if data_info is not None:
        plabel = ROOT.TLatex(
            pad.GetLeftMargin() + 0.03, 0.82,
            str(data_info))
        plabel.SetNDC()
        plabel.SetTextFont(43)
        plabel.SetTextSize(textsize)
        with pad:
            plabel.Draw()
        keepalive(pad, plabel)

    # draw the ATLAS label
    if atlas_label:
        ATLAS_label(0.67, 0.89,
                    sep=0.132, pad=pad, sqrts=None,
                    text=atlas_label, textsize=textsize)

    pad.Update()
    pad.Modified()


def draw(name,
         category,
         data=None,
         data_info=None,
         model=None,
         model_colors=None,
         signal=None,
         signal_scale=1.,
         signal_on_top=False,
         signal_linestyles=None,
         signal_colors=None,
         show_signal_error=False,
         fill_signal=False,
         stack_signal=True,
         units=None,
         plot_label=None,
         ylabel='Events',
         blind=False,
         show_ratio=False,
         ratio_range=None,
         ratio_height=0.15,
         ratio_margin=0.05,
         output_formats=None,
         systematics=None,
         systematics_components=None,
         integer=False,
         textsize=22,
         logy=False,
         separate_legends=False,
         legend_leftmargin=0.39,
         ypadding=None,
         legend_position='right',
         range=None,
         output_name=None,
         output_dir=PLOTS_DIR,
         arrow_values=None,
         overflow=True,
         show_pvalue=True,
         top_label=None):

    if model is None and data is None and signal is None:
        # insufficient input
        raise ValueError(
            "at least one of model, data, "
            "or signal must be specified")

    if model is not None:
        if not isinstance(model, (list, tuple)):
            model = [model]
        if overflow:
            for hist in model:
                hist[1] += hist[0]
                hist[-2] += hist[-1]
    if signal is not None:
        if not isinstance(signal, (list, tuple)):
            signal = [signal]
        if overflow:
            for hist in signal:
                hist[1] += hist[0]
                hist[-2] += hist[-1]
    if data is not None and overflow:
        data[1] += data[0]
        data[-2] += data[-1]

    # objects will be populated with all histograms in the main pad
    objects = []
    legends = []

    if show_ratio and (data is None or model is None):
        # cannot show the ratio if data or model was not specified
        show_ratio=False

    if ypadding is None:
        # select good defaults for log or linear scales
        if logy:
            ypadding = (.6, .0)
        else:
            ypadding = (.35, .0)

    if show_ratio:
        # define the pad setup for the ratio plot
        if ratio_range is None:
            ratio_range = (0, 2)

        # start with defaults in current gStyle
        prev_style = ROOT.gStyle
        style = prev_style.Clone()

        # plot dimensions in pixels
        figheight = baseheight = style.GetCanvasDefH()
        figwidth = basewidth = style.GetCanvasDefW()

        # margins
        left_margin = style.GetPadLeftMargin()
        bottom_margin = style.GetPadBottomMargin()
        top_margin = style.GetPadTopMargin()
        right_margin = style.GetPadRightMargin()

        figheight += (ratio_height + ratio_margin) * figheight
        ratio_height += bottom_margin + ratio_margin / 2.

        style.SetTitleYOffset(style.GetTitleYOffset() * figheight / baseheight)
        style.cd()

        # main canvas
        fig = Canvas(width=int(figwidth), height=int(figheight))
        fig.SetMargin(0, 0, 0, 0)

        # top pad for histograms
        hist_pad = Pad(0., ratio_height, 1., 1.,
                       name='top', title='top')
        hist_pad.SetBottomMargin(ratio_margin / 2.)
        hist_pad.SetTopMargin(top_margin)
        hist_pad.SetLeftMargin(left_margin)
        hist_pad.SetRightMargin(right_margin)
        hist_pad.Draw()

        # bottom pad for ratio plot
        fig.cd()
        ratio_pad = Pad(0, 0, 1, ratio_height,
                        name='ratio', title='ratio')
        ratio_pad.SetBottomMargin(bottom_margin / ratio_height)
        ratio_pad.SetTopMargin(ratio_margin / (2. * ratio_height))
        ratio_pad.SetLeftMargin(left_margin)
        ratio_pad.SetRightMargin(right_margin)
        ratio_pad.Draw()
        hist_pad.cd()
    else:
        prev_style = None
        # simple case without ratio plot
        fig = Canvas()
        hist_pad = fig

    if logy:
        hist_pad.SetLogy()

    if signal is not None:
        if signal_scale != 1.:
            scaled_signal = []
            for sig in signal:
                scaled_h = sig * signal_scale
                scaled_h.SetTitle(r'%g #times %s' % (
                    signal_scale,
                    sig.GetTitle()))
                scaled_signal.append(scaled_h)
        else:
            scaled_signal = signal
        if signal_colors is not None:
            set_colors(scaled_signal, signal_colors)
        for i, s in enumerate(scaled_signal):
            s.drawstyle = 'HIST'
            if fill_signal:
                s.fillstyle = 'solid'
                s.fillcolor = s.linecolor
                s.linewidth = 0
                s.linestyle = 'solid'
                alpha = .75
            else:
                s.fillstyle = 'hollow'
                s.linewidth = 3
                if signal_linestyles is not None:
                    s.linestyle = signal_linestyles[i]
                else:
                    s.linestyle = 'solid'
                alpha = 1.

    if model is not None:
        if model_colors is not None:
            set_colors(model, model_colors)
        # create the model stack
        model_stack = HistStack()
        for hist in model:
            hist.SetLineWidth(0)
            hist.drawstyle = 'hist'
            model_stack.Add(hist)
        if signal is not None and signal_on_top:
            for s in scaled_signal:
                model_stack.Add(s)
        objects.append(model_stack)

    if signal is not None and not signal_on_top:
        if stack_signal:
            # create the signal stack
            signal_stack = HistStack()
            for hist in scaled_signal:
                signal_stack.Add(hist)
            objects.append(signal_stack)
        else:
            objects.extend(scaled_signal)

    if model is not None:
        # draw uncertainty band
        total_model, high_band_model, low_band_model = uncertainty_band(
            model, systematics, systematics_components)
        high = total_model + high_band_model
        low = total_model - low_band_model
        error_band_model = rootpy_utils.get_band(
            low, high,
            middle_hist=total_model)
        error_band_model.fillstyle = '/'
        error_band_model.fillcolor = 13
        error_band_model.linecolor = 10
        error_band_model.markersize = 0
        error_band_model.markercolor = 10
        error_band_model.drawstyle = 'e2'
        objects.append(error_band_model)

    if signal is not None and show_signal_error:
        total_signal, high_band_signal, low_band_signal = uncertainty_band(
            signal, systematics, systematics_components)
        high = (total_signal + high_band_signal) * signal_scale
        low = (total_signal - low_band_signal) * signal_scale
        if signal_on_top:
            high += total_model
            low += total_model
        hist_pad.cd()
        error_band_signal = rootpy_utils.get_band(
            low, high,
            middle_hist=total_signal * signal_scale)
        error_band_signal.fillstyle = '\\'
        error_band_signal.fillcolor = 13
        error_band_signal.linecolor = 10
        error_band_signal.markersize = 0
        error_band_signal.markercolor = 10
        error_band_signal.drawstyle = 'e2'
        objects.append(error_band_signal)

    if data is not None and blind is not True:
        # create the data histogram
        if isinstance(blind, tuple):
            low, high = blind
            # zero out bins in blind region
            for bin in data.bins():
                if (low < bin.x.high <= high or low <= bin.x.low < high):
                    data[bin.idx] = (0., 0.)
        # convert data to TGraphAsymmErrors with Poisson errors
        data_poisson = data.poisson_errors()
        data_poisson.markersize = 1.2
        data_poisson.drawstyle = 'PZ'
        objects.append(data_poisson)

        # draw ratio plot
        if model is not None and show_ratio:
            ratio_pad.cd()
            total_model = sum(model)
            ratio_hist = Hist.divide(data, total_model)
            # remove bins where data is zero
            max_dev = 0
            for bin in data.bins():
                if bin.value <= 0:
                    ratio_hist[bin.idx] = (-100, 0)
                else:
                    ratio_value = ratio_hist[bin.idx].value
                    dev =  abs(ratio_value - 1)
                    if dev > max_dev:
                        max_dev = dev

            if max_dev < 0.2:
                ratio_range = (0.8, 1.2)
            elif max_dev < 0.4:
                ratio_range = (0.6, 1.4)

            ruler_high = (ratio_range[1] + 1.) / 2.
            ruler_low = (ratio_range[0] + 1.) / 2.

            ratio_hist.linecolor = 'black'
            ratio_hist.linewidth = 2
            ratio_hist.fillstyle = 'hollow'

            # draw empty copy of ratio_hist first so lines will show
            ratio_hist_tmp = ratio_hist.Clone()
            ratio_hist_tmp.Reset()
            ratio_hist_tmp.Draw()
            ratio_hist_tmp.yaxis.SetLimits(*ratio_range)
            ratio_hist_tmp.yaxis.SetRangeUser(*ratio_range)
            ratio_hist_tmp.yaxis.SetTitle('Data / Model')
            ratio_hist_tmp.yaxis.SetNdivisions(4)
            # not certain why the following is needed
            ratio_hist_tmp.yaxis.SetTitleOffset(style.GetTitleYOffset())

            ratio_xrange = range or ratio_hist.bounds()

            ratio_hist_tmp.xaxis.SetLimits(*ratio_xrange)
            #ratio_hist_tmp.xaxis.SetRangeUser(*ratio_xrange)

            ratio_hist_tmp.xaxis.SetTickLength(
                ratio_hist_tmp.xaxis.GetTickLength() * 2)

            # draw ratio=1 line
            line = Line(ratio_xrange[0], 1,
                        ratio_xrange[1], 1)
            line.linestyle = 'dashed'
            line.linewidth = 2
            line.Draw()

            # draw high ratio line
            line_up = Line(ratio_xrange[0], ruler_high,
                           ratio_xrange[1], ruler_high)
            line_up.linestyle = 'dashed'
            line_up.linewidth = 2
            line_up.Draw()

            # draw low ratio line
            line_dn = Line(ratio_xrange[0], ruler_low,
                           ratio_xrange[1], ruler_low)
            line_dn.linestyle = 'dashed'
            line_dn.linewidth = 2
            line_dn.Draw()

            # draw band below points on ratio plot
            ratio_hist_high = Hist.divide(
                total_model + high_band_model, total_model)
            ratio_hist_low = Hist.divide(
                total_model - low_band_model, total_model)
            ratio_pad.cd()
            error_band = rootpy_utils.get_band(
                ratio_hist_high, ratio_hist_low)
            error_band.fillstyle = '/'
            error_band.fillcolor = '#858585'
            error_band.Draw('same E2')

            # draw points above band
            ratio_hist.Draw('same E0')

    if separate_legends:
        right_legend = Legend(len(signal) + 1 if signal is not None else 1,
            pad=hist_pad,
            leftmargin=legend_leftmargin,
            rightmargin=0.12,
            margin=0.35,
            textsize=textsize,
            entrysep=0.02,
            entryheight=0.04,
            topmargin=0.15)
        right_legend.AddEntry(data, style='lep')
        if signal is not None:
            for s in reversed(scaled_signal):
                right_legend.AddEntry(s, style='F' if fill_signal else 'L')
        legends.append(right_legend)
        if model is not None:
            n_entries = len(model)
            if systematics:
                n_entries += 1
            model_legend = Legend(n_entries,
                pad=hist_pad,
                leftmargin=0.05,
                rightmargin=0.46,
                margin=0.35,
                textsize=textsize,
                entrysep=0.02,
                entryheight=0.04,
                topmargin=0.15)
            for hist in reversed(model):
                model_legend.AddEntry(hist, style='F')
            if systematics:
                model_err_band = error_band_model.Clone()
                model_err_band.linewidth = 0
                model_err_band.linecolor = 'white'
                model_err_band.fillcolor = '#858585'
                model_err_band.title = 'Uncert.'
                model_legend.AddEntry(model_err_band, style='F')
            legends.append(model_legend)
    else:
        n_entries = 1
        if signal is not None:
            n_entries += len(scaled_signal)
        if model is not None:
            n_entries += len(model)
            if systematics:
                n_entries += 1
        if legend_position == 'left':
            legend = Legend(n_entries,
                pad=hist_pad,
                leftmargin=0.05,
                rightmargin=0.46,
                margin=0.35,
                textsize=textsize,
                entrysep=0.02,
                entryheight=0.04,
                topmargin=0.15)
        else:
            legend = Legend(n_entries,
                pad=hist_pad,
                leftmargin=legend_leftmargin,
                rightmargin=0.12,
                margin=0.35,
                textsize=textsize,
                entrysep=0.02,
                entryheight=0.04,
                topmargin=0.15)
        if data is not None:
            legend.AddEntry(data, style='lep')
        if signal is not None:
            for s in reversed(scaled_signal):
                legend.AddEntry(s, style='F' if fill_signal else 'L')
        if model:
            for hist in reversed(model):
                legend.AddEntry(hist, style='F')
            model_err_band = error_band_model.Clone()
            model_err_band.linewidth = 0
            model_err_band.linecolor = 'white'
            model_err_band.fillcolor = '#858585'
            model_err_band.title = 'Uncert.'
            legend.AddEntry(model_err_band, style='F')
        legends.append(legend)

    # draw the objects
    axes, bounds = rootpy_utils.draw(
        objects,
        pad=hist_pad,
        logy=logy,
        ypadding=ypadding,
        logy_crop_value=1E-1)

    xaxis, yaxis = axes
    xmin, xmax, ymin, ymax = bounds

    # draw the legends
    hist_pad.cd()
    for legend in legends:
        legend.Draw()

    if show_ratio:
        # hide x labels on top hist
        xaxis.SetLabelOffset(100)
        base_hist = ratio_hist_tmp
        base_hist.xaxis.SetTitleOffset(
            base_hist.xaxis.GetTitleOffset() * 3)
        base_hist.xaxis.SetLabelOffset(
            base_hist.xaxis.GetLabelOffset() * 4)

    format_plot(hist_pad, template=data or model[0],
                xaxis=ratio_hist_tmp.xaxis if show_ratio else xaxis,
                yaxis=yaxis,
                xaxes=(xaxis, ratio_hist_tmp.xaxis) if show_ratio else None,
                xlabel=name, ylabel=ylabel, units=units, xlimits=range,
                left_label=category.label, right_label=plot_label,
                data_info=data_info, integer=integer)

    # draw arrows
    if arrow_values is not None:
        arrow_top = ymin + (ymax - ymin) / 2.
        hist_pad.cd()
        for value in arrow_values:
            arrow = Arrow(value, arrow_top, value, ymin, 0.05, '|>')
            arrow.SetAngle(30)
            arrow.SetLineWidth(2)
            arrow.Draw()

    if show_pvalue and data is not None and model:
        hist_pad.cd()
        total_model = sum(model)
        # show p-value and chi^2
        pvalue = total_model.Chi2Test(data, 'WW')
        pvalue_label = ROOT.TLatex(
            0.6, 0.97,
            "p-value={0:.2f}".format(pvalue))
        pvalue_label.SetNDC(True)
        pvalue_label.SetTextFont(43)
        pvalue_label.SetTextSize(16)
        pvalue_label.Draw()
        chi2 = total_model.Chi2Test(data, 'WW CHI2/NDF')
        chi2_label = ROOT.TLatex(
            0.78, 0.97,
            "#chi^{{2}}/ndf={0:.2f}".format(chi2))
        chi2_label.SetNDC(True)
        chi2_label.SetTextFont(43)
        chi2_label.SetTextSize(16)
        chi2_label.Draw()

    if top_label is not None:
        hist_pad.cd()
        label = ROOT.TLatex(
            hist_pad.GetLeftMargin() + 0.08, 0.97,
            top_label)
        label.SetNDC(True)
        label.SetTextFont(43)
        label.SetTextSize(16)
        label.Draw()

    if output_name is not None:
        # create the output filename
        filename = 'var_{0}_{1}'.format(
            category.name,
            output_name.lower().replace(' ', '_'))
        if logy:
            filename += '_logy'
        filename += '_root'

        # generate list of requested output formats
        if output_formats is None:
            output_formats = ('png',)
        elif isinstance(output_formats, basestring):
            output_formats = output_formats.split()

        # save the figure
        for format in output_formats:
            output_filename = '{0}.{1}'.format(filename, format)
            save_canvas(fig, output_dir, output_filename)

    if prev_style is not None:
        prev_style.cd()
    return fig


def plot_significance(signal, background, ax):

    if isinstance(signal, (list, tuple)):
        signal = sum(signal)
    if isinstance(background, (list, tuple)):
        background = sum(background)

    # plot the signal significance on the same axis
    sig_ax = ax.twinx()
    sig, max_sig, max_cut = significance(signal, background)
    bins = list(background.xedges())[:-1]

    log.info("Max signal significance %.2f at %.2f" % (max_sig, max_cut))

    sig_ax.plot(bins, sig, 'k--', label='Signal Significance')
    sig_ax.set_ylabel(r'$S / \sqrt{S + B}$',
            color='black', fontsize=15, position=(0., 1.), va='top', ha='right')
    #sig_ax.tick_params(axis='y', colors='red')
    sig_ax.set_ylim(0, max_sig * 2)
    plt.text(max_cut, max_sig + 0.02, '(%.2f, %.2f)' % (max_cut, max_sig),
            ha='right', va='bottom',
            axes=sig_ax)
    """
    plt.annotate('(%.2f, %.2f)' % (max_cut, max_sig), xy=(max_cut, max_sig),
            xytext=(max_cut + 0.05, max_sig),
                 arrowprops=dict(color='black', shrink=0.15),
                 ha='left', va='center', color='black')
    """


def plot_grid_scores(grid_scores, best_point, params, name,
                     label_all_bins=False,
                     label_all_ticks=False,
                     n_ticks=10,
                     title=None,
                     format='png'):

    param_names = sorted(grid_scores[0][0].keys())
    param_values = dict([(pname, []) for pname in param_names])
    for pvalues, score, cv_scores in grid_scores:
        for pname in param_names:
            param_values[pname].append(pvalues[pname])

    # remove duplicates
    for pname in param_names:
        param_values[pname] = np.unique(param_values[pname]).tolist()

    scores = np.empty(shape=[len(param_values[pname]) for pname in param_names])

    for pvalues, score, cv_scores in grid_scores:
        index = []
        for pname in param_names:
            index.append(param_values[pname].index(pvalues[pname]))
        scores.itemset(tuple(index), score)

    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = plt.axes([.12, .15, .8, .75])
    cmap = cm.get_cmap('jet', 100)
    img = ax.imshow(scores, interpolation="nearest", cmap=cmap,
            aspect='auto',
            origin='lower')

    if label_all_ticks:
        plt.xticks(range(len(param_values[param_names[1]])),
                param_values[param_names[1]])
        plt.yticks(range(len(param_values[param_names[0]])),
                param_values[param_names[0]])
    else:
        trees = param_values[param_names[1]]
        def tree_formatter(x, pos):
            if x < 0 or x >= len(trees):
                return ''
            return str(trees[int(x)])

        leaves = param_values[param_names[0]]
        def leaf_formatter(x, pos):
            if x < 0 or x >= len(leaves):
                return ''
            return '%.2f' % leaves[int(x)]

        ax.xaxis.set_major_formatter(FuncFormatter(tree_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(leaf_formatter))
        ax.xaxis.set_major_locator(MaxNLocator(n_ticks, integer=True,
            prune='lower', steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(n_ticks, integer=True,
            steps=[1, 2, 5, 10]))
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_rotation(45)

    ax.set_xlabel(params[param_names[1]], fontsize=12,
            position=(1., 0.), ha='right')
    ax.set_ylabel(params[param_names[0]], fontsize=12,
            position=(0., 1.), va='top')

    ax.set_frame_on(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    for row in range(scores.shape[0]):
        for col in range(scores.shape[1]):
            decor={}
            if ((param_values[param_names[0]].index(best_point[param_names[0]])
                 == row) and
                (param_values[param_names[1]].index(best_point[param_names[1]])
                 == col)):
                decor = dict(weight='bold',
                             bbox=dict(boxstyle="round,pad=0.5",
                                       ec='black',
                                       fill=False))
            if label_all_bins or decor:
                plt.text(col, row, "%.3f" % (scores[row][col]), ha='center',
                         va='center', **decor)
    if title:
        plt.suptitle(title)

    plt.colorbar(img, fraction=.06, pad=0.03)
    plt.axis("tight")
    plt.savefig(os.path.join(PLOTS_DIR, "grid_scores_%s.%s") % (
        name, format), bbox_inches='tight')
    plt.clf()


def hist_scores(hist, scores, systematic='NOMINAL'):

    for sample, scores_dict in scores:
        scores, weight = scores_dict[systematic]
        hist.fill_array(scores, weight)


def plot_clf(background_scores,
             category,
             signal_scores=None,
             signal_scale=1.,
             data_scores=None,
             name=None,
             draw_histograms=True,
             draw_data=False,
             save_histograms=False,
             hist_template=None,
             bins=10,
             min_score=0,
             max_score=1,
             signal_colors=cm.spring,
             systematics=None,
             unblind=False,
             **kwargs):

    if hist_template is None:
        if hasattr(bins, '__iter__'):
            # variable width bins
            hist_template = Hist(bins)
            min_score = min(bins)
            max_score = max(bins)
        else:
            hist_template = Hist(bins, min_score, max_score)

    bkg_hists = []
    for bkg, scores_dict in background_scores:
        hist = hist_template.Clone(title=bkg.label)
        scores, weight = scores_dict['NOMINAL']
        hist.fill_array(scores, weight)
        hist.decorate(**bkg.hist_decor)
        hist.systematics = {}
        for sys_term in scores_dict.keys():
            if sys_term == 'NOMINAL':
                continue
            sys_hist = hist_template.Clone()
            scores, weight = scores_dict[sys_term]
            sys_hist.fill_array(scores, weight)
            hist.systematics[sys_term] = sys_hist
        bkg_hists.append(hist)

    if signal_scores is not None:
        sig_hists = []
        for sig, scores_dict in signal_scores:
            sig_hist = hist_template.Clone(title=sig.label)
            scores, weight = scores_dict['NOMINAL']
            sig_hist.fill_array(scores, weight)
            sig_hist.decorate(**sig.hist_decor)
            sig_hist.systematics = {}
            for sys_term in scores_dict.keys():
                if sys_term == 'NOMINAL':
                    continue
                sys_hist = hist_template.Clone()
                scores, weight = scores_dict[sys_term]
                sys_hist.fill_array(scores, weight)
                sig_hist.systematics[sys_term] = sys_hist
            sig_hists.append(sig_hist)
    else:
        sig_hists = None

    if data_scores is not None and draw_data and unblind is not False:
        data, data_scores = data_scores
        if isinstance(unblind, float):
            if sig_hists is not None:
                # unblind up to `unblind` % signal efficiency
                sum_sig = sum(sig_hists)
                cut = efficiency_cut(sum_sig, 0.3)
                data_scores = data_scores[data_scores < cut]
        data_hist = hist_template.Clone(title=data.label)
        data_hist.decorate(**data.hist_decor)
        data_hist.fill_array(data_scores)
        if unblind >= 1 or unblind is True:
            log.info("Data events: %d" % sum(data_hist))
            log.info("Model events: %f" % sum(sum(bkg_hists)))
            for hist in bkg_hists:
                log.info("{0} {1}".format(hist.GetTitle(), sum(hist)))
            log.info("Data / Model: %f" % (sum(data_hist) / sum(sum(bkg_hists))))
    else:
        data_hist = None

    if draw_histograms:
        output_name = 'event_bdt_score'
        if name is not None:
            output_name += '_' + name
        for logy in (False, True):
            draw(data=data_hist,
                 model=bkg_hists,
                 signal=sig_hists,
                 signal_scale=signal_scale,
                 category=category,
                 name="BDT Score",
                 output_name=output_name,
                 show_ratio=data_hist is not None,
                 model_colors=None,
                 signal_colors=signal_colors,
                 systematics=systematics,
                 logy=logy,
                 **kwargs)
    return bkg_hists, sig_hists, data_hist


def draw_ROC(bkg_scores, sig_scores):
    # draw ROC curves for all categories
    hist_template = Hist(100, -1, 1)
    plt.figure()
    for category, (bkg_scores, sig_scores) in category_scores.items():
        bkg_hist = hist_template.Clone()
        sig_hist = hist_template.Clone()
        hist_scores(bkg_hist, bkg_scores)
        hist_scores(sig_hist, sig_scores)
        bkg_array = np.array(bkg_hist)
        sig_array = np.array(sig_hist)
        # reverse cumsum
        bkg_eff = bkg_array[::-1].cumsum()[::-1]
        sig_eff = sig_array[::-1].cumsum()[::-1]
        bkg_eff /= bkg_array.sum()
        sig_eff /= sig_array.sum()
        plt.plot(sig_eff, 1. - bkg_eff,
                 linestyle='-',
                 linewidth=2.,
                 label=category)
    plt.legend(loc='lower left')
    plt.ylabel('Background Rejection')
    plt.xlabel('Signal Efficiency')
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid()
    plt.savefig(os.path.join(PLOTS_DIR, 'ROC.png'), bbox_inches='tight')


def draw_ratio(a, b, field, category,
               textsize=22,
               ratio_range=(0,2),
               ratio_line_values=[0.5,1,1.5],
               optional_label_text=None,
               normalize=True):
    """
    Draw a canvas with two Hists normalized to unity on top
    and a ratio plot between the two hist
    Parameters:
    - a: Nominal Hist (denominator in the ratio)
    - b: Shifted Hist (numerator in the ratio)
    - field: variable field (see variables.py)
    - category: analysis category (see categories/*)
    """
    if field in VARIABLES:
        xtitle = VARIABLES[field]['root']
    else:
        xtitle = field
    plot = RatioPlot(xtitle=xtitle,
                     ytitle='{0}Events'.format(
                         'Normalized ' if normalize else ''),
                     ratio_title='A / B',
                     ratio_range=ratio_range,
                     ratio_line_values=ratio_line_values)
    if normalize:
        a_integral = a.integral()
        if a_integral != 0:
            a /= a_integral
        b_integral = b.integral()
        if b_integral != 0:
            b /= b_integral
    a.title = 'A: ' + a.title
    b.title = 'B: ' + b.title
    a.color = 'black'
    b.color = 'red'
    a.legendstyle = 'L'
    b.legendstyle = 'L'
    a.markersize = 0
    b.markersize = 0
    a.linewidth = 2
    b.linewidth = 2
    a.fillstyle = 'hollow'
    b.fillstyle = 'hollow'
    a.linestyle = 'solid'
    b.linestyle = 'dashed'
    a.drawstyle='hist E0'
    b.drawstyle='hist E0'
    plot.draw('main', [a, b], ypadding=(0.3, 0.))
    ratio = Hist.divide(a, b, fill_value=-1)
    ratio.drawstyle = 'hist'
    ratio.color = 'black'
    ratio_band = Graph(ratio, fillstyle='/', fillcolor='black', linewidth=0)
    ratio_band.drawstyle = '20'
    plot.draw('ratio', [ratio_band, ratio])
    with plot.pad('main') as pad:
        # legend
        leg = Legend([a, b],
                     leftmargin=0.25, topmargin=0.1,
                     margin=0.18, textsize=textsize)
        leg.Draw()
        # draw the category label
        label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.04, 0.87,
            category.label)
        label.SetNDC()
        label.SetTextFont(43)
        label.SetTextSize(textsize)
        label.Draw()
        # show p-value and chi^2
        pvalue = a.Chi2Test(b, 'WW')
        pvalue_label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.04, 0.8,
            "p-value={0:.2f}".format(pvalue))
        pvalue_label.SetNDC(True)
        pvalue_label.SetTextFont(43)
        pvalue_label.SetTextSize(textsize)
        pvalue_label.Draw()
        chi2 = a.Chi2Test(b, 'WW CHI2/NDF')
        chi2_label = ROOT.TLatex(
            pad.GetLeftMargin() + 0.04, 0.72,
            "#frac{{#chi^{{2}}}}{{ndf}}={0:.2f}".format(chi2))
        chi2_label.SetNDC(True)
        chi2_label.SetTextFont(43)
        chi2_label.SetTextSize(textsize)
        chi2_label.Draw()
        if optional_label_text is not None:
            optional_label = ROOT.TLatex(pad.GetLeftMargin()+0.55,0.87,
                                         optional_label_text )
            optional_label.SetNDC(True)
            optional_label.SetTextFont(43)
            optional_label.SetTextSize(textsize)
            optional_label.Draw()
    return plot


def compare(a, b, field_dict, category, region, name, year,
            path='plots/shapes', **kwargs):
    a_hists, field_scale = a.get_field_hist(field_dict, category)
    b_hists, _ = b.get_field_hist(field_dict, category)
    a.draw_array(a_hists, category, region, field_scale=field_scale)
    b.draw_array(b_hists, category, region, field_scale=field_scale)
    for field,_ in field_dict.items():
        # draw ratio plot
        a_hist = a_hists[field]
        b_hist = b_hists[field]
        plot = draw_ratio(a_hist, b_hist,
                          field, category, **kwargs)
        save_canvas(plot, path, '{0}/shape_{0}_{1}_{2}_{3}.png'.format(
            name, field, category.name, year % 1000))
