import os
import sys
import math
import itertools

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.font_manager as fm
from matplotlib import rc
from matplotlib.ticker import (AutoMinorLocator, NullFormatter,
                               MaxNLocator, FuncFormatter, MultipleLocator)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import ROOT
from rootpy.plotting import Canvas, Pad, Legend, Hist, Hist2D, HistStack
import rootpy.plotting.root2matplotlib as rplt
from rootpy.io import root_open
from rootpy.plotting.shapes import Line
from rootpy.plotting.utils import get_limits, get_band
from rootpy.plotting.style.atlas.labels import ATLAS_label
from rootpy.plotting.contrib.quantiles import qqgraph
from rootpy.plotting.contrib import plot_corrcoef_matrix
from rootpy.memory.keepalive import keepalive
from rootpy.stats.histfactory import HistoSys, split_norm_shape

from .variables import VARIABLES
from . import PLOTS_DIR, MMC_MASS
from .systematics import iter_systematics, systematic_name
from .stats.utils import efficiency_cut, significance
from . import log; log = log[__name__]


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


def set_colours(hists, colour_map=cm.jet):

    for i, h in enumerate(hists):
        colour = colour_map((i + 1) / float(len(hists) + 1))
        h.SetColor(colour)


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
                 signal_colour_map=cm.spring,
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
                color = signal_colour_map((i + 1) / float(len(signals) + 1))
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

        for i in xrange(len(total_high)):
            total_max[i] = max(total_high[i], total_low[i], total_model[i])
            total_min[i] = min(total_high[i], total_low[i], total_model[i])

        if total_min.Integral() <= 0:
            log.warning("{0}: lower bound is non-positive".format(term))
        if total_max.Integral() <= 0:
            log.warning("{0}: upper bound is non-positive".format(term))

        var_high.append(total_max)
        var_low.append(total_min)

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
        mpl=False,
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
        field_hist = sample.get_field_hist(vars)
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
            field_hist = sample.get_field_hist(vars)
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
        data_field_hist = data.get_field_hist(vars)
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

        _range = var_info['range']
        if isinstance(_range, dict):
            _range = _range.get(category.name.upper(), _range[None])

        fig = draw(model=[m[field] for m in model_hists],
             data=data_field_hist[field] if data_field_hist else None,
             data_info=str(data_field_hist[field].datainfo) if data_field_hist else None,
             signal=[s[field] for s in signal_hists] if signal_hists else None,
             category=category,
             name=var_info['title'] if mpl else var_info['root'],
             units=var_info.get('units', None),
             range=_range,
             output_name=output_name,
             mpl=mpl,
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
        mpl=False,
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

        _range = var_info['range']
        if isinstance(_range, dict):
            _range = _range.get(category.name.upper(), _range[None])

        fig = draw_channel(field_channel[field],
             data_info=str(analysis.data.info),
             category=category,
             name=var_info['title'] if mpl else var_info['root'],
             units=var_info.get('units', None),
             range=_range,
             output_name=output_name,
             blind=blind,
             integer=var_info.get('integer', False),
             **kwargs)
        figs[field] = fig
    return field_channel, figs


def draw_channel(channel, systematics=True, fit=None, **kwargs):
    """
    Draw a HistFactory::Channel only include OverallSys systematics
    in resulting band as an illustration of the level of uncertainty
    since correlations of the NPs are not known and it is not
    possible to draw the statistically correct error band.
    """
    if fit is not None:
        log.warning("applying snapshot on channel {0}".format(channel.name))
        channel = channel.apply_snapshot(fit)
    if channel.data and channel.data.hist:
        data_hist = channel.data.hist
    else:
        data_hist = None
    model_hists = []
    signal_hists = []
    systematics_terms = {}
    sys_names = channel.sys_names()
    for sample in channel.samples:
        nominal_hist = sample.hist
        if systematics:
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
    return draw(
        data=data_hist,
        model=model_hists or None,
        signal=signal_hists or None,
        systematics=systematics_terms,
        **kwargs)


def draw(name,
         output_name,
         category,
         data=None,
         data_info=None,
         model=None,
         signal=None,
         signal_scale=1.,
         signal_on_top=False,
         signal_linestyles=None,
         show_signal_error=False,
         stacked_model=True,
         stacked_signal=True,
         units=None,
         range=None,
         plot_label=None,
         ylabel='Events',
         model_colour_map=None,
         signal_colour_map=None,
         fill_signal=False,
         blind=False,
         show_ratio=False,
         output_formats=None,
         systematics=None,
         systematics_components=None,
         integer=False,
         textsize=25,
         logy=False):

    if model is None and data is None and signal is None:
        raise ValueError(
            'at least one of model, data, or signal must be specified')

    range = model[0].bounds()

    if show_ratio and model is None:
        show_ratio = False

    if output_formats is None:
        output_formats = ('png',)
    elif isinstance(output_formats, basestring):
        output_formats = output_formats.split()
    if data is None:
        show_ratio=False

    ypadding = (.5, .0)

    if show_ratio:
        ratio_range = (0, 2)

        figheight = baseheight = 6.
        figwidth = basewidth = width

        vscale = 1.
        left_margin = 0.16
        bottom_margin = 0.16
        top_margin = 0.05

        if signal is not None and plot_signal_significance:
            right_margin = 0.10
        else:
            right_margin = 0.04
        ratio_sep_margin = 0.030

        width = 1. - right_margin - left_margin
        height = 1. - top_margin - bottom_margin

        ratio_abs_height = 1.975
        hist_abs_height = 6.

        figheight += ratio_abs_height + ratio_sep_margin
        vscale = figheight / baseheight
        bottom_margin /= vscale
        top_margin /= vscale
        ratio_height = ratio_abs_height / figheight
        hist_height = (hist_abs_height / figheight
                       - top_margin - bottom_margin)
        rect_hist = [
            left_margin, bottom_margin + ratio_height + ratio_sep_margin,
            width, hist_height]
        rect_ratio = [left_margin, bottom_margin, width, ratio_height]
        fig = Canvas(width=int(figwidth * 100), height=int(figheight * 100))
        fig.SetLeftMargin(0)
        fig.SetBottomMargin(0)
        fig.SetRightMargin(0)
        fig.SetTopMargin(0)
        hist_pad = Pad(0., rect_hist[1] * 0.9, 1., 1., name='top', title='top')
        hist_pad.SetBottomMargin(0.03)
        hist_pad.SetLeftMargin(rect_hist[0])
        hist_pad.SetRightMargin(1. - rect_hist[2] - rect_hist[0])
        hist_pad.SetTopMargin(1.01 - rect_hist[3] - rect_hist[1])
        hist_pad.Draw()
    else:
        fig = Canvas()
        hist_pad = fig

    if logy:
        hist_pad.SetLogy()

    if model is not None and model_colour_map is not None:
        set_colours(model, model_colour_map)

    if signal is not None:
        # always make signal a list
        if not isinstance(signal, (list, tuple)):
            signal = [signal]
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
        if signal_colour_map is not None:
            set_colours(scaled_signal, signal_colour_map)
        for i, s in enumerate(scaled_signal):
            if fill_signal:
                s.fillstyle = 'solid'
                s.fillcolor = s.linecolor
                s.linewidth = 0
                s.linestyle = 'solid'
                alpha = .75
            else:
                s.fillstyle = 'hollow'
                s.linewidth = 2
                if signal_linestyles is not None:
                    s.linestyle = signal_linestyles[i]
                else:
                    s.linestyle = 'solid'
                alpha = 1.

    ymin, ymax = float('-inf'), float('inf')

    if model is not None:
        # plot model stack with ROOT
        hist_pad.cd()
        model_stack = HistStack()
        for hist in model:
            hist.SetLineWidth(0)
            hist.drawstyle = 'hist'
            model_stack.Add(hist)
        if signal is not None and signal_on_top:
            for s in scaled_signal:
                s.drawstyle = 'hist'
                model_stack.Add(s)
        model_stack.Draw()
        _, _, ymin, ymax = get_limits(model_stack,
                logy=logy,
                ypadding=ypadding,
                logy_crop_value=1E-1)
        model_stack.SetMinimum(ymin)
        model_stack.SetMaximum(ymax)
        model_stack.Draw()
        #ratio_abs_height / figheight
        #model_stack.GetXaxis().SetTickLength(
        #    model_stack.GetXaxis().GetTickLength() * figheight / hist_abs_height)
        if integer:
            model_stack.GetXaxis().SetNdivisions(7, True)
        else:
            model_stack.GetXaxis().SetNdivisions(507, True)
        model_stack.yaxis.SetLimits(ymin, ymax)
        model_stack.yaxis.SetRangeUser(ymin, ymax)

    if signal is not None and not signal_on_top:
        hist_pad.cd()
        signal_stack = HistStack()
        for hist in scaled_signal:
            hist.drawstyle = 'hist'
            hist.fillstyle = 'hollow'
            hist.linewidth = 4
            signal_stack.Add(hist)
        if stacked_signal:
            signal_stack.Draw('SAME')
            _, _, _ymin, _ymax = get_limits(signal_stack,
                logy=logy,
                ypadding=ypadding,
                logy_crop_value=1E-1)
            model_stack.SetMinimum(ymin)
            model_stack.SetMaximum(ymax)
            signal_stack.SetMinimum(ymin)
            signal_stack.SetMaximum(ymax)
            signal_stack.Draw('SAME')
            signal_stack.yaxis.SetLimits(ymin, ymax)
            signal_stack.yaxis.SetRangeUser(ymin, ymax)
        else:
            for h in signal_stack:
                h.Draw('SAME')

    if systematics:
        if model is not None:
            # draw systematics band
            total_model, high_band_model, low_band_model = uncertainty_band(
                    model, systematics, systematics_components)
            high = total_model + high_band_model
            low = total_model - low_band_model
            hist_pad.cd()
            error_band_model = get_band(
                low,
                high,
                middle_hist=total_model)
            error_band_model.fillstyle = '/'
            error_band_model.fillcolor = '#858585'
            error_band_model.Draw('same e2')

        if signal is not None and show_signal_error:
            total_signal, high_band_signal, low_band_signal = uncertainty_band(
                signal, systematics, systematics_components)
            high = (total_signal + high_band_signal) * signal_scale
            low = (total_signal - low_band_signal) * signal_scale
            if signal_on_top:
                high += total_model
                low += total_model
            hist_pad.cd()
            error_band_signal = get_band(
                low,
                high,
                middle_hist=total_signal * signal_scale)
            error_band_signal.fillstyle = '\\'
            error_band_signal.fillcolor = '#858585'
            error_band_signal.Draw('same e2')
            signal_stack.Draw('SAME')

    if data is not None and blind is not True:
        if isinstance(blind, tuple):
            low, high = blind
            # zero out bins in blind region
            for bin in data.bins():
                if (low < bin.x.high <= high or low <= bin.x.low < high):
                    data[bin.idx] = (0., 0.)
        # draw data
        hist_pad.cd()
        _, _, _ymin, _ymax = get_limits(data,
            logy=logy,
            ypadding=ypadding,
            logy_crop_value=1E-1)
        if _ymin < ymin:
            ymin = _ymin
        if _ymax > ymax:
            ymax = _ymax
        data.SetMinimum(ymin)
        data.SetMaximum(ymax)
        data.linewidth = 2
        data.Draw('same E1')
        data.yaxis.SetLimits(ymin, ymax)
        data.yaxis.SetRangeUser(ymin, ymax)

        # draw ratio plot
        if model is not None and show_ratio:
            total_model = sum(model)
            ratio_hist = Hist.divide(data, total_model, option='B')
            # remove bins where data is zero
            for bin in data.bins():
                if bin.value == 0:
                    ratio_hist[bin.idx] = (-1, 0)
            ratio_hist.linecolor = 'black'
            ratio_hist.linewidth = 2
            ratio_hist.fillstyle = 'hollow'

            fig.cd()
            ratio_pad = Pad(
                0, 0, 1, (rect_ratio[1] + rect_ratio[3]) * 0.9,
                name='ratio', title='ratio')
            ratio_pad.SetBottomMargin(rect_ratio[1] * 3)
            ratio_pad.SetLeftMargin(rect_ratio[0])
            ratio_pad.SetRightMargin(1. - rect_ratio[2] - rect_ratio[0])
            ratio_pad.SetTopMargin(0.04)
            ratio_pad.Draw()
            ratio_pad.cd()

            # draw empty copy of ratio_hist first so lines will show
            ratio_hist_tmp = ratio_hist.Clone()
            ratio_hist_tmp.Reset()
            ratio_hist_tmp.Draw()

            ratio_hist_tmp.yaxis.SetLimits(*ratio_range)
            ratio_hist_tmp.yaxis.SetRangeUser(*ratio_range)
            ratio_hist_tmp.yaxis.SetTitle('Data / Model')
            ratio_hist_tmp.yaxis.SetNdivisions(4)
            ratio_hist_tmp.xaxis.SetLimits(*ratio_hist.bounds())
            ratio_hist_tmp.xaxis.SetRangeUser(*ratio_hist.bounds())
            ratio_hist_tmp.xaxis.SetTickLength(
                ratio_hist_tmp.xaxis.GetTickLength() * 2)

            # draw horizontal lines
            line = Line(ratio_hist.lowerbound(), 1,
                        ratio_hist.upperbound(), 1)
            line.linestyle = 'dashed'
            line.linewidth = 2
            line.Draw()

            line_up = Line(ratio_hist.lowerbound(), 1.50,
                            ratio_hist.upperbound(), 1.50)
            line_up.linestyle = 'dashed'
            line_up.linewidth = 2
            line_up.Draw()

            line_dn = Line(ratio_hist.lowerbound(), 0.50,
                            ratio_hist.upperbound(), 0.50)
            line_dn.linestyle = 'dashed'
            line_dn.linewidth = 2
            line_dn.Draw()

            # draw band below points
            if systematics:
                # plot band on ratio plot
                ratio_hist_high = Hist.divide(
                    total_model + high_band_model, total_model, option='B')
                ratio_hist_low = Hist.divide(
                    total_model - low_band_model, total_model, option='B')

                ratio_pad.cd()
                error_band = get_band(ratio_hist_high,
                                        ratio_hist_low)
                error_band.fillstyle = '/'
                error_band.fillcolor = '#858585'
                error_band.Draw('same E2')

            # draw points above band
            ratio_pad.cd()
            ratio_hist.Draw('same E0')

    if model is not None:
        hist_pad.cd()
        n_entries = len(model)
        if systematics:
            n_entries += 1
        model_legend = Legend(n_entries,
            pad=hist_pad,
            leftmargin=0.02,
            rightmargin=0.5,
            margin=0.45,
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
            model_err_band.title = 'stat #oplus sys uncert'
            model_legend.AddEntry(model_err_band, style='F')
        model_legend.Draw()

    hist_pad.cd()
    right_legend = Legend(len(signal) + 1 if signal is not None else 1,
        pad=hist_pad,
        leftmargin=0.3,
        rightmargin=0.12,
        margin=0.3,
        textsize=textsize,
        entrysep=0.02,
        entryheight=0.04,
        topmargin=0.15)
    right_legend.AddEntry(data, style='lep')
    if signal is not None:
        for s in reversed(scaled_signal):
            right_legend.AddEntry(s, style='F' if fill_signal else 'L')

    right_legend.Draw()

    if data is not None:
        binw = list(data.xwidth())
    elif model is not None:
        binw = list(model[0].xwidth())
    else:
        if isinstance(signal, (list, tuple)):
            binw = list(signal[0].xwidth())
        else:
            binw = list(signal.xwidth())
    binwidths = list(set(['%.2g' % w for w in binw]))

    if units is not None:
        label = '%s [%s]' % (name, units)
        if len(binwidths) == 1:
            # constant width bins
            ylabel = '%s / %s [%s]' % (ylabel, binwidths[0], units)
    else:
        label = name
        if len(binwidths) == 1:
            ylabel = '%s / %s' % (ylabel, binwidths[0])

    model_stack.yaxis.SetTitle(ylabel)
    base_hist = model_stack
    if show_ratio:
        # hide x labels on top hist
        model_stack.xaxis.SetLabelOffset(100)
        base_hist = ratio_hist_tmp
        base_hist.xaxis.SetTitleOffset(
            base_hist.xaxis.GetTitleOffset() * 3)
        base_hist.xaxis.SetLabelOffset(
            base_hist.xaxis.GetLabelOffset() * 4)
    base_hist.xaxis.SetTitle(label)

    if range is not None:
        model_stack.xaxis.SetLimits(*range)
        model_stack.xaxis.SetRangeUser(*range)

    filename = os.path.join(PLOTS_DIR,
        'var_%s_%s' %
        (category.name,
         output_name.lower().replace(' ', '_')))

    if logy:
        filename += '_logy'

    filename += '_root'

    hist_pad.cd()
    label = ROOT.TLatex(
        hist_pad.GetLeftMargin() + 0.03, 0.89,
        category.root_label)
    label.SetNDC()
    label.SetTextFont(43)
    label.SetTextSize(textsize)
    label.Draw()
    keepalive(hist_pad, label)

    if data_info is not None:
        plabel = ROOT.TLatex(
            hist_pad.GetLeftMargin() + 0.03, 0.82,
            data_info)
        plabel.SetNDC()
        plabel.SetTextFont(43)
        plabel.SetTextSize(textsize)
        plabel.Draw()
        keepalive(hist_pad, plabel)

    ATLAS_label(0.65, 0.89,
        sep=0.14, pad=hist_pad, sqrts=None,
        text="Internal", textsize=textsize)

    hist_pad.Update()
    hist_pad.Modified()
    hist_pad.RedrawAxis()

    for format in output_formats:
        output_filename = '%s.%s' % (filename, format)
        fig.SaveAs(output_filename)

    return fig


def draw_tmp(name,
         output_name,
         category,
         data=None,
         data_info=None,
         model=None,
         signal=None,
         signal_scale=1.,
         signal_on_top=False,
         show_signal_error=False,
         stacked_model=True,
         stacked_signal=True,
         plot_signal_significance=True,
         units=None,
         range=None,
         plot_label=None,
         ylabel='Events',
         model_colour_map=None,
         signal_colour_map=None,
         fill_signal=False,
         blind=False,
         show_ratio=False,
         show_qq=False,
         output_formats=None,
         systematics=None,
         systematics_components=None,
         mpl=False,
         width=8.,
         integer=False,
         logy=False,
         xtick_formatter=None,
         xtick_locator=None,
         xtick_rotation=None):

    if model is None and data is None and signal is None:
        raise ValueError(
            'at least one of model, data, or signal must be specified')

    range = model[0].bounds()

    root = not mpl

    if root:
        # not supported for now
        plot_signal_significance = False

    if show_ratio and model is None:
        show_ratio = False

    if show_qq and model is None:
        show_qq = False

    if output_formats is None:
        output_formats = ('png',)
    elif isinstance(output_formats, str):
        output_formats = output_formats.split(',')
    if data is None:
        show_ratio=False
        show_qq=False

    ratio_range = (0, 2)

    figheight = baseheight = 6.
    figwidth = basewidth = width

    vscale = 1.
    left_margin = 0.16
    bottom_margin = 0.16
    top_margin = 0.05

    if signal is not None and plot_signal_significance:
        right_margin = 0.10
    else:
        right_margin = 0.04
    ratio_sep_margin = 0.030

    ypadding = (.5, .05)

    width = 1. - right_margin - left_margin
    height = 1. - top_margin - bottom_margin

    ratio_abs_height = 1.975
    qq_abs_height = 6.
    hist_abs_height = 6.

    if show_ratio and not show_qq:
        figheight += ratio_abs_height + ratio_sep_margin

        vscale = figheight / baseheight
        bottom_margin /= vscale
        top_margin /= vscale

        ratio_height = ratio_abs_height / figheight
        hist_height = (hist_abs_height / figheight
                       - top_margin - bottom_margin)

        rect_hist = [left_margin,
                     bottom_margin + ratio_height + ratio_sep_margin,
                     width, hist_height]
        rect_ratio = [left_margin, bottom_margin, width, ratio_height]

    elif show_qq and not show_ratio:
        figheight += qq_abs_height

        vscale = figheight / baseheight
        bottom_margin /= vscale
        top_margin /= vscale

        gap = bottom_margin
        qq_height = qq_abs_height / figheight - gap - top_margin
        hist_height = hist_abs_height / figheight - bottom_margin

        rect_qq = [left_margin, bottom_margin + hist_height + gap, width, qq_height]
        rect_hist = [left_margin, bottom_margin, width, hist_height]

    elif show_ratio and show_qq:
        figheight += ratio_abs_height + qq_abs_height

        vscale = figheight / baseheight
        bottom_margin /= vscale
        top_margin /= vscale

        ratio_height = ratio_abs_height / figheight

        gap = bottom_margin
        qq_height = qq_abs_height / figheight - gap - top_margin
        hist_height = hist_abs_height / figheight - bottom_margin

        rect_qq = [left_margin, bottom_margin + ratio_height + hist_height + gap, width, qq_height]
        rect_hist = [left_margin, bottom_margin + ratio_height, width, hist_height]
        rect_ratio = [left_margin, bottom_margin, width, ratio_height]

    else:
        rect_hist = [left_margin, bottom_margin, width, height]

    if root:
        fig = Canvas(width=int(figwidth * 100), height=int(figheight * 100))
        fig.SetLeftMargin(0)
        fig.SetBottomMargin(0)
        fig.SetRightMargin(0)
        fig.SetTopMargin(0)
        hist_pad = Pad(0., rect_hist[1] * 0.9, 1., 1., name='top', title='top')
        hist_pad.SetBottomMargin(0.03)
        hist_pad.SetLeftMargin(rect_hist[0])
        hist_pad.SetRightMargin(1. - rect_hist[2] - rect_hist[0])
        hist_pad.SetTopMargin(1.01 - rect_hist[3] - rect_hist[1])
        if logy:
            hist_pad.SetLogy()
        hist_pad.Draw()
    else:
        fig = plt.figure(figsize=(figwidth, figheight), dpi=100)
        prop = fm.FontProperties(size=14)
        hist_ax = plt.axes(rect_hist)
        if logy:
            hist_ax.set_yscale('log', nonposy='clip')
            bottom = 1E-1
        else:
            bottom = 0

    if model is not None and model_colour_map is not None:
        set_colours(model, model_colour_map)

    if signal is not None:
        # always make signal a list
        if not isinstance(signal, (list, tuple)):
            signal = [signal]
        if signal_scale != 1.:
            scaled_signal = []
            for sig in signal:
                scaled_h = sig * signal_scale
                if root:
                    scaled_h.SetTitle(r'%s (#times %g)' % (
                        sig.GetTitle(),
                        signal_scale))
                else:
                    scaled_h.SetTitle(r'%s ($\times\/%g$)' % (
                        sig.GetTitle(),
                        signal_scale))
                scaled_signal.append(scaled_h)
        else:
            scaled_signal = signal
        if signal_colour_map is not None:
            set_colours(scaled_signal, signal_colour_map)
        for s in scaled_signal:
            if fill_signal:
                s.fillstyle = 'solid'
                s.fillcolor = s.linecolor
                s.linewidth = 0
                s.linestyle = 'solid'
                alpha = .75
            else:
                s.fillstyle = 'hollow'
                s.linewidth = 2
                s.linestyle = 'dashed'
                alpha = 1.

    ymin, ymax = float('-inf'), float('inf')

    if model is not None:
        if root:
            # plot model stack with ROOT
            hist_pad.cd()
            model_stack = HistStack()
            for hist in model:
                hist.SetLineWidth(0)
                hist.drawstyle = 'hist'
                model_stack.Add(hist)
            if signal is not None and signal_on_top:
                for s in scaled_signal:
                    s.drawstyle = 'hist'
                    model_stack.Add(s)
            model_stack.Draw()
            _, _, ymin, ymax = get_limits(model_stack,
                    logy=logy,
                    ypadding=ypadding)
            model_stack.SetMinimum(ymin)
            model_stack.SetMaximum(ymax)
            model_stack.Draw()
            ratio_abs_height / figheight
            model_stack.GetXaxis().SetTickLength(
                model_stack.GetXaxis().GetTickLength() * figheight / hist_abs_height)
            model_stack.yaxis.SetLimits(ymin, ymax)
            model_stack.yaxis.SetRangeUser(ymin, ymax)
        else:
            model_bars = rplt.bar(
                model + scaled_signal if (
                    signal is not None and signal_on_top)
                else model,
                linewidth=0,
                stacked=True,
                yerr='quadratic' if not systematics else False,
                axes=hist_ax,
                ypadding=ypadding)

            if signal is not None and signal_on_top:
                signal_bars = model_bars[len(model):]
                model_bars = model_bars[:len(model)]

    if signal is not None and not signal_on_top:
        if root:
            hist_pad.cd()
            signal_stack = HistStack()
            for hist in scaled_signal:
                hist.drawstyle = 'hist'
                hist.fillstyle = 'hollow'
                hist.linewidth = 4
                signal_stack.Add(hist)

            if stacked_signal:
                signal_stack.Draw('SAME')
                _, _, _ymin, _ymax = get_limits(signal_stack,
                    logy=logy,
                    ypadding=ypadding)
                model_stack.SetMinimum(ymin)
                model_stack.SetMaximum(ymax)
                signal_stack.SetMinimum(ymin)
                signal_stack.SetMaximum(ymax)
                signal_stack.Draw('SAME')
                signal_stack.yaxis.SetLimits(ymin, ymax)
                signal_stack.yaxis.SetRangeUser(ymin, ymax)

            else:
                for h in signal_stack:
                    h.Draw('SAME')

        else:
            if fill_signal:
                signal_bars = rplt.bar(
                    scaled_signal,
                    stacked=True, #yerr='quadratic',
                    axes=hist_ax,
                    alpha=alpha,
                    ypadding=ypadding)
            else:
                signal_bars = rplt.hist(
                    scaled_signal,
                    stacked=True,
                    alpha=alpha,
                    axes=hist_ax,
                    ypadding=ypadding,
                    zorder=1000)
            if plot_signal_significance:
                plot_significance(signal, model, ax=hist_ax)

    if model is not None and show_qq:
        qq_ax = plt.axes(rect_qq)
        gg_graph = qqgraph(data, sum(model))
        gg_graph.SetTitle('QQ plot')
        y = np.array(list(gg_graph.y()))
        y_up = y + np.array(list(gg_graph.yerrh()))
        y_low = y - np.array(list(gg_graph.yerrl()))
        f = qq_ax.fill_between(
            list(gg_graph.x()),
            y_low,
            y_up,
            interpolate=True,
            facecolor='green',
            linewidth=0,
            label='68% CL band')
        #l = qq_ax.plot(xrange(-10, 10), xrange(-10, 10), 'b--')[0]
        diag = [max(gg_graph.xedgesl(0), min(y)),
                max(gg_graph.xedgesh(-1), max(y))]
        l = Line2D(diag, diag, color='b', linestyle='--')
        qq_ax.add_line(l)
        p, _, _ = rplt.errorbar(gg_graph, axes=qq_ax, snap_zero=False,
                                xerr=False, yerr=False)
        qq_ax.set_ylabel('Model', position=(0., 1.), va='top')
        qq_ax.set_xlabel('Data', position=(1., 0.), ha='right')
        leg = qq_ax.legend([p, Patch(facecolor='green', linewidth=0), l],
                           ['QQ plot', '68% CL band', 'Diagonal'],
                           loc='lower right', prop=prop)
        frame = leg.get_frame()
        frame.set_linewidth(0)
        qq_ax.set_xlim((gg_graph.xedgesl(0), gg_graph.xedgesh(-1)))
        qq_ax.set_ylim((min(y_low), max(y_up)))

    if systematics:
        if model is not None:
            # draw systematics band
            total_model, high_band_model, low_band_model = uncertainty_band(
                    model, systematics, systematics_components)
            high = total_model + high_band_model
            low = total_model - low_band_model
            if root:
                hist_pad.cd()
                error_band_model = get_band(
                    low,
                    high,
                    middle_hist=total_model)
                error_band_model.fillstyle = '/'
                error_band_model.fillcolor = '#cccccc'
                error_band_model.Draw('same e2')

            else:
                # draw band as hatched histogram with base of model - low_band
                # and height of high_band + low_band
                rplt.fill_between(
                    high,
                    low,
                    edgecolor='0.75',
                    linewidth=0,
                    facecolor=(0,0,0,0),
                    hatch='////',
                    axes=hist_ax,
                    zorder=3000)

        if signal is not None and show_signal_error:
            total_signal, high_band_signal, low_band_signal = uncertainty_band(
                signal, systematics, systematics_components)
            high = (total_signal + high_band_signal) * signal_scale
            low = (total_signal - low_band_signal) * signal_scale
            if signal_on_top:
                high += total_model
                low += total_model
            if root:
                hist_pad.cd()
                error_band_signal = get_band(
                    low,
                    high,
                    middle_hist=total_signal * signal_scale)
                error_band_signal.fillstyle = '\\'
                error_band_signal.fillcolor = '#cccccc'
                error_band_signal.Draw('same e2')
                signal_stack.Draw('SAME')
            else:
                rplt.fill_between(
                    high,
                    low,
                    edgecolor='0.75',
                    linewidth=0,
                    facecolor=(0,0,0,0),
                    hatch=r'\\\\',
                    axes=hist_ax,
                    zorder=4000)

    if data is not None and blind is not True:
        if isinstance(blind, tuple):
            low, high = blind
            # zero out bins in blind region
            for bin in data.bins():
                if (low < bin.x.high <= high or low <= bin.x.low < high):
                    data[bin.idx] = (0., 0.)
        # draw data
        if root:
            hist_pad.cd()
            _, _, _ymin, _ymax = get_limits(data,
                    logy=logy,
                    ypadding=ypadding)
            if _ymin < ymin:
                ymin = _ymin
            if _ymax > ymax:
                ymax = _ymax
            data.SetMinimum(ymin)
            data.SetMaximum(ymax)
            data.linewidth = 2
            data.Draw('same E1')
            data.yaxis.SetLimits(ymin, ymax)
            data.yaxis.SetRangeUser(ymin, ymax)

        else:
            data_bars = rplt.errorbar(data,
                fmt='o', axes=hist_ax,
                ypadding=ypadding,
                emptybins=False,
                barsabove=True,
                zorder=5000)

        # draw ratio plot
        if model is not None and show_ratio:
            total_model = sum(model)
            ratio_hist = Hist.divide(data, total_model, option='B')
            # remove bins where data is zero
            for bin in data.bins():
                if bin.value == 0:
                    ratio_hist[bin.idx] = (-1, 0)
            ratio_hist.linecolor = 'black'
            ratio_hist.linewidth = 2
            ratio_hist.fillstyle = 'hollow'

            if root:
                fig.cd()
                ratio_pad = Pad(
                    0, 0, 1, (rect_ratio[1] + rect_ratio[3]) * 0.9,
                    name='ratio', title='ratio')
                ratio_pad.SetBottomMargin(rect_ratio[1] * 3)
                ratio_pad.SetLeftMargin(rect_ratio[0])
                ratio_pad.SetRightMargin(1. - rect_ratio[2] - rect_ratio[0])
                ratio_pad.SetTopMargin(0.04)
                ratio_pad.Draw()
                ratio_pad.cd()

                # draw empty copy of ratio_hist first so lines will show
                ratio_hist_tmp = ratio_hist.Clone()
                ratio_hist_tmp.Reset()
                ratio_hist_tmp.Draw()

                ratio_hist_tmp.yaxis.SetLimits(*ratio_range)
                ratio_hist_tmp.yaxis.SetRangeUser(*ratio_range)
                ratio_hist_tmp.yaxis.SetTitle('Data / Model')
                ratio_hist_tmp.yaxis.SetNdivisions(4)
                ratio_hist_tmp.xaxis.SetLimits(*ratio_hist.bounds())
                ratio_hist_tmp.xaxis.SetRangeUser(*ratio_hist.bounds())
                ratio_hist_tmp.xaxis.SetTickLength(
                    ratio_hist_tmp.xaxis.GetTickLength() * 2)

                # draw horizontal lines
                line = Line(ratio_hist.lowerbound(), 1,
                            ratio_hist.upperbound(), 1)
                line.linestyle = 'dashed'
                line.linewidth = 2
                line.Draw()

                line_up = Line(ratio_hist.lowerbound(), 1.50,
                               ratio_hist.upperbound(), 1.50)
                line_up.linestyle = 'dashed'
                line_up.linewidth = 2
                line_up.Draw()

                line_dn = Line(ratio_hist.lowerbound(), 0.50,
                               ratio_hist.upperbound(), 0.50)
                line_dn.linestyle = 'dashed'
                line_dn.linewidth = 2
                line_dn.Draw()

            else:
                ratio_ax = plt.axes(rect_ratio)
                ratio_ax.axhline(y=1, color='black', linestyle=':',
                                 linewidth=1.5)
                ratio_ax.axhline(y=1.5, color='black', linestyle=':',
                                 linewidth=1.5)
                ratio_ax.axhline(y=0.5, color='black', linestyle=':',
                                 linewidth=1.5)
                ratio_ax.set_ylim(ratio_range)
                ratio_ax.set_xlim(hist_ax.get_xlim())
                #ratio_ax.yaxis.tick_right()
                ratio_ax.set_ylabel(r'Data / Model',
                        position=(0., 1.), va='center', ha='right')

            # draw band below points
            if systematics:
                # plot band on ratio plot
                ratio_hist_high = Hist.divide(
                    total_model + high_band_model, total_model, option='B')
                ratio_hist_low = Hist.divide(
                    total_model - low_band_model, total_model, option='B')

                if root:
                    ratio_pad.cd()
                    error_band = get_band(ratio_hist_high,
                                          ratio_hist_low)
                    error_band.fillstyle = '/'
                    error_band.fillcolor = '#7a7a7a'
                    error_band.Draw('same E2')

                else:
                    rplt.fill_between(
                        ratio_hist_high,
                        ratio_hist_low,
                        edgecolor='0.75',
                        linewidth=0,
                        facecolor=(0,0,0,0),
                        hatch='////',
                        axes=ratio_ax)

            # draw points above band
            if root:
                ratio_pad.cd()
                ratio_hist.Draw('same E0')

            else:
                rplt.step(ratio_hist, axes=ratio_ax)

    if model is not None:
        if root:
            hist_pad.cd()
            n_entries = len(model)
            if systematics:
                n_entries += 1
            model_legend = Legend(n_entries,
                pad=hist_pad,
                leftmargin=0.02,
                rightmargin=0.5,
                margin=0.3,
                textsize=20,
                entrysep=0.02,
                entryheight=0.045,
                topmargin=0.18 if data_info else 0.1)
            for hist in reversed(model):
                model_legend.AddEntry(hist, style='F')
            if systematics:
                model_err_band = error_band_model.Clone()
                model_err_band.linewidth = 0
                #model_err_band.linecolor = 'white'
                model_err_band.fillcolor = 'black'
                model_err_band.title = 'stat #oplus sys'
                model_legend.AddEntry(model_err_band, style='F')
            model_legend.Draw()
        else:
            model_legend = hist_ax.legend(
                reversed(model_bars), [h.title for h in reversed(model)],
                prop=prop,
                title=(category.label + '\n' + plot_label
                    if plot_label else category.label),
                loc='upper left')
            format_legend(model_legend)

    if root:
        hist_pad.cd()
        right_legend = Legend(len(signal) + 1 if signal is not None else 1,
            pad=hist_pad,
            leftmargin=0.4,
            rightmargin=0.12,
            margin=0.3,
            textsize=20,
            entrysep=0.02,
            entryheight=0.045,
            topmargin=0.1)
        right_legend.AddEntry(data, style='lep')
        if signal is not None:
            for s in reversed(scaled_signal):
                right_legend.AddEntry(s, style='F')
        right_legend.Draw()
    else:
        right_legend_bars = []
        right_legend_titles =[]

        if signal is not None:
            right_legend_bars += signal_bars
            right_legend_titles += [s.title for s in scaled_signal]

        if data is not None:
            right_legend_bars.append(data_bars)
            right_legend_titles.append(data.title)

        if right_legend_bars:
            right_legend = hist_ax.legend(
                right_legend_bars[::-1],
                right_legend_titles[::-1],
                prop=prop,
                loc='upper right')
            format_legend(right_legend)
            if model is not None:
                # re-add model legend
                hist_ax.add_artist(model_legend)

    if units is not None:
        label = '%s [%s]' % (name, units)
        if data is not None:
            binw = list(data.xwidth())
        elif model is not None:
            binw = list(model[0].xwidth())
        else:
            if isinstance(signal, (list, tuple)):
                binw = list(signal[0].xwidth())
            else:
                binw = list(signal.xwidth())
        binwidths = list(set(['%.3g' % w for w in binw]))
        if len(binwidths) == 1:
            # constant width bins
            ylabel = '%s / %s [%s]' % (ylabel, binwidths[0], units)
    else:
        label = name

    if root:
        model_stack.yaxis.SetTitle(ylabel)
        base_hist = model_stack
        if show_ratio:
            # hide x labels on top hist
            model_stack.xaxis.SetLabelOffset(100)
            base_hist = ratio_hist_tmp
            base_hist.xaxis.SetTitleOffset(
                base_hist.xaxis.GetTitleOffset() * 3)
            base_hist.xaxis.SetLabelOffset(
                base_hist.xaxis.GetLabelOffset() * 4)
        base_hist.xaxis.SetTitle(label)

    else:
        hist_ax.set_ylabel(ylabel, position=(0., 1.), va='center', ha='right')
        base_ax = hist_ax
        if show_ratio:
            base_ax = ratio_ax
        base_ax.set_xlabel(label, position=(1., 0.), ha='right')
        root_axes(hist_ax,
                  xtick_formatter=xtick_formatter,
                  logy=logy,
                  integer=integer,
                  no_xlabels=show_ratio,
                  vscale=1.5 if not (show_ratio or show_qq) else 1.,
                  xtick_rotation=xtick_rotation)
        if show_ratio:
            root_axes(ratio_ax,
                      xtick_formatter=xtick_formatter,
                      integer=integer,
                      vscale=1 if show_qq else vscale * .4,
                      xtick_rotation=xtick_rotation)
        if show_qq:
            root_axes(qq_ax,
                      vscale=vscale)

    if range is not None:
        if root:
            model_stack.xaxis.SetLimits(*range)
            model_stack.xaxis.SetRangeUser(*range)
        else:
            hist_ax.set_xlim(range)
        if show_ratio:
            if root:
                pass
            else:
                ratio_ax.set_xlim(range)

    filename = os.path.join(PLOTS_DIR,
        'var_%s_%s' %
        (category.name,
         output_name.lower().replace(' ', '_')))

    if logy:
        filename += '_logy'

    if mpl:
        filename += '_mpl'
    else:
        filename += '_root'

    if root:
        hist_pad.cd()
        label = ROOT.TLatex(rect_hist[0] + 0.03, 0.9, category.root_label)
        label.SetNDC()
        label.SetTextFont(43)
        label.SetTextSize(20)
        label.Draw()
        keepalive(hist_pad, label)

        if data_info is not None:
            plabel = ROOT.TLatex(rect_hist[0] + 0.03, 0.83, data_info)
            plabel.SetNDC()
            plabel.SetTextFont(43)
            plabel.SetTextSize(20)
            plabel.Draw()
            keepalive(hist_pad, plabel)

        ATLAS_label(0.75, 0.9, sep=0.09, pad=hist_pad, sqrts=None, text="Internal")

        hist_pad.Update()
        hist_pad.Modified()
        hist_pad.RedrawAxis()

    for format in output_formats:
        output_filename = '%s.%s' % (filename, format)
        if root:
            fig.SaveAs(output_filename)
        else:
            log.info("writing %s" % output_filename)
            plt.savefig(output_filename)

    if mpl:
        plt.close(fig)
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
            return str(leaves[int(x)])

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
             signal_colour_map=cm.spring,
             plot_signal_significance=True,
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
                 plot_signal_significance=plot_signal_significance,
                 category=category,
                 name="BDT Score",
                 output_name=output_name,
                 show_ratio=data_hist is not None,
                 model_colour_map=None,
                 signal_colour_map=signal_colour_map,
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
