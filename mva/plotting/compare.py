from rootpy.plotting import Legend, Hist, Graph
from rootpy.plotting.style.atlas.labels import ATLAS_label

import ROOT

from ..variables import VARIABLES, HH_VARIABLES, get_label
from .templates import RatioPlot
from ..defaults import TARGET_REGION
from .. import ATLAS_LABEL
from .. import save_canvas

VARIABLES.update(HH_VARIABLES)

def draw_ratio(a, b, field, category,
               textsize=22,
               ratio_range=(0, 2),
               ratio_line_values=[0.5, 1, 1.5],
               optional_label_text=None,
               normalize=True,
               logy=False):
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
        xtitle = get_label(field) #VARIABLES[field]['root']
    else:
        xtitle = field
    plot = RatioPlot(xtitle=xtitle,
                     ytitle='{0}Events'.format(
                         'Normalized ' if normalize else ''),
                     ratio_title='A / B',
                     ratio_limits=ratio_range,
                     ratio_line_values=ratio_line_values,
                     logy=logy)
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
                     leftmargin=0.4,
                     rightmargin=0.05,
                     topmargin=0.05,
                     # 0.2, 0.2, 0.45, margin=0.35, 
                     textsize=textsize)
        leg.Draw()
        # draw the category label
        if category is not None:
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
        if ATLAS_LABEL.lower() == 'internal':
            x = 0.67
            y = 1-pad.GetTopMargin()+0.005
        else:
            x = (1. - pad.GetRightMargin() - 0.03) - len(ATLAS_LABEL) * 0.025
            y = 1-pad.GetTopMargin()+0.01
        ATLAS_label(x, y,
                    sep=0.132, pad=pad, sqrts=None,
                    text=ATLAS_LABEL,
                    textsize=textsize)
    return plot


def compare(a, b, field_dict, category, name, year,
            region_a=None, region_b=None,
            path='plots/shapes', **kwargs):
    if region_a is None:
        region_a = TARGET_REGION
    if region_b is None:
        region_b = TARGET_REGION
    a_hists, field_scale = a.get_field_hist(field_dict, category)
    b_hists, _ = b.get_field_hist(field_dict, category)
    a.draw_array(a_hists, category, region_a, field_scale=field_scale)
    b.draw_array(b_hists, category, region_b, field_scale=field_scale)
    for field in field_dict:
        # draw ratio plot
        a_hist = a_hists[field]
        b_hist = b_hists[field]
        plot = draw_ratio(a_hist, b_hist,
                          field, category, **kwargs)
        for output in ('eps', 'png'):
            save_canvas(plot, path, '{0}/shape_{0}_{1}_{2}_{3}.{4}'.format(
                name, field, category.name, year % 1000, output))
