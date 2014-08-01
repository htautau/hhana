import ROOT
from rootpy.plotting.style.atlas.labels import ATLAS_label
from rootpy.memory.keepalive import keepalive
from .. import ATLAS_LABEL


def label_plot(pad, template, xaxis, yaxis,
               ylabel='Events', xlabel=None,
               units=None, data_info=None,
               category_label=None,
               right_label=None,
               atlas_label=None,
               textsize=22):

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

    # draw the category label
    if category_label:
        label = ROOT.TLatex(
            1. - pad.GetRightMargin(),
            1. - pad.GetTopMargin() + 0.02,
            category_label)
        label.SetNDC()
        label.SetTextFont(43)
        label.SetTextSize(textsize)
        label.SetTextAlign(31)
        with pad:
            label.Draw()
        keepalive(pad, label)

    # draw the right label
    if right_label is not None:
        label = ROOT.TLatex(0.68, 0.81, right_label)
        label.SetNDC()
        label.SetTextFont(43)
        label.SetTextSize(textsize)
        with pad:
            label.Draw()
        keepalive(pad, label)

    # draw the luminosity label
    if data_info is not None:
        plabel = ROOT.TLatex(
            pad.GetLeftMargin() + 0.03, 0.88,
            str(data_info))
        plabel.SetNDC()
        plabel.SetTextFont(43)
        plabel.SetTextSize(textsize)
        with pad:
            plabel.Draw()
        keepalive(pad, plabel)

    # draw the ATLAS label
    if atlas_label is not False:
        label = atlas_label or ATLAS_LABEL
        if label.lower() == 'internal':
            x = 0.67
        else:
            x = (1. - pad.GetRightMargin() - 0.03) - len(label) * 0.025
        ATLAS_label(x, 0.88,
                    sep=0.132, pad=pad, sqrts=None,
                    text=label,
                    textsize=textsize)

    pad.Update()
    pad.Modified()
