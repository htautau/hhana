# ROOT/rootpy imports

import ROOT
from rootpy.plotting import Canvas, Legend, Hist, Graph
from rootpy.plotting.shapes import Line
from rootpy.plotting.style.atlas.labels import ATLAS_label
from rootpy.plotting.utils import draw
from rootpy.context import preserve_current_canvas

gaussian_cdf_c = ROOT.Math.gaussian_cdf_c


def pvalue_plot(poi, pvalues, pad=None,
                xtitle='X', ytitle='P_{0}',
                linestyle=None):
    """
    Draw a pvalue plot

    Parameters
    ----------
    poi : list
        List of POI values tested
    pvalues : list
        List of p-values or list of lists of p-values to overlay
        multiple p-value curves
    pad : Canvas or Pad, optional (default=None)
        Pad to draw onto. Create new pad if None.
    xtitle : str, optional (default='X')
        The x-axis label (POI name)
    ytitle : str, optional (default='P_{0}')
        The y-axis label
    linestyle : str or list, optional (default=None)
        Line style for the p-value graph or a list of linestyles for
        multiple p-value graphs.

    Returns
    -------
    pad : Canvas
        The pad.

    """
    if not pvalues:
        raise ValueError("pvalues is empty")
    if not poi:
        raise ValueError("poi is empty")
    # determine if pvalues is list or list of lists
    if not isinstance(pvalues[0], (list, tuple)):
        pvalues = [pvalues]

    with preserve_current_canvas():
        if pad is None:
            pad = Canvas()
        pad.cd()
        pad.SetLogy()

        # create the axis
        min_poi, max_poi = min(poi), max(poi)
        haxis = Hist(1000, min_poi, max_poi)
        xaxis = haxis.xaxis
        yaxis = haxis.yaxis
        xaxis.SetRangeUser(min_poi, max_poi)
        haxis.Draw('AXIS')

        graphs = []
        for ipv, pv in enumerate(pvalues):
            graph = Graph(len(poi), linestyle='dashed', drawstyle='L')
            for idx, (point, pvalue) in enumerate(zip(poi, pv)):
                graph.SetPoint(idx, point, pvalue)
            if linestyle:
                if isinstance(linestyle, basestring):
                    graph.linestyle = linestyle
                else:
                    graph.linestyle = linestyle[ipv]
            graphs.append(graph)

        # automatically handles axis limits
        draw(graphs, pad=pad, same=True, logy=True,
             xtitle=xtitle, ytitle=ytitle,
             xaxis=xaxis, yaxis=yaxis, ypadding=(0.2, 0.1))

        # draw sigma levels up to minimum of pvalues
        line = Line()
        line.SetLineStyle(2)
        line.SetLineColor(2)
        latex = ROOT.TLatex()
        latex.SetNDC(False)
        latex.SetTextSize(20)
        latex.SetTextColor(2)
        min_pvalue = min(pvalues)
        sigma = 0
        while True:
            pvalue = gaussian_cdf_c(sigma)
            if pvalue < min_pvalue:
                break
            latex.DrawLatex(max_poi, pvalue, " {0}#sigma".format(sigma))
            line.DrawLine(min_poi, pvalue, max_poi, pvalue)
            sigma += 1

        pad.RedrawAxis()
        pad.Update()
    return pad


if __name__ == '__main__':
    from rootpy.plotting import Canvas, Pad, Legend, Hist, Graph, get_style
    mass_points = [100,105,120,125,130,135,140,145,150]
    pvalues = [
        [0.5, 0.25, 0.15, 0.05, 0.03, 0.01, 0.03, 0.05, 0.15, 0.25, 0.5],
        [0.4, 0.3, 0.17, 0.02, 0.01, 0.008, 0.08, 0.06, 0.14, 0.2, 0.2],
    ]
    style = get_style('ATLAS', shape='rect')
    # allow space for sigma labels on right
    style.SetPadRightMargin(0.05)
    with style:
        c = Canvas()
        pvalue_plot(mass_points, pvalues, pad=c, xtitle='m_{H} [GeV]',
                    linestyle=['dashed', 'solid'])
        ATLAS_label(0.57, 0.88, text="Internal 2012", sqrts=8, pad=c, sep=0.09)
        c.SaveAs('pvalue_plot.png')
