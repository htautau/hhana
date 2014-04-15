# ROOT/rootpy imports

import ROOT
from rootpy.plotting import Canvas, Legend, Hist, Graph
from rootpy.plotting.shapes import Line
from rootpy.plotting.utils import draw
from rootpy.memory import keepalive
from rootpy.context import preserve_current_canvas

gaussian_cdf_c = ROOT.Math.gaussian_cdf_c


def pvalue_plot(poi, pvalues, pad=None,
                xtitle='X', ytitle='P_{0}',
                linestyle=None,
                linecolor=None):
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
    linecolor : str or list, optional (default=None)
        Line color for the p-value graph or a list of linestyles for
        multiple p-value graphs.

    Returns
    -------
    pad : Canvas
        The pad.
    graphs : list of Graph
        The p-value graphs

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

        min_pvalue = float('inf')
        graphs = []
        for ipv, pv in enumerate(pvalues):
            graph = Graph(len(poi), linestyle='dashed', drawstyle='L', linewidth=2)
            
            for idx, (point, pvalue) in enumerate(zip(poi, pv)):
                graph.SetPoint(idx, point, pvalue)
            # ---> Set the line style
            if linestyle:
                if isinstance(linestyle, basestring):
                    graph.linestyle = linestyle
                else:
                    graph.linestyle = linestyle[ipv]
            # ---> Set the line color
            if linecolor:
                if isinstance(linecolor, basestring):
                    graph.linecolor = linecolor
                else:
                    graph.linecolor = linecolor[ipv]

            graphs.append(graph)
            curr_min_pvalue = min(pv)
            if curr_min_pvalue < min_pvalue:
                min_pvalue = curr_min_pvalue

        # automatically handles axis limits
        draw(graphs, pad=pad, same=True, logy=True,
             xtitle=xtitle, ytitle=ytitle,
             xaxis=xaxis, yaxis=yaxis, ypadding=(0.2, 0.1),
             logy_crop_value=1E-300)

        # draw sigma levels up to minimum of pvalues
        line = Line()
        line.SetLineStyle(2)
        line.SetLineColor(2)
        latex = ROOT.TLatex()
        latex.SetNDC(False)
        latex.SetTextSize(20)
        latex.SetTextColor(2)
        sigma = 0
        while True:
            pvalue = gaussian_cdf_c(sigma)
            if pvalue < min_pvalue:
                break
            keepalive(pad, latex.DrawLatex(max_poi, pvalue, " {0}#sigma".format(sigma)))
            keepalive(pad, line.DrawLine(min_poi, pvalue, max_poi, pvalue))
            sigma += 1

        pad.RedrawAxis()
        pad.Update()
    return pad, graphs


if __name__ == '__main__':
    from rootpy.plotting import Canvas, Legend, get_style
    from rootpy.plotting.style.atlas.labels import ATLAS_label

    mass_points = [100,105,120,125,130,135,140,145,150]
    pvalues = [
        [0.5, 0.25, 0.15, 0.05, 0.03, 0.01, 0.03, 0.05, 0.15, 0.25, 0.5],
        [0.4, 0.3, 0.17, 0.02, 0.01, 0.008, 0.08, 0.06, 0.14, 0.2, 0.2],
    ]
    names = ['A', 'B']
    style = get_style('ATLAS', shape='rect')
    # allow space for sigma labels on right
    style.SetPadRightMargin(0.05)
    with style:
        c = Canvas()
        _, graphs = pvalue_plot(
            mass_points, pvalues, pad=c, xtitle='m_{H} [GeV]',
            linestyle=['dashed', 'solid'])
        for name, graph in zip(names, graphs):
            graph.title = name
            graph.legendstyle = 'L'
        leg = Legend(graphs, leftmargin=0.4, topmargin=0.2)
        leg.Draw()
        ATLAS_label(0.57, 0.88, text="Internal 2012", sqrts=8, pad=c, sep=0.09)
        c.SaveAs('pvalue_plot.png')
