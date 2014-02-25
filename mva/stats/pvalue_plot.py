# ROOT/rootpy imports

import ROOT
from rootpy.plotting import Canvas, Pad, Legend, Hist, Graph, get_style
from rootpy.plotting.shapes import Line
from rootpy.plotting.style.atlas.labels import ATLAS_label
from rootpy.plotting.utils import draw

# local imports
from mva.lumi import LUMI

gaussian_cdf_c = ROOT.Math.gaussian_cdf_c


def pvalue_plot(mass_points, pvalues, name='pvalue_plot', format='png'):
    """
    Draw a pvalue plot

    Parameters
    ----------

    mass_points : list
        List of mass points

    pvalues : list
        List of p-values

    name : str
        Name of output file excluding extension

    format : str or list
        Image format or list of image formats

    """
    style = get_style('ATLAS', shape='rect')
    # allow space for sigma labels on right
    style.SetPadRightMargin(0.05)

    with style:
        c = Canvas()
        c.SetLogy()
        c.cd()

        haxis = Hist(1000, -500, 500)
        xaxis = haxis.xaxis
        yaxis = haxis.yaxis
        xaxis.SetRangeUser(100, 150)
        haxis.Draw("AXIS")
        #yaxis.title = "P_{0}"
        #xaxis.title = "m_{H} [GeV]"

        g_exp = Graph(len(mass_points), linestyle='dashed', drawstyle='L')
        for idx, (mass, pvalue) in enumerate(zip(mass_points, pvalues)):
            g_exp.SetPoint(idx, mass, pvalue)
        g_exp.linestyle = 'dashed'

        #g_obs.SetLineStyle(1);
        #g_obs.SetMarkerSize(0.8);

        # automatically handles axis limits
        draw(g_exp, pad=c, same=True, logy=True,
             xtitle="m_{H} [GeV]", ytitle="P_{0}",
             xaxis=xaxis, yaxis=yaxis, ypadding=(0.2, 0.1))

        ATLAS_label(0.57, 0.88, text="Internal 2012", sqrts=8, pad=c, sep=0.09)

        # TODO:
        lumi = LUMI[2012]
        #lumi_str= "#font[42]{ #int L dt = {:1.1f} fb^{-1}}".format(lumi/1000.)
        lumi_str= '{:1.1f}'.format(lumi/1000.)

        line = Line()
        line.SetLineStyle(2)
        line.SetLineColor(2)
        latex = ROOT.TLatex()
        latex.SetNDC(False)
        latex.SetTextSize(20)
        latex.SetTextColor(2)

        # draw sigma levels up to minimum of pvalues
        min_pvalue = min(pvalues)
        sigma = 0
        while True:
            pvalue = gaussian_cdf_c(sigma)
            if pvalue < min_pvalue:
                break
            latex.DrawLatex(151, pvalue, "{0}#sigma".format(sigma))
            line.DrawLine(100, pvalue, 150, pvalue)
            sigma += 1

        c.RedrawAxis()
        c.Update()
        if isinstance(format, basestring):
            format = [format]
        for fmt in format:
            c.SaveAs('{0}.{1}'.format(name, fmt))
