# ROOT/rootpy imports

import ROOT
from rootpy.plotting import Canvas, Pad, Legend, Hist
from rootpy.plotting.shapes import Line
from rootpy.plotting.style.atlas.labels import ATLAS_label
from rootpy.plotting.style import get_style,set_style

style = get_style('ATLAS')
style.SetPadLeftMargin(0.20)
style.SetPadRightMargin(0.05)
set_style( style )
# local imports
from mva.lumi import LUMI

def pvalue_plot(mass_points,exp_list):
    # def pvalue_plot(h_exp,h_obs):

    g_exp = ROOT.TGraph()
    for mass in mass_points:
        g_exp.SetPoint(g_exp.GetN(),mass,exp_list[mass])
    g_exp.SetLineStyle(2)
#     g_obs.SetLineStyle(1);
#     g_obs.SetMarkerSize(0.8);


    haxis = ROOT.TH1D("axis","axis",1000,-500,500)
    haxis.GetXaxis().SetRangeUser(100,150)
    c = Canvas(500,500,name="pvalue",title="pvalue")
    c.cd()
    haxis.Draw("AXIS")
    haxis.GetXaxis().SetRangeUser(100,150)
    haxis.GetXaxis().SetRangeUser(100,150)
    haxis.GetYaxis().SetRangeUser(10E-5,100)
    haxis.GetYaxis().SetTitle("P_{0}")
    haxis.GetXaxis().SetTitle("m_{H} [GeV]")
  
    g_exp.Draw("sameL")
    
    line = Line()
    line.SetLineStyle(2)
    line.SetLineColor(2)
    line.DrawLine(100,ROOT.Math.gaussian_cdf_c(0), 150, ROOT.Math.gaussian_cdf_c(0)  )
    line.DrawLine(100,ROOT.Math.gaussian_cdf_c(1), 150, ROOT.Math.gaussian_cdf_c(1)  )
    line.DrawLine(100,ROOT.Math.gaussian_cdf_c(2), 150, ROOT.Math.gaussian_cdf_c(2)  )
    line.DrawLine(100,ROOT.Math.gaussian_cdf_c(3), 150, ROOT.Math.gaussian_cdf_c(3)  )
    # line.DrawLine(100,ROOT.Math.gaussian_cdf_c(4), 150, ROOT.Math.gaussian_cdf_c(4)  )
    # line.DrawLine(100,ROOT.Math.gaussian_cdf_c(5), 150, ROOT.Math.gaussian_cdf_c(5)  )

    ATLAS_label(0.2, 0.9, text="Internal 2012", sqrts=8,pad=c)

    lumi = LUMI[2012]
#     lumi_str= "#font[42]{ #int L dt = {:1.1f} fb^{-1}}".format(lumi/1000.)
    lumi_str= '{:1.1f}'.format(lumi/1000.)

#     latex.SetNDC()
#     latex.SetTextSize(0.03)
#     latex.SetTextSize(0.027)
#     #         latex.DrawLatex(0.8,0.9 ,"#tau_{h} + #tau_{h}"); 
#     latex.SetTextSize(0.03)
#     latex.DrawLatex(0.2,0.9 ,"#font[72]{ATLAS}#font[42]{  Internal} ")
#     latex.DrawLatex(0.2,0.85 ,lumi_str)
#     latex.DrawLatex(0.2,0.80 ,"#sqrt{#it{s}} = 8 TeV ")
    
    
    latex = ROOT.TLatex(145,ROOT.Math.gaussian_cdf_c(1),"1#sigma")
    latex.SetNDC(False)
    latex.SetTextSize(0.02)
    latex.SetTextColor(2)
    latex.Draw()
#     latex.DrawLatex(152,ROOT.Math.gaussian_cdf_c(1),"1#sigma")
#     latex.DrawLatex(152,ROOT.Math.gaussian_cdf_c(2),"2#sigma")
#     latex.DrawLatex(152,ROOT.Math.gaussian_cdf_c(3),"3#sigma")
    #   latex.DrawLatex(152,ROOT.Math.gaussian_cdf_c(4),"4#sigma");
    
    c.SetLogy()
    c.RedrawAxis()
    c.Update()
    c.Print('toto.png')
#     return c
