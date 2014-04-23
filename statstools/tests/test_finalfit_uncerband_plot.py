from array import array
import rootpy
import ROOT
rootpy.log.basic_config_colorized()
from rootpy.io import root_open
from rootpy.plotting import Canvas,Graph
from rootpy.plotting import set_style

from statstools.finalfit_uncertband_plot import getPostFitPlottingObjects
from statstools.finalfit_uncertband_plot import UncertGraph

set_style('ATLAS')
# ------------------------------
def Fit_WS(workspace):
    """
    Fit the WS and compute the histograms and TGraphAssymErrors
    for the final plotting drawing

    Parameters
    ----------

    workspace : RooWorkspace
        HSG4 like workspace
    """

    # --> Get the Model Config object
    mc = workspace.obj("ModelConfig")
    if not mc:
        raise RuntimeError('Could not retrieve the ModelConfig object')
    mc.GetParametersOfInterest().first().setVal(1)
#     fit_res = 0
    roo_min = workspace.fit()
    fit_res = roo_min.save()
    fit_res.Print()

    # --> Get the data distribution
    obsData = workspace.data('obsData')
    if not obsData:
        raise RuntimeError('Could not retrieve the data histograms')
    # --> Get the simultaneous PDF
    simPdf = mc.GetPdf()
    if not simPdf:
        raise RuntimeError('Could not retrieve the simultaneous pdf')

    plotting_objects = getPostFitPlottingObjects(mc,obsData,simPdf,fit_res)
    out_file = ROOT.TFile('frames.root','RECREATE')
    for obj in plotting_objects:
        obj.Write()
    out_file.Close()


# ------------------------------------------------------------------------
#  -------              MAIN DRIVER                         --------------
# ------------------------------------------------------------------------


# rfile = root_open( '../../../workspaces_comparison/workspaces/mva/ws_measurement_hh_combination_125.root')
# ws = rfile.Get('combined')
# Fit_WS(ws)

# rfile = root_open( '../../workspaces/hh_combination_mva_125_v1/ws_measurement_hh_combination_125.root')
# ws = rfile.Get( 'combined' )
# Fit_WS(ws)

# channel_name = 'channel_boosted_125'
channel_name = 'channel_vbf_125_12'

frame_file = root_open('frames.root')
frame         = frame_file.Get(channel_name)
hbkg_plus_sig = frame_file.Get('hbkg_plus_sig_'+channel_name)
hbkg          = frame_file.Get('hbkg_'+channel_name)


hbkg.SetLineColor(ROOT.kRed)

curve_uncert = frame.getCurve( 'FitError_AfterFit_Mu0' )
graph = UncertGraph(hbkg, curve_uncert)

curve_uncert_sig = frame.getCurve('FitError_AfterFit')
graph_sig = UncertGraph( hbkg_plus_sig, curve_uncert_sig )


graph.fillstyle='solid'
graph.SetFillColor(ROOT.kRed-7)
graph.SetLineColor(ROOT.kRed)
graph.SetMarkerColor(ROOT.kRed)


c = Canvas()
c.cd()
# c.SetLogy()
hbkg.Draw('HIST')
graph.Draw('sameE2')
graph_sig.Draw( 'sameE2')
hbkg.Draw('SAMEHIST')
hbkg_plus_sig.Draw('SAMEHIST')
c.SaveAs('toto.png')



