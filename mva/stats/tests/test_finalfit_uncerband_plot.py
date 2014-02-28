from array import array
import rootpy
import ROOT
rootpy.log.basic_config_colorized()
from rootpy.interactive import wait
from rootpy.io import root_open
from rootpy.plotting import Canvas,Graph

from mva.stats.finalfit_uncertband_plot import Fit_WS
from mva.stats.finalfit_uncertband_plot import UncertGraph

ROOT.gROOT.SetBatch(0)

# rfile = root_open( '../../../workspaces_comparison/workspaces/mva/ws_measurement_hh_combination_125.root')
# ws = rfile.Get('combined')

# Fit_WS(ws)


frame_file = root_open( 'frames.root' )
frame = frame_file.Get( 'channel_vbf_125' )

h_Z       = frame_file.Get( 'channel_vbf_125_L_x_Ztautau_channel_vbf_125_overallSyst_x_StatUncert__obs_x_channel_vbf_125' )
h_QCD     = frame_file.Get( 'channel_vbf_125_L_x_QCD_channel_vbf_125_overallSyst_x_StatUncert__obs_x_channel_vbf_125' )
h_Other   = frame_file.Get( 'channel_vbf_125_L_x_Others_channel_vbf_125_overallSyst_x_StatUncert__obs_x_channel_vbf_125' )
h_Sig_gg  = frame_file.Get( 'channel_vbf_125_L_x_Signal_gg_125_channel_vbf_125_overallSyst_x_HistSyst__obs_x_channel_vbf_125' )
h_Sig_vbf = frame_file.Get( 'channel_vbf_125_L_x_Signal_VBF_125_channel_vbf_125_overallSyst_x_HistSyst__obs_x_channel_vbf_125' )
h_Sig_Z   = frame_file.Get( 'channel_vbf_125_L_x_Signal_Z_125_channel_vbf_125_overallSyst_x_HistSyst__obs_x_channel_vbf_125' )
h_Sig_W   = frame_file.Get( 'channel_vbf_125_L_x_Signal_W_125_channel_vbf_125_overallSyst_x_HistSyst__obs_x_channel_vbf_125' )

hbkg = h_Z.Clone("h_bkg")
hbkg.Add(h_QCD)
hbkg.Add(h_Other)

hbkg_plus_sig = hbkg.Clone("h_bkg_plus_sig")
hbkg_plus_sig.Add( h_Sig_gg  )
hbkg_plus_sig.Add( h_Sig_vbf )
hbkg_plus_sig.Add( h_Sig_Z   )
hbkg_plus_sig.Add( h_Sig_W   )





curve_uncert = frame.getCurve( 'FitError_AfterFit_Mu0' )
graph = UncertGraph( hbkg_plus_sig, curve_uncert )

curve_uncert_sig = frame.getCurve( 'FitError_AfterFit' )
graph_sig = UncertGraph( hbkg_plus_sig, curve_uncert_sig )

c = Canvas()
c.cd()
c.SetLogy()
graph.Draw('AP2')
graph_sig.Draw( 'sameP2')

c1 = Canvas()
c1.cd()
c1.SetLogy()
hbkg.Draw('HIST')
hbkg_plus_sig.Draw('sameHIST')

# wait()


